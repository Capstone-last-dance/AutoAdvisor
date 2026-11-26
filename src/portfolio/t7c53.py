r"""run_weekly 5b단계 (선택 단계): B-09 병행 모델 T7-80-C53의 forward 목표 비중 (B 소유, PM 11/20 동의).

설계는 docs/design_B.md 3.7절, 규칙은 b09-explore `docs/b09_explore.md` 26절 1~7이다 (고정 커밋 6d9ec49).
D의 run_weekly가 M8(5단계) 뒤에 `src.portfolio.t7c53:step`을 부르고, forward_test.record가 공식 기록과 같은
ft-{t} 커밋·태그에 `positions/wf_{YYYYMMDD}_B_t7c53.parquet`을 넣는다 (D PR #77).

규칙 (forward)
- 시작: splits.forward_test.ml_signals_from(2026-11-27). 그 전 신호일에는 쓰지 않는다 (11/6~11/20은 baseline 신호).
- 순위: 종목마다 rank_score를 그 종목의 최근 4개 ML 신호일(t 포함, 시작일부터) 평균으로 바꾸고 U_t 안에서 다시 매긴다
  (동점은 종목코드 순). 지난 ML 신호는 게시된 ft-* 태그에서 읽는다 (D past_tables).
- 격주: 시작일을 0번으로 신호일(주마다 마지막 거래일)을 센다. 기록이 빠진 주도 센다. 짝수 번째가 리밸런싱 주다.
- 실제 보유 H: 지난 병행 목표를 현금 100%에서 C 엔진(사전등록 비용, M9 청산)에 차례로 넣어 t 종가 보유를 만든다.
  기록이 없는 주는 리밸런싱 없이 직전 보유를 유지한다 (weekly.replay_holdings와 같다).
- 리밸런싱 주: 버퍼 규칙(H 중 rank ≤ 140 유지, 80종목까지 rank ≤ 80을 더함, 같은 비중) → crash_flag·M9 쿨다운·U_t 밖
  차단 → 종목 10%·업종 30% 한도(잘린 몫은 현금) → 모든 종목에 E = 0.466107을 곱한다.
- 사이 주: 목표 = t 종가의 실제 보유 비중. 차단 종목만 0으로 하고 한도 넘는 몫만 줄인다. E를 다시 곱하지 않는다.
- 기록: positions 9.3절 열 (constructor = buffer, exposure = 0.466107, regime_mc = M8 Step 5 값), CASH 행 포함, 합 1.
- 같은 입력이면 백테스트 규칙(`t7rl.build_positions`에 E = 0.466107)과 같은 positions를 낸다 (tests/portfolio/test_t7c53.py).
- 선택 단계라 실패해도 공식 기록은 그대로다 (run_weekly). 입력 표가 스텁이면 쓰지 않고 건너뛴다.
- 이번 실행에서 파일을 썼을 때만 ctx.memory["parallel_written"]에 저장소 상대 경로를 넣는다. 7단계는 이 표시가
  있을 때만 병행 파일을 첨부하면 된다 (PM #77 리뷰 11/21: 건너뛴 실행에 남아 있던 같은 날짜 파일이 붙지 않게).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio import t7rl, weekly
from src.portfolio.buffer_rule import buffer_rule_weights
from src.portfolio.engine_adapter import engine_inputs
from src.portfolio.params import BufferRuleParams, RiskLayerParams
from src.portfolio.risk_layer import check_invariants, compute_regime_mc, positions_frame, risk_layer_weights

NAME = "t7c53"
E = t7rl.E_C53
CONSTRUCTOR = "buffer"
# D의 함수 (PR #77). "모듈:함수"로 연결한다 (R-17). 테스트는 함수를 직접 넣는다
PAST_SIGNALS_ENTRY: Any = "src.pipeline.forward_test:past_tables"  # (t, "signals") → {신호일: signals 표}
PAST_PARALLEL_ENTRY: Any = "src.pipeline.forward_test:past_parallel_positions"  # (t) → {신호일: 병행 positions}
WRITTEN_KEY = "parallel_written"  # ctx.memory: 이번 실행에서 쓴 병행 파일 (저장소 상대 경로)


def run_id(date: str) -> str:
    """병행 기록 run_id (D forward_test.parallel_run_id와 같다)."""
    return f"{date.replace('-', '')}_B_{NAME}"


def first_date(cfg: Mapping[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(cfg["splits"]["forward_test"]["ml_signals_from"])


def smoothed_signals(current: pd.DataFrame, past: Mapping[str, pd.DataFrame], start: pd.Timestamp) -> pd.DataFrame:
    """t의 signals 행에 4주 평균 순위를 넣는다. past 중 start 이후의 ML 신호(model_version ≠ baseline)만 평균에 쓴다."""
    if (current["model_version"] == "baseline").any():
        raise ValueError("신호일 t의 signals가 ML 신호가 아니다 (baseline). 병행 기록은 ML 신호부터다")
    t = pd.Timestamp(pd.to_datetime(current["date"]).iloc[0])
    used = [df for d, df in sorted(past.items())
            if start <= pd.Timestamp(d) < t and not (df["model_version"] == "baseline").any()]
    allsig = pd.concat([*used, current], ignore_index=True)
    allsig["date"] = pd.to_datetime(allsig["date"])
    sm = t7rl.smooth_signals(allsig)
    return sm.loc[sm["date"] == t].reset_index(drop=True)


def replay_weights(engine: Any, recorded: Mapping[str, pd.DataFrame], dates: Sequence[str], date: str) -> dict[str, float]:
    """weekly.replay_holdings와 같은 재생이고, 보유 종목 대신 date 종가의 보유 비중을 돌려준다 (사이 주 목표)."""
    extra = sorted(set(recorded) - set(dates))
    if extra:
        raise ValueError(f"병행 기록이 시작일 ~ {date} 전 신호일이 아니다: {extra}")
    kept = [d for d in dates if d in recorded]
    if not kept:
        return {}
    run = getattr(engine, "run_to_signal_close", None)
    if run is None:
        raise RuntimeError("엔진에 run_to_signal_close가 없다 (C PR #22가 필요하다)")
    state = engine.initial_state()
    for d, nxt in zip(kept, [*kept[1:], date]):
        res = run(state, weekly._targets(recorded[d], d), d, nxt)
        state = res.state
    return {str(k): float(v) for k, v in res.weights_obs.items() if v > 0.0}


def parallel_positions(inputs: weekly.ForwardInputs, held: Mapping[str, float], rebalance: bool,
                       buffer: BufferRuleParams, risk: RiskLayerParams) -> pd.DataFrame:
    """규칙대로 t의 positions를 만든다. held = t 종가 실제 보유 비중."""
    universe = set(inputs.universe)
    if rebalance:
        w_raw = buffer_rule_weights(set(held), inputs.signals, inputs.universe, inputs.cooldown, buffer)
        e = E
    else:
        w_raw = pd.Series({k: v for k, v in held.items() if k in universe}, dtype="float64")
        e = 1.0  # 이미 E가 곱해진 실제 비중
    w_raw = w_raw[w_raw > 0.0]
    tickers = [str(x) for x in w_raw.index]
    crash = dict(zip(inputs.signals["ticker"].astype(str), inputs.signals["crash_flag"]))
    blocked = np.array([(tk not in universe) or crash.get(tk, 0) == 1 or (tk in inputs.cooldown) for tk in tickers], dtype=bool)
    labels = np.array([inputs.sectors.get(tk, risk.unknown_sector_label) for tk in tickers], dtype=object)
    codes = pd.factorize(labels)[0].astype(np.int64)
    w = risk_layer_weights(w_raw.to_numpy(dtype=float), blocked, codes, e, risk, "full")
    cash = check_invariants(w, codes, risk, "full")
    targets = {tk: float(x) for tk, x in zip(tickers, w) if x > 0.0}
    regime = compute_regime_mc(inputs.date, bm_close=inputs.bm_close, params=risk)
    return positions_frame(inputs.date, targets, cash, CONSTRUCTOR, E, regime)


def step(ctx) -> None:
    """5b단계. 신호일 t(시작일부터)의 T7-80-C53 목표 비중을 positions/wf_{YYYYMMDD}_B_t7c53.parquet에 쓴다."""
    cfg, buffer0, risk = weekly._load_params()
    t, start = ctx.date, first_date(cfg)
    ctx.memory.pop(WRITTEN_KEY, None)
    if pd.Timestamp(t) < start:
        ctx.log.info("5b단계 B 병행: %s는 시작일(%s) 전이라 쓰지 않는다", t, start.date())
        return
    stub = weekly._stub_inputs(ctx, (*weekly.SIGNAL_INPUTS, *weekly.ENGINE_INPUTS))
    if stub:
        ctx.log.warning("5b단계 B 병행: 입력 표가 스텁이라(%s) 쓰지 않는다", ", ".join(stub))
        return
    read_signals = weekly._resolve(PAST_SIGNALS_ENTRY, "D 지난 신호")
    if read_signals is None:
        raise RuntimeError("지난 신호를 읽는 D 함수가 연결되지 않았다 (t7c53.PAST_SIGNALS_ENTRY)")
    sig = smoothed_signals(ctx.read_table("signals", date=t), read_signals(t, "signals"), start)
    cooldown = weekly._call_m9(ctx, weekly.M9_COOLDOWN_ENTRY, "쿨다운", t)
    inputs = weekly.forward_inputs(t, sig, ctx.read_table("universe"), ctx.read_table("sector"), ctx.read_table("index"),
                                   () if cooldown is None else cooldown)
    dates = weekly.signal_dates_between(inputs.bm_close.index, start, t)
    rebalance = len(dates) % t7rl.EVERY == 0
    held = _replayed_weights(ctx, cfg, dates, t)
    positions = parallel_positions(inputs, held, rebalance, t7rl.t7_buffer(buffer0), risk)
    path = ctx.write_table("positions", positions, run_id=run_id(t))
    ctx.memory[WRITTEN_KEY] = f"positions/wf_{run_id(t)}.parquet"
    cash = float(positions.loc[positions["ticker"] == "CASH", "target_weight"].iloc[0])
    ctx.log.info("5b단계 B 병행: %s %s주 (시작일부터 %d번째), 보유 재생 %d종목 → %d종목, 현금 %.4f",
                 path, "리밸런싱" if rebalance else "사이", len(dates), len(held), len(positions) - 1, cash)


def _replayed_weights(ctx, cfg: Mapping[str, Any], dates: list[str], t: str) -> dict[str, float]:
    """시작일 ~ t 전 신호일의 병행 기록을 재생한 t 종가 보유 비중 (첫 신호일은 현금 100%)."""
    if not dates:
        return {}
    read = weekly._resolve(PAST_PARALLEL_ENTRY, "D 지난 병행 기록")
    if read is None:
        raise RuntimeError("지난 병행 기록을 읽는 D 함수가 연결되지 않았다 (t7c53.PAST_PARALLEL_ENTRY)")
    recorded = read(t)
    if not recorded:
        ctx.log.warning("5b단계 B 병행: 지난 신호일 %s의 병행 기록이 하나도 없어 현금 100%%에서 시작한다", ", ".join(dates))
        return {}
    missing = [d for d in dates if d not in recorded]
    if missing:
        ctx.log.warning("5b단계 B 병행: 기록이 없는 신호일 %s는 리밸런싱 없이 직전 보유를 유지한 것으로 재생한다",
                        ", ".join(missing))
    maps = weekly._call_m9(ctx, weekly.M9_ENGINE_ENTRY, "체결 지도", dates[0], t)
    cooldowns, events = (None, None) if maps is None else maps
    einputs = engine_inputs(*(ctx.read_table(name) for name in weekly.ENGINE_INPUTS))
    engine = weekly.build_engine(einputs, cfg, cooldowns=cooldowns, event_liquidations=events)
    return replay_weights(engine, recorded, dates, t)
