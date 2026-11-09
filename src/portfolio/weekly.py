"""run_weekly 4·5단계: forward test의 M4·M8 (5.2절, 6.8절, B 소유).

- m4_step(ctx): 신호일 t의 원시 목표 비중 w_raw(종목코드 → 비중 Series)를 ctx.memory["w_raw"]에 넣는다.
- m8_step(ctx): w_raw를 M8에 넣고 positions/{t}.parquet을 ctx.write_table로 쓴다.

규칙 (docs/design_B.md 5.4절)
- splits.forward_test.positions_from(2026-11-20) 전 신호일에는 positions를 쓰지 않는다 (6.8절, D-04가 거부한다).
- 구성기는 RL 최종 정책이 나오기 전까지 버퍼 규칙이다 (constructor = "buffer").
- 버퍼 규칙 단계 1의 H는 실제 보유다. 백테스트(run_buffer_rule)가 직전 체결 뒤 실제 보유를 쓰는 것과 같게,
  positions_from부터 기록된 목표 비중을 현금 100%에서 C 엔진에 차례로 넣어(run_to_signal_close) t 종가의
  보유를 만든다. positions는 목표라 정지·상하한가·M9 청산이 있으면 실제 보유와 다르기 때문이다.
- 기록이 없는 주(스텁 등으로 커밋되지 않은 주)는 리밸런싱 없이 직전 보유를 유지한 것으로 재생하고 경고한다.
  C의 사후 성과(evaluate_forward_record, PR #22)도 기록된 주만 이어 재생한다.
- 다른 역할의 함수는 아래 *_ENTRY에 "모듈:함수"로 연결한다 (R-17). 형태는 interfaces.py에 있다.
  - M9(PM, PR #34)가 없으면 쿨다운·M9 청산 없이 계산하고 기록한다. '없음'은 연결 전, 함수를 찾지 못함(PR 미병합),
    경보 파일 없음(FileNotFoundError)이다. 경고와 ctx.memory["deviations"]에 남기고 PM이 deviation log에 적는다
    (PM 11/9 결정). RECORD_WITHOUT_M9 = False면 스텁 실행으로 표시해 커밋을 막는다. M9의 다른 오류는 그대로 멈춘다.
  - 지난 기록이 필요한데(positions_from 다음 신호일부터) 읽는 함수(D)를 찾지 못하면 실패한다.
- 앞 단계가 B 입력 표를 스텁으로 썼으면 B 출력도 스텁이다 (빈 w_raw, 0행 positions).
"""
from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.portfolio.buffer_rule import buffer_rule_weights
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.interfaces import ForwardEngine
from src.portfolio.params import BufferRuleParams, RiskLayerParams, load_prereg
from src.portfolio.risk_layer import CASH, apply_risk_layer, sectors_at

CONSTRUCTOR = "buffer"  # RL 최종 정책이 나오면 바꾼다 (design_B.md 5.4절)

# 다른 역할의 함수 (R-17). 담당자가 이름을 알려 주면 "모듈:함수"로 채운다
M9_COOLDOWN_ENTRY: str | None = "src.events.alerts:cooldown_at"  # PM: CooldownFunction (PR #34)
M9_ENGINE_ENTRY: str | None = "src.events.alerts:engine_alerts"  # PM: EngineAlertsFunction (PR #34)
RECORDED_POSITIONS_ENTRY: str | None = "src.pipeline.forward_test:past_positions"  # D: RecordedPositionsFunction (PR #31)

# PM 11/9 결정: M9가 연결되지 않아도 기록을 미루지 않는다. 쿨다운·M9 청산 없이 계산하고 deviation으로 남긴다.
# False면 예전처럼 스텁 실행으로 표시해 7단계가 forward 커밋을 하지 않는다 (6.12절)
RECORD_WITHOUT_M9 = True

SIGNAL_INPUTS = ("signals", "universe", "sector", "index")
ENGINE_INPUTS = ("prices", "status", "index", "delisted", "delisted_reused")
MEMORY_KEY = "b_forward"  # 4단계가 5단계에 넘기는 ForwardInputs
DEVIATIONS_KEY = "deviations"  # ctx.memory: 이번 실행의 deviation log 기록 대상 (문장 목록)


@dataclass(frozen=True)
class ForwardInputs:
    """신호일 t에 알 수 있는 버퍼 규칙·M8 입력 (R-03)."""

    date: str
    signals: pd.DataFrame  # ticker, rank, crash_flag (U_t 종목 전부)
    universe: list[str]  # U_t
    sectors: dict[str, str]  # t일 유효 업종
    bm_close: pd.Series  # t일까지 코스피 200 종가
    cooldown: frozenset[str]  # M9: t+1 진입 금지


def forward_inputs(
    date: str,
    signals: pd.DataFrame,
    universe: pd.DataFrame,
    sector: pd.DataFrame,
    index: pd.DataFrame,
    cooldown: Iterable[str],
) -> ForwardInputs:
    """9.3절 표에서 신호일 t의 입력을 고른다. signals 종목은 U_t와 같아야 한다 (rl_data.build_snapshots와 같은 검사)."""
    t = pd.Timestamp(date)
    if (pd.to_datetime(signals["date"]) != t).any():
        raise ValueError(f"signals에 신호일 {date}가 아닌 행이 있다 (R-03)")
    u = universe.loc[pd.to_datetime(universe["date"]) == t]
    if u.empty:
        raise ValueError(f"universe에 신호일 {date}의 행이 없다")
    members = sorted(u.loc[u["in_universe"].astype(bool), "ticker"].astype(str))
    s = signals[["ticker", "rank", "crash_flag"]].astype({"ticker": str}).reset_index(drop=True)
    if s["ticker"].duplicated().any() or set(s["ticker"]) != set(members):
        raise ValueError(f"{date}: signals 종목과 U_t가 다르다")
    k = index.loc[index["index_code"] == "KOSPI200"]
    bm = pd.Series(k["close"].to_numpy(dtype="float64"), index=pd.to_datetime(k["date"])).sort_index().loc[:t]
    return ForwardInputs(date, s, members, sectors_at(sector, t), bm, frozenset(str(x) for x in cooldown))


def signal_dates_between(trading_days: Iterable[Any], start: Any, end: Any) -> list[str]:
    """start ≤ 신호일 < end인 신호일(주마다 마지막 거래일, calendar.signal_day). end 주까지의 거래일만 넘긴다."""
    days = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(trading_days)))))
    days = days[days <= pd.Timestamp(end)]
    iso = days.isocalendar()
    last = pd.Series(days, index=days).groupby([iso["year"].to_numpy(), iso["week"].to_numpy()]).max()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return [d.strftime("%Y-%m-%d") for d in sorted(last) if lo <= d < hi]


def _targets(positions: pd.DataFrame, date: str) -> dict[str, float]:
    if (pd.to_datetime(positions["date"]) != pd.Timestamp(date)).any():
        raise ValueError(f"{date} forward 기록의 positions에 다른 날짜 행이 있다")
    rows = positions.loc[(positions["ticker"] != CASH) & (positions["target_weight"] > 0.0)]
    if rows["ticker"].duplicated().any():
        raise ValueError(f"{date} forward 기록의 positions에 같은 종목이 두 번 있다")
    return {str(t): float(w) for t, w in zip(rows["ticker"], rows["target_weight"])}


def replay_holdings(
    engine: ForwardEngine, recorded: Mapping[str, pd.DataFrame], dates: Sequence[str], date: str
) -> set[str]:
    """dates(오름차순, positions_from ~ date 전 신호일) 중 기록된 날의 목표 비중을 현금 100%에서 차례로 체결하고
    date 종가의 보유 종목을 돌려준다. 기록이 없는 신호일에는 리밸런싱하지 않고 직전 보유를 유지한다
    (C evaluate_forward_record와 같다). 신호일이 아닌 날의 기록이 있으면 멈춘다."""
    extra = sorted(set(recorded) - set(dates))
    if extra:
        raise ValueError(f"forward 기록이 positions_from ~ {date} 전 신호일이 아니다: {extra}")
    kept = [d for d in dates if d in recorded]
    if not kept:
        return set()
    run = getattr(engine, "run_to_signal_close", None)
    if run is None:
        raise RuntimeError("엔진에 run_to_signal_close가 없다 (C PR #22가 필요하다)")
    state = engine.initial_state()
    for d, nxt in zip(kept, [*kept[1:], date]):
        res = run(state, _targets(recorded[d], d), d, nxt)
        state = res.state
    return {str(t) for t, w in res.weights_obs.items() if w > 0.0}


def buffer_w_raw(inputs: ForwardInputs, holdings: Iterable[str], buffer: BufferRuleParams) -> pd.Series:
    """버퍼 규칙 단계 1~4 (백테스트와 같은 buffer_rule_weights)."""
    return buffer_rule_weights(holdings, inputs.signals, inputs.universe, inputs.cooldown, buffer)


def risk_layer_positions(
    inputs: ForwardInputs, w_raw: pd.Series, risk: RiskLayerParams, constructor: str = CONSTRUCTOR
) -> pd.DataFrame:
    """M8 (6.9절)을 거친 positions 행. 불변식을 어기면 RiskLayerViolation (G0)."""
    return apply_risk_layer(
        w_raw, inputs.signals, inputs.sectors, inputs.bm_close, inputs.cooldown, inputs.universe, inputs.date,
        risk, constructor,
    )


# ---------------------------------------------------------------------------
# run_weekly 단계 함수 (fn(ctx) -> None)
# ---------------------------------------------------------------------------


def m4_step(ctx) -> None:
    """5.2절 4단계. ctx.memory["w_raw"]에 원시 목표 비중을 넣는다."""
    cfg, buffer, _ = _load_params()
    t, start = ctx.date, _positions_from(cfg)
    ctx.memory["w_raw"] = pd.Series(dtype="float64")
    ctx.memory.pop(MEMORY_KEY, None)
    if pd.Timestamp(t) < start:
        ctx.log.info("4단계 B: %s는 positions 기록 전(%s부터, 6.8절)이라 비중을 만들지 않는다", t, start.date())
        return
    if _stub_inputs(ctx, SIGNAL_INPUTS):
        ctx.log.warning("4단계 B: 입력 표가 스텁이라(%s) w_raw도 비운다", ", ".join(_stub_inputs(ctx, SIGNAL_INPUTS)))
        return
    cooldown = _call_m9(ctx, M9_COOLDOWN_ENTRY, "쿨다운", t)
    inputs = forward_inputs(
        t, ctx.read_table("signals", date=t), ctx.read_table("universe"), ctx.read_table("sector"),
        ctx.read_table("index"), () if cooldown is None else cooldown,
    )
    dates = signal_dates_between(inputs.bm_close.index, start, t)
    holdings = _replayed_holdings(ctx, cfg, dates, t) if dates else set()
    w_raw = buffer_w_raw(inputs, holdings, buffer)
    ctx.memory["w_raw"], ctx.memory[MEMORY_KEY] = w_raw, inputs
    ctx.log.info(
        "4단계 B: 지난 신호일 %d개를 재생한 보유 %d종목, 쿨다운 %d종목 → w_raw %d종목 (%s)",
        len(dates), len(holdings), len(inputs.cooldown), len(w_raw), CONSTRUCTOR,
    )


def m8_step(ctx) -> None:
    """5.2절 5단계. 4단계 w_raw에 M8을 적용해 positions/{t}.parquet을 쓴다."""
    cfg, _, risk = _load_params()
    t = ctx.date
    if pd.Timestamp(t) < _positions_from(cfg):
        ctx.log.info("5단계 B: %s는 positions 기록 전이라 쓰지 않는다 (6.8절)", t)
        return
    if "w_raw" not in ctx.memory:
        raise RuntimeError("4단계 w_raw가 없다")
    inputs = ctx.memory.get(MEMORY_KEY)
    if inputs is None or inputs.date != t:
        if not ctx.uses_stub:
            raise RuntimeError("4단계 입력이 없는데 스텁 실행도 아니다")
        from src.pipeline.run_weekly import empty_frame  # D의 스텁 형식 (9.3절 열, 0행)

        ctx.write_table("positions", empty_frame("positions"), date=t)
        ctx.log.warning("5단계 B: 앞 단계가 스텁이라 0행 positions를 스텁으로 쓴다")
        return
    positions = risk_layer_positions(inputs, ctx.memory["w_raw"], risk)
    path = ctx.write_table("positions", positions, date=t)
    cash = float(positions.loc[positions["ticker"] == CASH, "target_weight"].iloc[0])
    ctx.log.info(
        "5단계 B: %s %d종목, 현금 %.4f, E_t %.4f, regime_mc %d",
        path, len(positions) - 1, cash, positions["exposure"].iloc[0], positions["regime_mc"].iloc[0],
    )


def _replayed_holdings(ctx, cfg: Mapping[str, Any], dates: list[str], t: str) -> set[str]:
    read_recorded = _resolve(RECORDED_POSITIONS_ENTRY, "D 지난 forward positions")
    if read_recorded is None:
        raise RuntimeError("지난 forward positions를 읽는 D 함수가 연결되지 않았다 (weekly.RECORDED_POSITIONS_ENTRY)")
    recorded = read_recorded(t)
    missing = [d for d in dates if d not in recorded]
    if missing:
        ctx.log.warning(
            "4단계 B: forward 기록이 없는 신호일 %s는 리밸런싱 없이 직전 보유를 유지한 것으로 재생한다", ", ".join(missing)
        )
    if not recorded:
        return set()  # 기록이 하나도 없으면 아직 현금 100%다 (엔진을 만들 필요가 없다)
    maps = _call_m9(ctx, M9_ENGINE_ENTRY, "체결 지도", dates[0], t)
    cooldowns, events = (None, None) if maps is None else maps
    inputs = engine_inputs(*(ctx.read_table(name) for name in ENGINE_INPUTS))
    engine = build_engine(inputs, cfg, cooldowns=cooldowns, event_liquidations=events)
    return replay_holdings(engine, recorded, dates, t)


def _load_params() -> tuple[dict[str, Any], BufferRuleParams, RiskLayerParams]:
    cfg = load_prereg()
    return cfg, BufferRuleParams.from_prereg(cfg), RiskLayerParams.from_prereg(cfg)


def _positions_from(cfg: Mapping[str, Any]) -> pd.Timestamp:
    return pd.Timestamp(cfg["splits"]["forward_test"]["positions_from"])


def _stub_inputs(ctx, names: Iterable[str]) -> list[str]:
    return [name for name in names if name in ctx.stub_tables]


def _mark_stub(ctx, label: str) -> None:
    """다른 역할의 입력 없이 계산한 실행. 출력이 _stub/로 가고 7단계가 forward 커밋을 하지 않는다 (6.12절)."""
    if label not in ctx.stubbed:
        ctx.stubbed.append(label)
    ctx.log.warning("%s: 연결된 함수가 없어 이 실행을 스텁 실행으로 표시한다 (forward 커밋 안 함)", label)


def _call_m9(ctx, entry: Any, what: str, *args: str) -> Any:
    """M9 함수를 부른다. 연결 전, 함수를 찾지 못함(PR #34 미병합), 경보 파일 없음(FileNotFoundError)이면
    M9 없이 계산하고(_without_m9) None을 돌려준다. 다른 오류는 M9가 틀린 값을 냈을 수 있어 그대로 낸다."""
    try:
        fn = _resolve(entry, f"PM M9 {what}")
        if fn is not None:
            return fn(*args)
        reason = "함수가 연결되지 않았다"
    except EntryNotFound as e:
        reason = str(e)
    except FileNotFoundError as e:
        reason = f"경보 파일이 없다: {e}"
    _without_m9(ctx, what, reason)
    return None


def _without_m9(ctx, what: str, reason: str) -> None:
    """M9 함수(쿨다운 또는 체결 지도) 없이 계산한 실행. 기본은 기록하고 deviation으로 남긴다 (PM 11/9 결정)."""
    if not RECORD_WITHOUT_M9:
        _mark_stub(ctx, f"4(M9 {what} 없음)")
        return
    note = f"{ctx.date}: M9 {what} 없이 계산했다 ({reason}). PM 11/9 결정에 따라 deviation log에 남긴다"
    notes = ctx.memory.setdefault(DEVIATIONS_KEY, [])
    if note not in notes:
        notes.append(note)
    ctx.log.warning("4단계 B: %s", note)


class EntryNotFound(RuntimeError):
    """*_ENTRY에 이름이 있는데 함수를 찾지 못했다 (담당 PR 미병합 등)."""


def _resolve(entry: Any, what: str) -> Any:
    if entry is None or callable(entry):
        return entry
    module, _, name = str(entry).partition(":")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        raise EntryNotFound(f"{what} 함수 {entry}를 찾지 못했다 (담당 PR이 병합됐는지 확인한다)") from e
