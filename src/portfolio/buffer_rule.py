"""M4 버퍼 규칙 (6.5절, B 소유).

- buffer_rule_weights: 단계 1~4. 보유 집합 H와 신호일 t의 signals로 원시 목표 비중 w_raw를 만든다
- buffer_targets: 신호일 스냅샷에서 단계 1~5. RL 환경과 같은 M8 배열 경로를 쓴다 (R-06)
- run_buffer_rule: 모든 신호일에서 단계 1~5를 거쳐 M5 엔진으로 체결한다

RL 정책과 같은 스냅샷(rl_data), 같은 M8 함수, 같은 엔진 호출을 써서 G5 비교의 입력이 같게 한다.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.portfolio.interfaces import ExecutionEngine
from src.portfolio.params import BufferRuleParams, RiskLayerParams
from src.portfolio.risk_layer import Mode, check_invariants, positions_frame, risk_layer_weights
from src.portfolio.rl_data import SignalSnapshot


def buffer_rule_weights(
    holdings: Iterable[str],
    signals: pd.DataFrame,
    universe: Iterable[str],
    cooldown: Iterable[str],
    params: BufferRuleParams,
) -> pd.Series:
    """6.5절 버퍼 규칙 단계 1~4의 원시 목표 비중 w_raw (종목코드 → 비중, 나머지는 현금).

    단계 1: holdings는 직전 체결 후 보유 종목 집합 H다 (M5 상태에서 읽는다).
    단계 2: H 중 rank ≤ exit_rank_greater_than, crash_flag = 0, U_t 소속, 쿨다운 아님을 모두 만족하면 유지한다.
    단계 3: 유지 종목이 max_holdings보다 적으면 보유하지 않은 종목 중 rank ≤ enter_rank_max,
            crash_flag = 0, 쿨다운 아님을 만족하는 종목을 rank 오름차순으로 추가한다.
    단계 4: 보유 예정 종목에 1 ÷ 종목 수를 준다. 종목이 없으면 빈 Series(현금 100%)다.
    단계 5는 호출자가 결과를 M8에 넘기는 것이다 (R-06).

    signals: 신호일 t의 signals 행 (ticker, rank, crash_flag). rank는 U_t 안의 순위다 (6.4절).
    signals에 없는 보유 종목은 rank가 없으므로 유지하지 않는다. 같은 rank는 종목코드 순으로 정한다.
    """
    if signals["ticker"].duplicated().any():
        raise ValueError("signals에 같은 종목이 두 번 있다")
    s = signals.sort_values(["rank", "ticker"], kind="stable")
    rank = dict(zip(s["ticker"], s["rank"].astype(int)))
    allowed = {t for t, flag in zip(s["ticker"], s["crash_flag"]) if flag == 0} - set(cooldown)
    held, members = set(holdings), set(universe)

    keep = [t for t in rank if t in held and t in members and t in allowed and rank[t] <= params.exit_rank_greater_than]
    room = max(params.max_holdings - len(keep), 0)
    add = [t for t in rank if t not in held and t in allowed and rank[t] <= params.enter_rank_max][:room]
    chosen = sorted(keep + add, key=rank.__getitem__)
    return pd.Series(1.0 / max(len(chosen), 1), index=pd.Index(chosen, dtype=object), dtype="float64")


def buffer_targets(
    snap: SignalSnapshot,
    holdings: Iterable[str],
    buffer: BufferRuleParams,
    risk: RiskLayerParams,
    mode: Mode = "full",
) -> tuple[dict[str, float], float, float]:
    """신호일 스냅샷 t에서 단계 1~5. 반환값은 (M8을 거친 종목 비중, 현금 비중, E_t)다.

    M8은 RL 환경과 같은 배열 경로(risk_layer_weights, check_invariants)로 적용한다.
    mode="off"(G7 'M8·M9 미적용')이면 M9 쿨다운도 끄고, mode가 full이 아니면 E_t = 1이다 (6.9절).
    """
    cooldown = snap.cooldown if mode != "off" else np.zeros_like(snap.cooldown)
    signals = pd.DataFrame({"ticker": snap.tickers, "rank": snap.rank, "crash_flag": snap.crash_flag.astype(np.int8)})
    w_raw = buffer_rule_weights(holdings, signals, snap.tickers, snap.tickers[cooldown], buffer)

    w = np.zeros(len(snap.tickers))
    w[np.array([snap.index[t] for t in w_raw.index], dtype=np.int64)] = w_raw.to_numpy()
    exposure = snap.exposure if mode == "full" else 1.0
    w = risk_layer_weights(w, snap.crash_flag | cooldown, snap.sector_code, exposure, risk, mode)  # 단계 5: M8
    cash = check_invariants(w, snap.sector_code, risk, mode)  # G0
    return {str(snap.tickers[j]): float(w[j]) for j in np.flatnonzero(w > 0.0)}, cash, exposure


@dataclass(frozen=True)
class BufferRun:
    """run_buffer_rule의 결과."""

    positions: pd.DataFrame  # 9.3절 positions 행 (constructor = "buffer")
    excess_returns: np.ndarray  # 체결 구간마다 R_net − R_BM (6.6절 주간 초과수익 a_k)
    turnover: np.ndarray  # 체결 구간마다 체결일의 회전율 (M5)


def run_buffer_rule(
    snapshots: Sequence[SignalSnapshot],
    engine: ExecutionEngine,
    buffer: BufferRuleParams,
    risk: RiskLayerParams,
    mode: Mode = "full",
) -> BufferRun:
    """첫 신호일에 현금 100%로 시작해 모든 신호일에서 단계 1~5를 거쳐 M5로 체결한다.

    단계 1의 H는 직전 run_period가 돌려준 보유 종목(weights_obs)이다. 그래서 M9 청산이나 체결 실패로
    목표와 실제 보유가 다르면 실제 보유를 따른다. 마지막 신호일에는 다음 신호일이 없으므로 비중만 기록하고
    체결하지 않는다 (RL 환경의 episode="full"과 같다).
    """
    if not snapshots:
        raise ValueError("신호일 스냅샷이 없다")
    state, holdings = engine.initial_state(), set()
    frames, excess, turnover = [], [], []
    for i, snap in enumerate(snapshots):
        targets, cash, exposure = buffer_targets(snap, holdings, buffer, risk, mode)
        frames.append(positions_frame(snap.date, targets, cash, "buffer", exposure, snap.regime_mc))
        if i + 1 < len(snapshots):
            res = engine.run_period(state, targets, snap.date, snapshots[i + 1].date)
            state, holdings = res.state, {t for t, w in res.weights_obs.items() if w > 0.0}
            excess.append(res.r_net - res.r_bm)
            turnover.append(res.turnover)
    return BufferRun(
        pd.concat(frames, ignore_index=True),
        np.asarray(excess, dtype=np.float64),
        np.asarray(turnover, dtype=np.float64),
    )
