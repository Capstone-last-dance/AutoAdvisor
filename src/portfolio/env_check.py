"""작업 B-05 완료 기준 확인 (6.5절, B 소유).

(1) run_full_episode: RL 환경이 M5 엔진 위에서 첫 신호일부터 마지막 신호일까지 끝까지 돈다.
(2) compare_buffer_and_env: 같은 비중이면 버퍼 규칙 실행과 RL 환경 실행의 체결·비용 결과가 같다.
    RL 환경의 행동 해석 자리에 버퍼 규칙 단계 1~4의 w_raw를 넣고, M8·엔진 호출·보상 계산은 그대로 쓴다.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.buffer_rule import buffer_rule_weights, run_buffer_rule
from src.portfolio.params import BufferRuleParams, RiskLayerParams, RLParams
from src.portfolio.risk_layer import CASH
from src.portfolio.rl_data import SignalSnapshot
from src.portfolio.rl_env import PortfolioEnv

Policy = Callable[[np.ndarray], np.ndarray]


class BufferWeightsEnv(PortfolioEnv):
    """행동 해석 (6.5절 (1)~(3)) 대신 버퍼 규칙 단계 1~4의 w_raw를 슬롯에 넣는 RL 환경 (episode="full", λ = 0).

    단계 1의 H는 환경이 엔진에서 받은 보유 비중(weights_obs)이다. 버퍼 규칙 실행과 같은 출처다.
    """

    def __init__(
        self,
        snapshots: Sequence[SignalSnapshot],
        engine,
        rl: RLParams,
        risk: RiskLayerParams,
        buffer: BufferRuleParams,
    ) -> None:
        super().__init__(snapshots, engine, rl, risk, lam=0.0, episode="full")
        self.buffer = buffer

    def _interpret(self, action: np.ndarray) -> np.ndarray:
        snap = self.snapshots[self._i]
        signals = pd.DataFrame(
            {"ticker": snap.tickers, "rank": snap.rank, "crash_flag": snap.crash_flag.astype(np.int8)}
        )
        held = [t for t, w in self._weights.items() if w > 0.0]
        w_raw = buffer_rule_weights(held, signals, snap.tickers, snap.tickers[snap.cooldown], self.buffer)
        slot = {t: j for j, t in enumerate(self._slot_tickers)}
        missing = sorted(set(w_raw.index) - set(slot))
        if missing:
            raise RuntimeError(f"{snap.date}: 버퍼 규칙 종목 {missing}이 RL 후보 슬롯에 없다")
        w = np.zeros(self.rl.slots)
        w[[slot[t] for t in w_raw.index]] = w_raw.to_numpy()
        return w


def _step_all(env: PortfolioEnv, policy: Policy | None) -> list[dict[str, Any]]:
    obs, _ = env.reset(seed=0)
    infos = []
    while True:
        action = np.zeros(env.action_space.shape, dtype=np.float32) if policy is None else policy(obs)
        obs, _, _, truncated, info = env.step(action)
        infos.append(info)
        if truncated:
            return infos


def ledger_counts(engine) -> dict[str, int]:
    """엔진 원장에서 4.3절 규칙이 실제로 걸린 횟수 (체결 거부 이유별, 보유 중 폐지 청산)."""
    ledger = engine.ledger
    out = {"days": len(ledger), "fills": sum(len(row["fills"]) for row in ledger)}
    reasons = Counter(reason for row in ledger for reason in row["rejected"].values())
    out.update({"rejected_" + reason: int(n) for reason, n in sorted(reasons.items())})
    # 폐지일에는 그 종목의 보통 주문을 내지 않으므로, 그날 그 종목의 체결은 정리매매 청산이다
    last_day = {ticker: day for ticker, (day, _) in engine.delistings.items()}
    out["delisting_fills"] = sum(1 for row in ledger for f in row["fills"] if last_day.get(f.ticker) == row["date"])
    return out


def run_full_episode(
    snapshots: Sequence[SignalSnapshot],
    engine,
    rl: RLParams,
    risk: RiskLayerParams,
    *,
    lam: float = 0.0,
    policy: Policy | None = None,
) -> dict[str, Any]:
    """완료 기준 (1). policy가 없으면 행동 0(모든 로짓이 같음)으로 돈다. 엔진은 새로 만든 것을 넘긴다."""
    env = PortfolioEnv(snapshots, engine, rl, risk, lam, episode="full")
    start = perf_counter()
    infos = _step_all(env, policy)
    seconds = perf_counter() - start
    executed = [i for i in infos if i["executed"]]
    return {
        "n_steps": len(infos),
        "n_executed": len(executed),
        "first_date": infos[0]["date"],
        "last_date": infos[-1]["date"],
        "nav_end": executed[-1]["nav"] if executed else 1.0,
        "mean_excess_return": float(np.mean([i["excess_return"] for i in executed])) if executed else 0.0,
        "mean_turnover": float(np.mean([i["turnover"] for i in executed])) if executed else 0.0,
        "ms_per_step": 1000.0 * seconds / len(infos),
        "rebalance_count": int(engine.rebalance_count),
        "scaled_rebalance_count": int(engine.scaled_rebalance_count),
    }


def _fill_map(row: dict) -> dict[str, tuple[float, float, float]]:
    return {f.ticker: (f.signed_quantity, f.notional, f.cost) for f in row["fills"]}


def compare_buffer_and_env(
    snapshots: Sequence[SignalSnapshot],
    make_engine: Callable[[], Any],
    rl: RLParams,
    risk: RiskLayerParams,
    buffer: BufferRuleParams,
) -> dict[str, Any]:
    """완료 기준 (2). make_engine()은 record_ledger=True인 새 엔진을 돌려줘야 한다.

    반환값의 max_* 는 두 실행의 차이 최댓값이다. 목표 비중, 일별 체결 수량·금액·비용, 일별 NAV·현금,
    구간 초과수익·회전율을 비교한다. 체결 거부 이유나 체결 종목이 다르면 mismatches에 날짜를 남긴다.
    """
    buffer_engine, env_engine = make_engine(), make_engine()
    buf = run_buffer_rule(snapshots, buffer_engine, buffer, risk)
    infos = _step_all(BufferWeightsEnv(snapshots, env_engine, rl, risk, buffer), None)

    weight_diff = 0.0
    pos = buf.positions[buf.positions["ticker"] != CASH]
    for info in infos:
        b = pos[pos["date"] == pd.Timestamp(info["date"])].set_index("ticker")["target_weight"].to_dict()
        if set(b) != set(info["targets"]):
            raise AssertionError(f"{info['date']}: 목표 종목이 다르다")
        weight_diff = max([weight_diff] + [abs(b[t] - info["targets"][t]) for t in b])

    mismatches, fill_diff, nav_diff = [], 0.0, 0.0
    if len(buffer_engine.ledger) != len(env_engine.ledger):
        raise AssertionError("두 실행의 원장 길이가 다르다")
    for a, e in zip(buffer_engine.ledger, env_engine.ledger):
        fa, fe = _fill_map(a), _fill_map(e)
        if a["date"] != e["date"] or set(fa) != set(fe) or a["rejected"] != e["rejected"]:
            mismatches.append(a["date"])
            continue
        fill_diff = max([fill_diff] + [abs(x - y) for t in fa for x, y in zip(fa[t], fe[t])] + [abs(a["cost"] - e["cost"])])
        nav_diff = max(nav_diff, abs(a["nav_after"] - e["nav_after"]), abs(a["cash"] - e["cash"]))

    executed = [i for i in infos if i["executed"]]
    env_excess = np.array([i["excess_return"] for i in executed])
    env_turnover = np.array([i["turnover"] for i in executed])
    return {
        "n_signal_dates": len(snapshots),
        "n_periods": len(executed),
        "ledger_days": len(buffer_engine.ledger),
        "mismatched_days": mismatches,
        "max_weight_diff": weight_diff,
        "max_fill_diff": fill_diff,
        "max_nav_diff": nav_diff,
        "max_excess_return_diff": float(np.max(np.abs(env_excess - buf.excess_returns), initial=0.0)),
        "max_turnover_diff": float(np.max(np.abs(env_turnover - buf.turnover), initial=0.0)),
        "ledger_counts": ledger_counts(buffer_engine),
    }
