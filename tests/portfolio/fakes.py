"""B 테스트 전용 가짜 입력. C의 M5, A의 신호·피처, PM의 쿨다운을 대신한다 (R-17).

실제 실험, 보고서, 대시보드에 쓰지 않는다. 가격은 무작위 경로이고 거래일은 KRX 캘린더가 아니다.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.interfaces import PeriodResult
from src.portfolio.params import RiskLayerParams
from src.portfolio.risk_layer import market_state_table
from src.portfolio.rl_data import SignalSnapshot


def bm_crash_rebound_path(seed: int = 0) -> pd.Series:
    """상승 → 급락 → 반등이 들어 있는 코스피 200 대용 경로 (regime_mc가 0과 1을 모두 지나도록)."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate([np.full(350, 0.0015), np.full(100, -0.0045), np.full(50, 0.008), np.zeros(200)])
    days = pd.bdate_range("2010-01-04", periods=len(drift))
    return pd.Series(100.0 * np.exp(np.cumsum(drift + rng.normal(0.0, 0.01, len(drift)))), index=days)


def fake_ir(a: np.ndarray) -> float:
    """6.6절 IR 식. C의 M5 함수가 나오기 전까지 테스트에서만 쓴다."""
    a = np.asarray(a, dtype=np.float64)
    if len(a) < 2 or a.std(ddof=1) == 0.0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * np.sqrt(52))


class FakeEngine:
    """M5 대체. 신호일 다음 거래일 종가 체결, 4.8절 비용 식. 4.3절 상·하한가, 거래정지,
    상장폐지 규칙과 M9 이벤트 청산은 구현하지 않는다."""

    def __init__(self, closes: pd.DataFrame, bm_close: pd.Series, costs: Mapping[str, float]) -> None:
        self._pos = {d: i for i, d in enumerate(closes.index.strftime("%Y-%m-%d"))}
        self._col = {t: j for j, t in enumerate(closes.columns)}
        self._px = closes.to_numpy(dtype=np.float64)
        self._bm = bm_close.reindex(closes.index).to_numpy(dtype=np.float64)
        self._buy = costs["commission_per_side"] + costs["slippage_per_side"]
        self._sell = costs["sell_tax"] + costs["commission_per_side"] + costs["slippage_per_side"]

    def initial_state(self) -> dict[str, Any]:
        return {"qty": {}, "cash": 1.0}

    def run_period(self, state, targets, signal_date, next_signal_date) -> PeriodResult:
        i0, i1 = self._pos[signal_date] + 1, self._pos[next_signal_date] + 1  # t+1, t′+1
        px0 = self._px[i0]
        value = {t: q * px0[self._col[t]] for t, q in state["qty"].items()}
        nav_start = state["cash"] + sum(value.values())
        new_value = {t: w * nav_start for t, w in targets.items()}
        names = set(value) | set(new_value)
        diff = {t: new_value.get(t, 0.0) - value.get(t, 0.0) for t in names}
        cost = sum(max(d, 0.0) for d in diff.values()) * self._buy + sum(max(-d, 0.0) for d in diff.values()) * self._sell
        cash = nav_start - sum(new_value.values()) - cost
        qty = {t: v / px0[self._col[t]] for t, v in new_value.items() if v > 0.0}
        cols = [self._col[t] for t in qty]
        navs = cash + self._px[i0 : i1 + 1][:, cols] @ np.array(list(qty.values()), dtype=np.float64)
        nav_obs, px_obs = navs[-2], self._px[i1 - 1]  # t′일 종가
        return PeriodResult(
            nav_start=nav_start,
            nav_end=float(navs[-1]),
            nav_max=float(navs.max()),
            r_net=float(navs[-1] / nav_start - 1.0),
            r_bm=float(self._bm[i1] / self._bm[i0] - 1.0),
            turnover=0.5 * sum(abs(d) for d in diff.values()) / nav_start,
            weights_obs={t: q * px_obs[self._col[t]] / nav_obs for t, q in qty.items()},
            cash_obs=cash / nav_obs,
            state={"qty": qty, "cash": cash},
        )


@dataclass
class ScriptedEngine:
    """run_period 결과를 테스트가 정한다. 기본은 가격 변화와 비용이 없는 체결이다."""

    script: Callable[[int, Mapping[str, float]], PeriodResult] | None = None
    calls: list[tuple[dict[str, float], str, str]] = field(default_factory=list)

    def initial_state(self) -> None:
        return None

    def run_period(self, state, targets, signal_date, next_signal_date) -> PeriodResult:
        self.calls.append((dict(targets), signal_date, next_signal_date))
        if self.script is not None:
            return self.script(len(self.calls) - 1, targets)
        cash = 1.0 - sum(targets.values())
        return PeriodResult(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, dict(targets), cash, None)


def make_snapshot(
    date: str = "2020-01-03",
    n: int = 60,
    crash: Sequence[int] = (),
    cooldown: Sequence[int] = (),
    sectors: Sequence[str] | None = None,
    exposure: float = 1.0,
    regime_mc: int = 0,
) -> SignalSnapshot:
    """종목코드 000001~이 rank 1~n인 스냅샷. crash, cooldown은 rank 번호로 준다."""
    tickers = np.array([f"{i:06d}" for i in range(1, n + 1)], dtype=object)
    rank = np.arange(1, n + 1)
    labels = np.array(sectors if sectors is not None else [f"S{i % 10}" for i in range(n)], dtype=object)
    return SignalSnapshot(
        date=date,
        tickers=tickers,
        rank=rank,
        rank_score_pct=(n - rank + 1) / n,
        crash_prob=np.linspace(0.0, 0.5, n),
        crash_flag=np.isin(rank, crash),
        cooldown=np.isin(rank, cooldown),
        vol_20=np.linspace(1.0, 0.0, n),
        sector_code=pd.factorize(labels)[0].astype(np.int64),
        mkt_vol_20=0.15,
        mkt_ret_20=0.01,
        regime_mc=regime_mc,
        exposure=exposure,
        index={t: i for i, t in enumerate(tickers)},
    )


@dataclass
class SyntheticMarket:
    closes: pd.DataFrame  # 거래일 × 종목코드
    bm_close: pd.Series
    signal_dates: list[str]
    signals: pd.DataFrame  # 9.3절 signals 열
    features: pd.DataFrame  # 9.3절 features 열 중 B가 읽는 것
    universe: pd.DataFrame  # 9.3절 universe 열
    sector: pd.DataFrame  # 9.3절 sector 열
    cooldown: dict[str, frozenset[str]]


def make_market(
    risk: RiskLayerParams,
    seed: int = 0,
    n_tickers: int = 80,
    n_universe: int = 60,
    start: str = "2009-01-01",
    end: str = "2013-12-31",
    first_signal: str = "2011-01-01",
) -> SyntheticMarket:
    """9.3절 형식의 무작위 시장. 신호일은 주마다 마지막 영업일이다."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    tickers = [f"{i:06d}" for i in range(1, n_tickers + 1)]
    closes = pd.DataFrame(
        1000.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (len(days), n_tickers)), axis=0)),
        index=days,
        columns=tickers,
    )
    bm = pd.Series(250.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(days)))), index=days)

    d = days[days >= pd.Timestamp(first_signal)][:-1]  # 마지막 날은 체결일이 없다
    iso = d.isocalendar()
    week_last = pd.Series(d, index=d).groupby([iso["year"].to_numpy(), iso["week"].to_numpy()]).max()
    signal_dates = sorted(x.strftime("%Y-%m-%d") for x in week_last)

    states = market_state_table(bm, risk, mode="no_exposure")
    ret20 = bm / bm.shift(20) - 1.0
    sig_rows, feat_rows, uni_rows, cooldown = [], [], [], {}
    for date in signal_dates:
        t = pd.Timestamp(date)
        members = sorted(rng.choice(tickers, n_universe + 5, replace=False))
        uni, outside = members[:n_universe], members[n_universe:]
        score = rng.normal(size=n_universe)
        rank = np.argsort(np.argsort(-score)) + 1
        prob = rng.uniform(size=n_universe)
        flag = np.zeros(n_universe, dtype=np.int8)
        flag[np.argsort(-prob)[: math.ceil(0.10 * n_universe)]] = 1
        for i, tk in enumerate(uni):
            sig_rows.append((t.date(), tk, score[i], int(rank[i]), prob[i], int(flag[i]), "synthetic"))
            feat_rows.append(
                (t.date(), tk, rng.uniform(), states.at[t, "sigma"], ret20.at[t], int(states.at[t, "regime_mc"]))
            )
        uni_rows += [(t.date(), tk, i + 1, True) for i, tk in enumerate(uni)]
        uni_rows += [(t.date(), tk, n_universe + i + 1, False) for i, tk in enumerate(outside)]
        cooldown[date] = frozenset(rng.choice(uni, 2, replace=False))

    signals = pd.DataFrame(
        sig_rows, columns=["date", "ticker", "rank_score", "rank", "crash_prob", "crash_flag", "model_version"]
    )
    features = pd.DataFrame(
        feat_rows, columns=["date", "ticker", "vol_20", "mkt_vol_20", "mkt_ret_20", "regime_mc"]
    )
    universe = pd.DataFrame(uni_rows, columns=["date", "ticker", "mcap_rank", "in_universe"])
    covered = [tk for i, tk in enumerate(tickers) if i % 10 != 0]  # 10종목 중 1개는 업종 정보 없음
    sector = pd.DataFrame(
        {
            "ticker": covered,
            "sector": [f"S{int(tk) % 8}" for tk in covered],
            "valid_from": pd.Timestamp(start).date(),
            "valid_to": None,
        }
    )
    return SyntheticMarket(closes, bm, signal_dates, signals, features, universe, sector, cooldown)
