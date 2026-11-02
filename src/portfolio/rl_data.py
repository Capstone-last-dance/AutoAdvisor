"""RL 환경 입력: 신호일별 스냅샷 (6.5절 관측, B 소유).

9.3절의 signals, features, universe, sector 표와 코스피 200 종가, M9 쿨다운으로 만든다.
스냅샷 t의 값은 모두 신호일 t에 알 수 있는 정보다 (R-03).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.portfolio.interfaces import CooldownFunction
from src.portfolio.params import RiskLayerParams
from src.portfolio.risk_layer import market_state_table, sectors_at, state_from_row


@dataclass(frozen=True, eq=False)
class SignalSnapshot:
    """신호일 t 하나의 관측과 M8 입력. 종목 배열은 모두 rank 오름차순(1이 최고)이다."""

    date: str  # YYYY-MM-DD
    tickers: np.ndarray  # U_t 종목코드 (6자리 문자열)
    rank: np.ndarray  # signals.rank
    rank_score_pct: np.ndarray  # signals.rank_score의 U_t 내 백분위
    crash_prob: np.ndarray  # signals.crash_prob
    crash_flag: np.ndarray  # signals.crash_flag = 1 (bool)
    cooldown: np.ndarray  # 체결일에 M9 쿨다운 (bool)
    vol_20: np.ndarray  # features.vol_20 (이미 U_t 내 순위로 바꾼 값)
    sector_code: np.ndarray  # t일 유효 업종 번호 ('미분류' 포함)
    mkt_vol_20: float
    mkt_ret_20: float
    regime_mc: int
    exposure: float  # E_t (M8 Step 4~5)
    index: dict[str, int] = field(repr=False)  # 종목코드 → 배열 위치

    @property
    def n_sector_codes(self) -> int:
        return int(self.sector_code.max()) + 1 if len(self.sector_code) else 0


def _by_date(df: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    return dict(tuple(df.groupby(pd.to_datetime(df["date"]))))


def _market_common(f: pd.DataFrame, col: str, date: str) -> float:
    values = f[col].unique()
    if len(values) != 1:
        raise ValueError(f"{date}: 시장 공통 피처 {col}의 값이 종목마다 다르다")
    return float(values[0])


def build_snapshots(
    dates: Sequence[str],
    signals: pd.DataFrame,
    features: pd.DataFrame,
    universe: pd.DataFrame,
    sector: pd.DataFrame,
    bm_close: pd.Series,
    cooldown_at: CooldownFunction,
    params: RiskLayerParams,
) -> list[SignalSnapshot]:
    """dates(오름차순 신호일)의 스냅샷. 표의 열 이름과 형식은 9.3절을 따른다.

    features에 regime_mc 열이 있으면 M8의 계산과 같은지 확인한다 (A가 compute_regime_mc를 호출해 만든다).
    """
    states = market_state_table(bm_close, params)
    sig_g, feat_g, uni_g = _by_date(signals), _by_date(features), _by_date(universe)
    snapshots = []
    for date in dates:
        t = pd.Timestamp(date)
        if t not in sig_g or t not in feat_g or t not in uni_g:
            raise ValueError(f"{date}: signals, features, universe 중 행이 없는 표가 있다")
        u = uni_g[t]
        u_tickers = set(u.loc[u["in_universe"].astype(bool), "ticker"])
        s = sig_g[t].sort_values(["rank", "ticker"], kind="stable").reset_index(drop=True)
        if set(s["ticker"]) != u_tickers or s["ticker"].duplicated().any():
            raise ValueError(f"{date}: signals 종목과 U_t가 다르다")
        tickers = s["ticker"].to_numpy(dtype=object)
        f = feat_g[t].set_index("ticker").reindex(tickers)
        if f["vol_20"].isna().any():
            raise ValueError(f"{date}: U_t 종목의 features.vol_20이 없다")

        state = state_from_row(states.loc[t], t)
        if "regime_mc" in f.columns and int(_market_common(f, "regime_mc", date)) != state.regime_mc:
            raise ValueError(f"{date}: features.regime_mc가 M8의 계산과 다르다")
        sectors = sectors_at(sector, t)
        labels = np.array([sectors.get(tk, params.unknown_sector_label) for tk in tickers], dtype=object)
        cooldown = set(cooldown_at(date))

        snapshots.append(
            SignalSnapshot(
                date=t.strftime("%Y-%m-%d"),
                tickers=tickers,
                rank=s["rank"].to_numpy(dtype=np.int64),
                rank_score_pct=s["rank_score"].rank(pct=True, method="average").to_numpy(dtype=np.float64),
                crash_prob=s["crash_prob"].to_numpy(dtype=np.float64),
                crash_flag=s["crash_flag"].to_numpy() == 1,
                cooldown=np.array([tk in cooldown for tk in tickers], dtype=bool),
                vol_20=f["vol_20"].to_numpy(dtype=np.float64),
                sector_code=pd.factorize(labels)[0].astype(np.int64),
                mkt_vol_20=_market_common(f, "mkt_vol_20", date),
                mkt_ret_20=_market_common(f, "mkt_ret_20", date),
                regime_mc=state.regime_mc,
                exposure=state.exposure,
                index={tk: i for i, tk in enumerate(tickers)},
            )
        )
    return snapshots
