"""M8 리스크 레이어 (6.9절, B 소유).

같은 입력이면 항상 같은 출력을 내는 순수 함수다. 버퍼 규칙과 RL은 모두 이 모듈을 거친 비중으로만
체결한다 (R-06). 신호일 t의 계산에는 t일 종가까지의 코스피 200만 쓴다 (R-03).

- apply_risk_layer: 원시 목표 비중 w_raw → positions 행 (9.3절). 버퍼 규칙과 파이프라인이 호출한다
- risk_layer_weights, check_invariants: 같은 계산의 배열 경로. RL 환경이 스텝마다 호출한다
- market_state, market_state_table, sigma_series: Step 4~5 (E_t, regime_mc)
- compute_regime_mc: 공개 함수. A가 피처 regime_mc를 만들 때 호출한다
- compute_vol_target_annual: risk_layer.vol_target_annual(Q-12) 계산
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.portfolio.params import RiskLayerParams, load_prereg

CASH = "CASH"
CONSTRUCTORS = ("buffer", "rl")

# full: Step 1~6
# no_exposure: G4 '노출 조절 제외' 진단. Step 4~5를 끄고 E_t = 1로 둔다 (판정에 쓰지 않는다)
# off: G7 'M8·M9 미적용' 실행. Step 2~6과 쿨다운을 끄고 Step 1의 crash_flag·U_t 필터만 남긴다
Mode = Literal["full", "no_exposure", "off"]
MODES = ("full", "no_exposure", "off")

DateLike = str | dt.date | pd.Timestamp


class RiskLayerViolation(RuntimeError):
    """6.9절 불변식 위반. 실험을 중단한다 (G0)."""


class InsufficientHistory(ValueError):
    """Step 4~5 창을 채울 코스피 200 이력이 부족하다 (design_B.md [결정 필요] D4)."""


@dataclass(frozen=True)
class MarketState:
    """신호일 t의 Step 4~5 결과."""

    sigma: float
    regime_mc: int
    exposure: float  # E_t. mode가 full이 아니면 1.0


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode = {mode!r}: {MODES} 중 하나여야 한다")


def _as_bm(bm_close: pd.Series) -> pd.Series:
    bm = bm_close.astype("float64")
    bm.index = pd.DatetimeIndex(bm.index)
    if bm.index.has_duplicates or not bm.index.is_monotonic_increasing:
        raise ValueError("코스피 200 종가의 날짜가 유일한 오름차순이 아니다")
    if bm.isna().any() or (bm <= 0.0).any():
        raise ValueError("코스피 200 종가에 결측이나 0 이하 값이 있다")
    return bm


def sigma_series(bm_close: pd.Series, params: RiskLayerParams) -> pd.Series:
    """Step 4의 sigma_t = std(r_{t−19}, …, r_t; ddof=1) × √252, r_s = ln(BM(s) / BM(s−1)).

    창을 채울 이력이 없는 날은 NaN이다.
    """
    bm = _as_bm(bm_close)
    n_vol = params.vol_lookback_days
    r = np.log(bm / bm.shift(1))
    return r.rolling(n_vol, min_periods=n_vol).std(ddof=1) * np.sqrt(params.annualization_daily)


def market_state_table(bm_close: pd.Series, params: RiskLayerParams, mode: Mode = "full") -> pd.DataFrame:
    """거래일마다 Step 4~5를 계산한다. 모든 창이 과거 방향이라 각 행은 그날 종가까지만 쓴다.

    열: sigma, bear, rebound, regime_mc, exposure. 창을 채울 이력이 없는 날은 NaN이다.
    다만 rebound_t = 0이면 bear_t의 창이 차지 않아도 regime_mc_t = bear_t × 0 = 0으로 정해진다 (D4).
    """
    _check_mode(mode)
    bm = _as_bm(bm_close)
    rp = params.momentum_crash_regime

    sigma = sigma_series(bm, params)  # Step 4

    # Step 5: DD_s = 1 − BM(s) / max(BM(s−251), …, BM(s)), bear_t = max(DD_{t−59}, …, DD_t) ≥ 0.20
    n_high = rp.drawdown_high_lookback_days
    dd = 1.0 - bm / bm.rolling(n_high, min_periods=n_high).max()
    n_recent = rp.drawdown_recent_window_days
    dd_recent = dd.rolling(n_recent, min_periods=n_recent).max()
    bear = (dd_recent >= rp.drawdown_threshold).astype("float64").where(dd_recent.notna())
    # rebound_t = BM(t) / BM(t−20) − 1 ≥ 0.10
    ret = bm / bm.shift(rp.rebound_lookback_days) - 1.0
    rebound = (ret >= rp.rebound_threshold).astype("float64").where(ret.notna())
    regime = (bear * rebound).where(rebound != 0.0, 0.0)  # bear가 NaN이어도 rebound = 0이면 0 (D4)

    if mode == "full":
        # E_t = min(1.00, max(0.30, vol_target_annual / sigma_t)). sigma_t = 0이면 exposure_max
        e = (params.require_vol_target() / sigma).clip(params.exposure_min, params.exposure_max)
        e = e.where(regime != 1.0, np.minimum(e, rp.exposure_cap))
        e = e.where(sigma.notna() & regime.notna())
    else:
        e = pd.Series(1.0, index=bm.index)
    return pd.DataFrame({"sigma": sigma, "bear": bear, "rebound": rebound, "regime_mc": regime, "exposure": e})


def _window_days(params: RiskLayerParams) -> int:
    """신호일 하나의 Step 4~5에 필요한 코스피 200 종가 개수."""
    rp = params.momentum_crash_regime
    return max(
        params.vol_lookback_days + 1,
        rp.drawdown_high_lookback_days + rp.drawdown_recent_window_days - 1,
        rp.rebound_lookback_days + 1,
    )


def state_from_row(row: pd.Series, date: DateLike) -> MarketState:
    """market_state_table의 한 행을 MarketState로 바꾼다. 결측이면 InsufficientHistory."""
    if row[["sigma", "regime_mc", "exposure"]].isna().any():
        raise InsufficientHistory(f"{pd.Timestamp(date).date()}: Step 4~5 창을 채울 코스피 200 이력이 부족하다")
    return MarketState(sigma=float(row["sigma"]), regime_mc=int(row["regime_mc"]), exposure=float(row["exposure"]))


def market_state(bm_close: pd.Series, date: DateLike, params: RiskLayerParams, mode: Mode = "full") -> MarketState:
    """신호일 t의 Step 4~5. t일보다 늦은 종가는 잘라내고 계산한다 (R-03)."""
    t = pd.Timestamp(date)
    bm = _as_bm(bm_close).loc[:t]
    if bm.empty or bm.index[-1] != t:
        raise ValueError(f"{t.date()} 코스피 200 종가가 없다")
    table = market_state_table(bm.iloc[-_window_days(params):], params, mode)
    return state_from_row(table.iloc[-1], t)


def risk_layer_weights(
    w_raw: np.ndarray,
    blocked: np.ndarray,
    sector_codes: np.ndarray,
    exposure: float,
    params: RiskLayerParams,
    mode: Mode = "full",
) -> np.ndarray:
    """Step 1~3과 Step 6을 종목 배열에 적용한다. 반환값은 종목 비중이고, 현금은 1 − 합이다.

    blocked: Step 1 대상 (crash_flag = 1, M9 쿨다운, U_t 밖)
    sector_codes: 0 이상의 업종 번호. 업종 정보가 없는 종목은 '미분류' 번호 하나로 묶는다
    exposure: Step 4~5의 E_t
    잘린 비중은 다른 종목에 재배분하지 않고 현금으로 둔다.
    """
    w = np.where(blocked, 0.0, np.asarray(w_raw, dtype=np.float64))  # Step 1
    if mode == "off":
        return w
    w = np.minimum(w, params.stock_cap)  # Step 2
    sector_sums = np.bincount(sector_codes, weights=w, minlength=1)  # Step 3
    over = sector_sums > params.sector_cap
    if over.any():
        scale = np.ones_like(sector_sums)
        scale[over] = params.sector_cap / sector_sums[over]
        w = w * scale[sector_codes]
    return exposure * w  # Step 6


def check_invariants(w: np.ndarray, sector_codes: np.ndarray, params: RiskLayerParams, mode: Mode = "full") -> float:
    """6.9절 불변식을 확인하고 현금 비중을 돌려준다. 하나라도 어기면 RiskLayerViolation (G0).

    mode가 off이면 한도를 끈 실행(R-06 예외)이므로 비음수, 현금 ≥ 0, 합계만 확인한다.
    """
    tol = params.invariant_tolerance
    w = np.asarray(w, dtype=np.float64)
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise RiskLayerViolation("종목 비중이 음수이거나 유한하지 않다")
    if mode != "off":
        if np.any(w > params.stock_cap + tol):
            raise RiskLayerViolation(f"종목 비중 {w.max():.12f} > stock_cap {params.stock_cap}")
        sector_sums = np.bincount(sector_codes, weights=w, minlength=1)
        if np.any(sector_sums > params.sector_cap + tol):
            raise RiskLayerViolation(f"업종 비중 합 {sector_sums.max():.12f} > sector_cap {params.sector_cap}")
    total = float(w.sum())
    if total > 1.0 + tol:
        raise RiskLayerViolation(f"종목 비중 합 {total:.12f} > 1 (현금 < 0)")
    cash = max(0.0, 1.0 - total)
    if abs(total + cash - 1.0) > tol:
        raise RiskLayerViolation("비중 합계가 1이 아니다")
    return cash


def positions_frame(
    date: DateLike,
    targets: Mapping[str, float],
    cash: float,
    constructor: str,
    exposure: float,
    regime_mc: int,
) -> pd.DataFrame:
    """positions 행 (9.3절). 비중이 0보다 큰 종목과 "CASH" 행만 쓴다.

    date 열은 io.py(D-02) 규칙대로 시각 없는 datetime64[ns]다.
    """
    if constructor not in CONSTRUCTORS:
        raise ValueError(f"constructor = {constructor!r}: {CONSTRUCTORS} 중 하나여야 한다")
    tickers = sorted(t for t, w in targets.items() if w > 0.0)
    n = len(tickers) + 1
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex([pd.Timestamp(date).normalize()] * n).as_unit("ns"),
            "ticker": tickers + [CASH],
            "target_weight": np.array([targets[t] for t in tickers] + [cash], dtype=np.float64),
            "constructor": [constructor] * n,
            "exposure": np.full(n, exposure, dtype=np.float64),
            "regime_mc": np.full(n, regime_mc, dtype=np.int8),
        }
    )


def sectors_at(sector: pd.DataFrame, date: DateLike) -> dict[str, str]:
    """sector 표(9.3절)에서 t일에 유효한 업종: valid_from ≤ t이고, valid_to가 null이거나 t ≤ valid_to."""
    t = pd.Timestamp(date)
    valid_from = pd.to_datetime(sector["valid_from"])
    valid_to = pd.to_datetime(sector["valid_to"])
    current = sector.loc[(valid_from <= t) & (valid_to.isna() | (valid_to >= t)) & sector["sector"].notna()]
    if current["ticker"].duplicated().any():
        raise ValueError(f"{t.date()}: 한 종목에 유효한 업종 행이 둘 이상이다")
    return dict(zip(current["ticker"], current["sector"]))


def apply_risk_layer(
    w_raw: Mapping[str, float] | pd.Series,
    signals: pd.DataFrame,
    sectors: Mapping[str, str],
    bm_close: pd.Series,
    cooldown: Iterable[str],
    universe: Iterable[str],
    date: DateLike,
    params: RiskLayerParams,
    constructor: str,
    mode: Mode = "full",
) -> pd.DataFrame:
    """6.9절 M8. 원시 목표 비중 w_raw(종목코드 → 비중, 합 ≤ 1)를 positions 행(9.3절)으로 바꾼다.

    signals: 신호일 t의 signals 행 (ticker, crash_flag). date 열이 있으면 모두 t여야 한다 (R-03)
    sectors: t일에 유효한 업종표 (sectors_at). 없는 종목은 unknown_sector_label로 묶는다
    bm_close: 코스피 200 종가 (날짜 → 종가). t일보다 늦은 값은 쓰지 않는다
    cooldown: 체결일에 신규 진입이 금지된 종목 (M9). mode가 off이면 쓰지 않는다
    universe: U_t 종목코드
    """
    _check_mode(mode)
    t = pd.Timestamp(date)
    w = pd.Series(w_raw, dtype="float64")
    if w.index.has_duplicates:
        raise ValueError("w_raw의 종목코드가 중복된다")
    if not np.all(np.isfinite(w.to_numpy())) or (w < 0.0).any():
        raise ValueError("w_raw에 음수나 유한하지 않은 값이 있다")
    if w.sum() > 1.0 + params.invariant_tolerance:
        raise ValueError(f"w_raw 합 {w.sum():.12f}가 1을 넘는다")
    w = w[w > 0.0]
    tickers = [str(x) for x in w.index]

    if "date" in signals.columns and (pd.to_datetime(signals["date"]) != t).any():
        raise ValueError("signals에 신호일 t가 아닌 행이 있다 (R-03)")
    crash = dict(zip(signals["ticker"], signals["crash_flag"]))
    in_universe = set(universe)
    missing = [tk for tk in tickers if tk in in_universe and tk not in crash]
    if missing:
        raise ValueError(f"U_t 종목인데 signals 행이 없다: {missing}")
    cooldown_set = set() if mode == "off" else set(cooldown)
    blocked = np.array(
        [(tk not in in_universe) or crash.get(tk, 0) == 1 or (tk in cooldown_set) for tk in tickers], dtype=bool
    )
    labels = np.array([sectors.get(tk, params.unknown_sector_label) for tk in tickers], dtype=object)
    codes = pd.factorize(labels)[0].astype(np.int64)

    state = market_state(bm_close, t, params, mode)
    w_final = risk_layer_weights(w.to_numpy(), blocked, codes, state.exposure, params, mode)
    cash = check_invariants(w_final, codes, params, mode)
    targets = {tk: float(x) for tk, x in zip(tickers, w_final) if x > 0.0}
    return positions_frame(t, targets, cash, constructor, state.exposure, state.regime_mc)


def load_kospi200() -> pd.Series:
    """index 표(9.3절)의 KOSPI200 종가 (날짜 → 종가). io.py(D-02)로만 읽는다 (R-18)."""
    try:
        from src.common.io import read_table
    except ModuleNotFoundError as e:
        if e.name not in ("src.common.io", "src.common"):
            raise
        raise RuntimeError("src/common/io.py(작업 D-02)가 아직 없다. bm_close를 직접 넘겨야 한다") from e
    index = read_table("index")
    kospi = index.loc[index["index_code"] == "KOSPI200"]
    return pd.Series(kospi["close"].to_numpy(dtype="float64"), index=pd.to_datetime(kospi["date"])).sort_index()


def compute_regime_mc(
    date: DateLike, *, bm_close: pd.Series | None = None, params: RiskLayerParams | None = None
) -> int:
    """신호일 t의 모멘텀 붕괴 국면 플래그 (6.9절 Step 5). A가 피처 regime_mc를 만들 때 호출한다.

    bm_close를 넘기지 않으면 load_kospi200()으로 읽고, params를 넘기지 않으면 configs/prereg.yaml에서
    읽는다. Step 5에는 vol_target_annual이 필요 없다. t일보다 늦은 종가는 쓰지 않는다 (R-03).
    2011년 초처럼 bear_t의 창(종가 311개)이 차지 않아도 rebound_t = 0이면 0을 돌려준다. rebound_t = 1인데
    bear_t를 계산할 수 없거나, rebound_t의 창(종가 21개)도 차지 않으면 InsufficientHistory를 낸다 (D4).
    """
    if params is None:
        params = RiskLayerParams.from_prereg(load_prereg())
    if bm_close is None:
        bm_close = load_kospi200()
    return market_state(bm_close, date, params, mode="no_exposure").regime_mc


def compute_vol_target_annual(bm_close: pd.Series, params: RiskLayerParams, years: tuple[int, int]) -> float:
    """risk_layer.vol_target_annual (Q-12): years[0]년 첫 거래일부터 years[1]년 마지막 거래일까지 매 거래일
    계산한 sigma_t(Step 4와 같은 식)의 산술평균이다. years는 params.vol_target_years(cfg) = (2011, 2015).

    years[1]년보다 늦은 종가는 계산 전에 잘라낸다 (6.9절: 2016년 이후 데이터 금지). 다만 그 해가 끝까지
    들어왔는지 확인하려고 다음 거래일이 있는지는 본다. 첫해 첫 거래일의 sigma_t에는 직전 20개 수익률이
    필요하므로 그 앞 종가(2010년 12월분)는 쓴다.
    """
    first, last = years
    bm = _as_bm(bm_close)
    end = pd.Timestamp(f"{last}-12-31")
    if not (bm.index > end).any():
        raise ValueError(f"{last}년 다음 거래일 종가가 없어 {last}년이 끝까지 들어왔는지 확인할 수 없다")
    sigma = sigma_series(bm.loc[:end], params)
    window = sigma.loc[pd.Timestamp(f"{first}-01-01") : end]
    if window.empty:
        raise ValueError(f"{first}~{last}년 코스피 200 종가가 없다")
    if window.isna().any():
        raise InsufficientHistory(
            f"{window.index[0].date()}의 sigma_t를 계산할 종가 {params.vol_lookback_days + 1}개가 없다"
        )
    return float(window.mean())
