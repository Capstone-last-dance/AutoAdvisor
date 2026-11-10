"""M2 가격·수급·제도·시장 피처 (6.2절, 담당 A). 정의는 6.2절 표 그대로다 (R-14).

입력은 거래일 × 종목 pivot이다. 모든 창은 t 이전 거래일만 센다(t−k는 시장 거래일 기준 k번째 이전). 시세가 없는 날은 결측이다.
순위 변환·결측 처리는 `rank_and_fill`이 한다. 이 모듈은 값 계산만 한다.
"""
import numpy as np
import pandas as pd

ANN = np.sqrt(252)

RANK_FEATURES = ["mom_12_1", "mom_12_7", "rev_1m", "rev_1w", "high52_prox", "vol_20", "vol_60", "max_ret_20",
                 "turnover_spike", "log_mcap"]
RAW_FEATURES = ["penny_flag", "ss_allowed"]
MARKET_FEATURES = ["mkt_vol_20", "mkt_ret_20", "regime_mc", "ss_regime"]


def mom_12_1(close):
    return close.shift(21) / close.shift(252) - 1


def vol_60(close):
    logret = np.log(close / close.shift(1))
    return logret.rolling(60, min_periods=60).std(ddof=1) * ANN


def compute_stock_features(close, close_raw, value, mcap) -> dict:
    """각 인자: 인덱스 거래일, 열 ticker의 DataFrame. 반환: 피처 이름 -> 같은 모양의 DataFrame (순위 변환 전).

    투자자별 순매수 피처(retail_net_20, foreign_net_20, inst_net_20)는 DEC-22로 뺐다 — 원천(investor_flows)이
    없어서 6.2절 피처 표에서 제외가 확정됐다."""
    ret = close.pct_change(fill_method=None)
    logret = np.log(close / close.shift(1))
    turn = value / mcap.where(mcap > 0)
    f = {
        "mom_12_1": mom_12_1(close),
        "mom_12_7": close.shift(147) / close.shift(252) - 1,
        "rev_1m": close / close.shift(21) - 1,
        "rev_1w": close / close.shift(5) - 1,
        "high52_prox": close / close.rolling(252, min_periods=252).max(),
        "vol_20": logret.rolling(20, min_periods=20).std(ddof=1) * ANN,
        "vol_60": vol_60(close),
        "max_ret_20": ret.rolling(20, min_periods=20).max(),
        "turnover_spike": turn.rolling(5, min_periods=5).mean() / turn.shift(5).rolling(60, min_periods=60).mean(),
        "log_mcap": np.log(mcap.where(mcap > 0)),
        "penny_flag": (close_raw < 1000).astype("float64").where(close_raw.notna()),
    }
    # 0 나누기 등으로 생긴 무한대는 결측으로 본다
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in f.items()}


def compute_market_features(bm: pd.Series, regime_mc: pd.Series, ss_regime: pd.Series) -> pd.DataFrame:
    """bm: 코스피 200 종가(인덱스 거래일). regime_mc: B의 compute_regime_mc 결과. ss_regime: 4.9절 값."""
    r = np.log(bm / bm.shift(1))
    return pd.DataFrame({
        "mkt_vol_20": r.rolling(20, min_periods=20).std(ddof=1) * ANN,
        "mkt_ret_20": bm / bm.shift(20) - 1,
        "regime_mc": regime_mc.reindex(bm.index),
        "ss_regime": ss_regime.reindex(bm.index),
    })


def ss_allowed(ss_regime: float, in_kospi200: object, in_kosdaq150: object) -> float:
    """4.9절: 0이면 1, 1이면 코스피 200 또는 코스닥 150 구성종목일 때만 1, 2이면 0. 구성종목 정보가 없으면(결측) 결측."""
    if pd.isna(ss_regime):           # 날짜표에 없는 신호일을 구성종목 규칙으로 흘려보내지 않는다
        return np.nan
    if ss_regime == 0:
        return 1.0
    if ss_regime == 2:
        return 0.0
    a = None if pd.isna(in_kospi200) else bool(in_kospi200)
    b = None if pd.isna(in_kosdaq150) else bool(in_kosdaq150)
    if a or b:                       # numpy/pandas 불리언도 다루도록 `is True`로 비교하지 않는다
        return 1.0
    if a is None or b is None:       # 한쪽 목록이 없으면 구성종목이 아니라고 단정할 수 없다
        return np.nan
    return 0.0


def rank_and_fill(df: pd.DataFrame, rank_cols=RANK_FEATURES) -> pd.DataFrame:
    """한 신호일 t의 유니버스 종목 행렬(열: 피처)에 순위 변환과 결측 처리를 적용한다.
    순위 = U_t 안의 백분위(pct, average). 결측은 0.5로 채우고 {피처}_na(0/1) 열을 만든다. 원값 피처의 결측은 여기서 채우지 않는다."""
    out = df.copy()
    for c in rank_cols:
        na = out[c].isna()
        out[c] = out[c].rank(pct=True, method="average").fillna(0.5)
        out[f"{c}_na"] = na.astype("int8")
    return out
