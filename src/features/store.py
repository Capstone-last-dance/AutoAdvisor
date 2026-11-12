"""A-05 피처 스토어 조립 (6.2절). 신호일별로 U_t 종목의 피처 행렬을 만든다.

- 순위 변환·결측 처리는 U_t 안에서 한다 (price_features.rank_and_fill).
- ss_regime(4.9절 날짜)은 부록 3 Q-11 검증 전이라 이 모듈에 값을 넣지 않고 입력으로 받는다 (R-08).
- regime_mc는 B의 compute_regime_mc 결과를 입력으로 받는다 (R-17). 없으면 stub_regime_mc를 쓰고, 스텁이 쓰였음을 호출자가 알아야 한다.
- 원값 피처(penny_flag, 시장 공통)에 결측이 생기면 임의로 채우지 않고 예외를 낸다 (6.2절: [결정 필요]).
- ss_allowed는 예외다: 구성종목 원천이 없다(이슈 #5). PM 결정(2026-11-04)에 따라 결측을 그대로 저장하고,
  v1 모델(gate·ranker) 입력에서만 뺀다(`src/models/gate.py`의 `NON_FEATURES`). ss_regime은 그대로 둔다.
"""
import pandas as pd

from src.features.price_features import (MARKET_FEATURES, RANK_FEATURES, RAW_FEATURES, compute_market_features,
                                         compute_stock_features, rank_and_fill, ss_allowed)

STOCK_RAW = ["penny_flag"]


def stub_regime_mc(index: pd.DatetimeIndex) -> pd.Series:
    """B의 compute_regime_mc가 없을 때의 스텁. 항상 0. 이 값으로 만든 피처는 _stub/에만 쓴다."""
    return pd.Series(0, index=index, dtype="int8")


def compute_feature_matrix(close, close_raw, value, mcap, bm: pd.Series,
                           regime_mc: pd.Series, ss_regime: pd.Series, universe: dict,
                           in_kospi200: pd.DataFrame | None = None, in_kosdaq150: pd.DataFrame | None = None) -> pd.DataFrame:
    """universe: {신호일 t: U_t 종목 리스트}. in_kospi200/in_kosdaq150: 신호일 × ticker nullable boolean (없으면 결측).

    구성종목 정보가 없는 (t, ticker)는 ss_allowed가 결측이다(원천 자체가 없음, 이슈 #5). PM 결정(2026-11-04)에
    따라 이 결측은 예외를 내지 않고 그대로 저장한다 — v1 모델 입력에서 빼는 건 호출자(gate.py NON_FEATURES)의 몫이다."""
    stock = compute_stock_features(close, close_raw, value, mcap)
    market = compute_market_features(bm, regime_mc, ss_regime)
    out = []
    for t in sorted(universe):
        tk = list(universe[t])
        if not tk:
            continue
        df = pd.DataFrame({name: stock[name].loc[t].reindex(tk) for name in RANK_FEATURES + STOCK_RAW})
        df = rank_and_fill(df)
        regime = market.loc[t, "ss_regime"]
        k2 = in_kospi200.loc[t].reindex(tk) if in_kospi200 is not None and t in in_kospi200.index else pd.Series(pd.NA, index=tk)
        q1 = in_kosdaq150.loc[t].reindex(tk) if in_kosdaq150 is not None and t in in_kosdaq150.index else pd.Series(pd.NA, index=tk)
        df["ss_allowed"] = [ss_allowed(regime, a, b) for a, b in zip(k2, q1)]
        for m in MARKET_FEATURES:
            df[m] = market.loc[t, m]
        df.insert(0, "ticker", df.index.astype(str))
        df.insert(0, "date", t)
        out.append(df.reset_index(drop=True))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    bad = {c: int(res[c].isna().sum()) for c in STOCK_RAW + MARKET_FEATURES if len(res) and res[c].isna().any()}
    if bad:
        raise ValueError(f"[결정 필요] 원값 피처에 결측이 있다 (6.2절): {bad}")
    return res


def merge_features(a: pd.DataFrame, pm: pd.DataFrame | None) -> pd.DataFrame:
    """A의 피처와 PM의 이벤트·뉴스 피처를 (date, ticker)로 합친다. PM이 U_t 밖 종목을 넣었거나 U_t 종목을 빠뜨리면 예외."""
    if pm is None:
        return a
    dup = (set(a.columns) & set(pm.columns)) - {"date", "ticker"}
    if dup:
        raise ValueError(f"A와 PM 피처 이름이 겹친다: {sorted(dup)} (R-14)")
    key = ["date", "ticker"]
    extra = pm.merge(a[key], on=key, how="left", indicator=True).query("_merge == 'left_only'")
    if len(extra):
        raise ValueError(f"PM 피처에 U_t 밖 종목 행이 {len(extra)}건 있다 (6.2절 완료 기준 (3))")
    merged = a.merge(pm, on=key, how="left", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError(f"PM 피처에 없는 U_t 종목이 {int((merged['_merge'] != 'both').sum())}건 있다")
    return merged.drop(columns="_merge")


def missing_rate_table(features: pd.DataFrame) -> pd.DataFrame:
    """6.2절 완료 기준 (1): 피처별 결측률. 순위 피처는 {피처}_na 평균, 원값 피처는 NaN 비율."""
    rows = [(f, float(features[f"{f}_na"].mean())) for f in RANK_FEATURES if f"{f}_na" in features]
    rows += [(f, float(features[f].isna().mean())) for f in RAW_FEATURES + MARKET_FEATURES if f in features]
    return pd.DataFrame(rows, columns=["feature", "missing_rate"]).set_index("feature")


def missing_by_year(features: pd.DataFrame) -> pd.DataFrame:
    """연도별 결측률 (이력 부족이 초기에 몰리는지 본다)."""
    y = pd.to_datetime(features["date"]).dt.year
    cols = [f"{f}_na" for f in RANK_FEATURES if f"{f}_na" in features]
    return features[cols].groupby(y).mean().rename(columns=lambda c: c[:-3])


def m2_step(ctx) -> None:
    """run_weekly 2단계: 신호일 t의 피처 행렬 (6.2절, A-05).

    A의 `build_features.build`는 1단계가 쓴 `prices`·`index`·`universe`(t까지만 있다, R-03)에서
    전체 신호일 피처를 만든다 — 신호일마다 다시 부르지 않고 t 행만 추려 쓴다.

    PM 피처는 `merge_features(a, pm)`로 합칠 자리가 이미 있지만, PM이 이 단계에 넘길 표(공시·뉴스
    피처)를 아직 정하지 않았다(issue #5, 1단계와 같은 질문). 그래서 지금은 `pm=None`으로 A 피처만
    쓴다 — 지금 forward test 신호도 PM 피처 없이 돌고 있어 결과가 달라지지 않는다(R-08). PM이
    표 이름·시점을 정하면 여기서 `ctx.read_table`로 읽어 넘기기만 하면 된다."""
    from src.features.build_features import build as build_feature_matrix

    prices, index, universe = ctx.read_table("prices"), ctx.read_table("index"), ctx.read_table("universe")
    features, _ = build_feature_matrix(prices, index, universe)
    t = features[features["date"] == pd.Timestamp(ctx.date)].reset_index(drop=True)
    if t.empty:
        raise ValueError(f"신호일 {ctx.date}의 피처 행이 없다 (universe에 그 신호일이 있는지 확인)")
    ctx.write_table("features", merge_features(t, pm=None), date=ctx.date)
