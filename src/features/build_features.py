"""A-05 피처 행렬 생성 (6.2절). 9.3절 정제 테이블에서 신호일 × U_t 종목의 피처를 만든다.

    python3 -m src.features.build_features --out data/interim/features.parquet

입력은 공용 I/O(`src/common/io.py`)로 읽는 prices·index·universe다. `regime_mc`는 B의
`compute_regime_mc`(6.9절 Step 5), `ss_regime`은 4.9절 날짜표를 쓴다. PM 피처(공시·뉴스)는
`store.merge_features`로 나중에 합친다 — 이 모듈은 A 피처까지만 만든다.

[결정 필요] 4.9절 공매도 레짐 표에서 **2009-06-01~2011-08-09, 2011-11-10~2013-11-13 두 구간이
"미확정"**이다(비금융주 허용·금융주 금지). 학습 구간 신호일의 약 3년이 여기 걸리는데 시장 공통
피처라 결측이면 `compute_feature_matrix`가 멈춘다. 그래서 `UNDECIDED_REGIME`로 값을 모아 두고
기본값을 0(전 종목 허용에 가장 가까움)으로 둔다. PM 확정 전까지는 가정이며, 산출물의 meta에
그 사실과 영향 받은 신호일 수를 적는다. 1(코스피200·코스닥150만 허용)은 훨씬 좁은 구간이라 쓰지 않는다.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import io
from src.features.store import compute_feature_matrix
from src.portfolio.risk_layer import compute_regime_mc

UNDECIDED_REGIME = 0   # [결정 필요] 금융주만 금지된 구간의 ss_regime (PM 확정 대기)

# 4.9절 표. (시작일, 끝일, ss_regime). None은 표가 "미확정"으로 둔 구간이다.
SS_REGIME_TABLE = (
    ("2008-10-01", "2009-05-31", 2),
    ("2009-06-01", "2011-08-09", None),
    ("2011-08-10", "2011-11-09", 2),
    ("2011-11-10", "2013-11-13", None),
    ("2013-11-14", "2020-03-15", 0),
    ("2020-03-16", "2021-05-02", 2),
    ("2021-05-03", "2023-11-05", 1),
    ("2023-11-06", "2025-03-30", 2),
    ("2025-03-31", None, 0),
)


def ss_regime_series(days) -> tuple:
    """거래일마다 4.9절 ss_regime. (값 Series, 미확정 구간에서 가정으로 채운 거래일 수)를 돌려준다."""
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(list(days))))
    out = pd.Series(pd.NA, index=idx, dtype="Int8")
    assumed = 0
    for start, end, value in SS_REGIME_TABLE:
        mask = idx >= pd.Timestamp(start)
        if end is not None:
            mask &= idx <= pd.Timestamp(end)
        out[mask] = UNDECIDED_REGIME if value is None else value
        if value is None:
            assumed += int(mask.sum())
    if out.isna().any():
        missing = idx[out.isna()]
        raise ValueError(f"4.9절 표가 덮지 못한 거래일 {len(missing)}개: {missing.min().date()}~{missing.max().date()}")
    return out.astype("int8"), assumed


def regime_mc_series(bm: pd.Series, dates) -> pd.Series:
    """신호일마다 B의 compute_regime_mc. 시장 피처는 신호일 행만 쓰이므로 신호일만 계산한다."""
    values = {pd.Timestamp(t): int(compute_regime_mc(t, bm_close=bm)) for t in dates}
    return pd.Series(values, dtype="int8").reindex(bm.index)


def build(prices: pd.DataFrame, index: pd.DataFrame, universe: pd.DataFrame) -> tuple:
    """A 피처 행렬과 meta를 돌려준다."""
    wide = {col: prices.pivot(index="date", columns="ticker", values=col) for col in ("close", "close_raw", "value", "mcap")}
    bm = index.loc[index["index_code"] == "KOSPI200"].set_index("date")["close"].sort_index()
    members = universe.loc[universe["in_universe"], ["date", "ticker"]]
    u = {pd.Timestamp(t): g["ticker"].tolist() for t, g in members.groupby("date", sort=True)}
    ss, assumed_days = ss_regime_series(bm.index)
    features = compute_feature_matrix(wide["close"], wide["close_raw"], wide["value"], wide["mcap"], bm,
                                      regime_mc_series(bm, u), ss, u)
    assumed_signals = sum(1 for t in u if _is_undecided(t))
    meta = {"signal_dates": len(u), "rows": len(features),
            "ss_regime_assumed_trading_days": assumed_days, "ss_regime_assumed_signal_dates": assumed_signals,
            "ss_regime_assumption": f"4.9절 미확정 구간을 {UNDECIDED_REGIME}으로 두었다 (PM 확정 대기)"}
    return features, meta


def _is_undecided(t) -> bool:
    t = pd.Timestamp(t)
    return any(value is None and pd.Timestamp(start) <= t <= (pd.Timestamp(end) if end else pd.Timestamp.max)
               for start, end, value in SS_REGIME_TABLE)


def main(argv=None):
    ap = argparse.ArgumentParser(description="A-05 피처 행렬을 만든다")
    ap.add_argument("--out", default="data/interim/features.parquet")
    a = ap.parse_args(argv)
    features, meta = build(io.read_table("prices"), io.read_table("index"), io.read_table("universe"))
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    meta["path"] = str(path)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
