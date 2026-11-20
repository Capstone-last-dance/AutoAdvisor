"""성능 트랙 신호 (11/19): A 베이스라인 신호(B1)를 전 신호일에 만든다. A의 `src/models/baseline.baseline_signals`와 같은 계산이다.

  python -I explore/b09/run.py sig_b1.py check      # 표본 신호일에서 A 함수 결과와 같은지 (rank·rank_score·crash_prob·crash_flag)

- b1  : rank_score = mom_12_1의 U_t 백분위(결측 0.5), rank = rank_score 내림차순(동점은 종목코드), crash_prob = vol_60 백분위,
        crash_flag = vol_60 상위 ceil(0.10 × N_t). t일 종가까지의 가격만 쓴다 (R-03). 학습이 없어 교차 적합이 필요 없다.
- b1ml: rank·rank_score는 b1, crash_prob·crash_flag와 나머지 열은 ML 신호 그대로 (학습 구간 rlsig, 시험 wfnews).
- 이 모듈은 common.py를 부르지 않는다 (common.Ctx와 rl_screen.Base가 이 모듈을 부른다).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[2])  # 저장소 루트 (단독 실행 때도 src를 찾게)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import pandas as pd

from src.features.price_features import mom_12_1, vol_60
from src.models.baseline import FLAG_FRACTION

B1_COLS = ["date", "ticker", "rank_score", "rank", "crash_prob", "crash_flag"]
_CACHE: dict = {}


def _factors(prices: pd.DataFrame):
    key = id(prices)
    if key not in _CACHE:
        close = prices.pivot(index="date", columns="ticker", values="close")
        _CACHE[key] = (mom_12_1(close), vol_60(close))
    return _CACHE[key]


def b1_one(mom_t: pd.Series, vol_t: pd.Series, tickers: list, t) -> pd.DataFrame:
    """baseline_signals와 같은 순서·같은 연산 (U_t 종목 중 시세가 없는 것은 결측으로 둔다)."""
    have = [x for x in tickers if x in mom_t.index]
    cols = have + sorted(set(tickers) - set(have))
    df = pd.DataFrame({"ticker": cols, "mom": mom_t.reindex(cols).values, "vol": vol_t.reindex(cols).values})
    n = len(cols)
    df["rank_score"] = df["mom"].rank(pct=True, method="average").fillna(0.5).astype("float64")
    df["crash_prob"] = df["vol"].rank(pct=True, method="average").fillna(0.5).astype("float64")
    ordered = df.sort_values(["rank_score", "ticker"], ascending=[False, True])
    df["rank"] = pd.Series(np.arange(1, n + 1), index=ordered.index).reindex(df.index).astype("int32")
    k = math.ceil(FLAG_FRACTION * n)
    order = df.dropna(subset=["vol"]).sort_values(["vol", "ticker"], ascending=[False, True])
    df["crash_flag"] = df["ticker"].isin(set(order["ticker"].iloc[:k])).astype("int8")
    df.insert(0, "date", pd.Timestamp(t))
    return df[B1_COLS].sort_values("rank").reset_index(drop=True)


def b1_table(prices: pd.DataFrame, universe: pd.DataFrame, dates) -> pd.DataFrame:
    mom, vol = _factors(prices)
    u = universe.loc[universe["in_universe"].astype(bool)]
    by = {pd.Timestamp(d): g["ticker"].tolist() for d, g in u.groupby(pd.to_datetime(u["date"]))}
    out = []
    for d in dates:
        t = pd.Timestamp(d)
        out.append(b1_one(mom.loc[t], vol.loc[t], by[t], t))
    return pd.concat(out, ignore_index=True)


def _keys(df: pd.DataFrame) -> pd.MultiIndex:
    """자료형(datetime 단위, 문자열 dtype)과 관계없이 비교할 (날짜 문자열, 종목코드 문자열) 키."""
    return pd.MultiIndex.from_arrays([pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
                                      df["ticker"].astype(str).to_numpy()])


def apply_sig(sig: str, ml: pd.DataFrame, b1: pd.DataFrame) -> pd.DataFrame:
    """ml(날짜·종목 = U_t)을 sig 판으로 바꾼다. 날짜 집합과 종목이 같아야 한다. 날짜·종목 열의 자료형은 ml에 맞춘다."""
    ml = ml.copy()
    ml["date"] = pd.to_datetime(ml["date"])
    b1 = b1.copy()
    b1["date"] = pd.to_datetime(b1["date"]).astype(ml["date"].dtype)
    b1["ticker"] = b1["ticker"].astype(str)
    b1 = b1[b1["date"].isin(ml["date"].unique())].reset_index(drop=True)
    km, kb = _keys(ml), _keys(b1)
    if len(km) != len(kb) or km.has_duplicates or not km.sort_values().equals(kb.sort_values()):
        raise ValueError("B1 신호와 ML 신호의 (date, ticker)가 다르다")
    if sig == "b1":
        b1["ticker"] = b1["ticker"].astype(ml["ticker"].dtype)
        return b1
    if sig == "b1ml":
        add = b1[["date", "ticker", "rank", "rank_score"]].rename(columns={"ticker": "_k"})
        m = (ml.drop(columns=["rank", "rank_score"]).assign(_k=ml["ticker"].astype(str))
             .merge(add, on=["date", "_k"], how="left", validate="one_to_one").drop(columns="_k"))
        if m["rank"].isna().any():
            raise ValueError("B1 rank가 비는 행이 있다")
        return m
    raise ValueError(f"모르는 신호 판: {sig}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from src.common import io
    from src.models.baseline import baseline_signals
    from src.portfolio.env_check_report import FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE, kospi200, signal_dates

    if sys.argv[1:] != ["check"]:
        raise SystemExit("usage: sig_b1.py check")
    prices, universe = io.read_table("prices"), io.read_table("universe")
    bm = kospi200(io.read_table("index"))
    dates = signal_dates(universe, bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)
    june = {}
    for d in dates:  # 해마다 6월 첫 신호일
        if d[5:7] == "06":
            june.setdefault(d[:4], d)
    pick = [dates[0], dates[1]] + sorted(june.values()) + [dates[-1]]
    close = prices.pivot(index="date", columns="ticker", values="close")
    mine = b1_table(prices, universe, pick)
    bad = 0
    for d in pick:
        t = pd.Timestamp(d)
        u_t = universe.loc[(pd.to_datetime(universe["date"]) == t) & universe["in_universe"].astype(bool), "ticker"].tolist()
        a = baseline_signals(close, u_t, t).sort_values("ticker").reset_index(drop=True)
        b = mine[mine["date"] == t].sort_values("ticker").reset_index(drop=True)
        same = (len(a) == len(b) and (a["ticker"].values == b["ticker"].values).all()
                and (a["rank"].values == b["rank"].values).all() and (a["crash_flag"].values == b["crash_flag"].values).all()
                and np.array_equal(a["rank_score"].values, b["rank_score"].values)
                and np.array_equal(a["crash_prob"].values, b["crash_prob"].values))
        bad += not same
        print(d, "N", len(b), "flags", int(b["crash_flag"].sum()), "같음" if same else "다름", flush=True)
    print(f"표본 {len(pick)}개 중 다름 {bad}개")
    if bad:
        raise SystemExit(1)
