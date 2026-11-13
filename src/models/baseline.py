"""A-04 베이스라인 신호 (6.3절 forward test 첫 기록용 플래그, 6.4절 B1).

rank_score = mom_12_1 (6.2절 피처, U_t 안의 백분위 순위, 결측은 0.5). rank는 rank_score 내림차순 순위(1이 최고).
crash_flag = U_t에서 vol_60 상위 ceil(0.10 × N_t)개. 학습이 없고 단일 변수 로지스틱 회귀와 순위가 같다.
crash_prob는 확률이 아니라 vol_60의 U_t 내 백분위(0~1)다. 베이스라인에는 보정된 확률이 없어, 확률이 필요한 소비자(B의 RL 관측)에게는
순위 점수로 전달한다 (model_version = "baseline"로 구분한다).
t일 종가까지의 가격만 쓴다 (R-03).
"""
import argparse
import json
import math

import numpy as np
import pandas as pd

from src.data import io_shim
from src.features.price_features import mom_12_1, vol_60

MODEL_VERSION = "baseline"
FLAG_FRACTION = 0.10  # 부록 1 models.gate.flag_top_fraction


def baseline_signals(close: pd.DataFrame, tickers: list, t) -> pd.DataFrame:
    """close: 거래일 × ticker 수정종가 (t 이후 행이 있어도 쓰지 않는다). tickers: 신호일 t의 U_t."""
    t = pd.Timestamp(t)
    c = close.loc[:t, [x for x in tickers if x in close.columns]]
    missing = sorted(set(tickers) - set(c.columns))
    if missing:
        c = c.join(pd.DataFrame(np.nan, index=c.index, columns=missing))   # 시세가 없는 종목도 U_t에 있으면 결측으로 포함
    mom = mom_12_1(c).loc[t]
    vol = vol_60(c).loc[t]
    n = len(c.columns)
    df = pd.DataFrame({"ticker": list(c.columns), "mom": mom.values, "vol": vol.values})
    df["rank_score"] = df["mom"].rank(pct=True, method="average").fillna(0.5).astype("float64")
    df["crash_prob"] = df["vol"].rank(pct=True, method="average").fillna(0.5).astype("float64")
    ordered = df.sort_values(["rank_score", "ticker"], ascending=[False, True])      # 동점은 종목코드 순으로 정해 결과를 고정한다
    df["rank"] = pd.Series(np.arange(1, n + 1), index=ordered.index).reindex(df.index).astype("int32")
    k = math.ceil(FLAG_FRACTION * n)
    order = df.dropna(subset=["vol"]).sort_values(["vol", "ticker"], ascending=[False, True])
    flagged = set(order["ticker"].iloc[:k])
    df["crash_flag"] = df["ticker"].isin(flagged).astype("int8")
    df.insert(0, "date", t)
    df["model_version"] = MODEL_VERSION
    return df[["date", "ticker", "rank_score", "rank", "crash_prob", "crash_flag", "model_version"]].sort_values("rank").reset_index(drop=True)


def run_step(ctx) -> None:
    """D의 run_weekly(5.2절 3단계, M3a·M3b) entry: "src.models.baseline:run_step".
    A-08 최종 모델(generate_signals)이 나오기 전까지 이 베이스라인으로 signals를 만든다.
    ctx는 StepContext(D 소유)와 같은 모양만 있으면 된다: read_table(name), write_table(name, df, date=...)."""
    t = pd.Timestamp(ctx.date)
    universe = ctx.read_table("universe")
    u_t = universe.loc[(universe["date"] == t) & universe["in_universe"], "ticker"].tolist()
    if not u_t:
        raise ValueError(f"{ctx.date}: universe에 그 신호일이 없거나 U_t가 비어 있다")
    prices = ctx.read_table("prices")
    close = prices.pivot(index="date", columns="ticker", values="close")
    signals = baseline_signals(close, u_t, t)
    ctx.write_table("signals", signals, date=ctx.date)


class _IoCtx:
    """run_step을 run_weekly.py 없이 수동으로 돌릴 때 쓰는 최소 ctx (io_shim 기반)."""

    def __init__(self, date: str):
        self.date = date

    def read_table(self, name):
        return io_shim.read(name)

    def write_table(self, name, df, date=None):
        return io_shim.write(name, df, date=date)


def main(argv=None):
    ap = argparse.ArgumentParser(description="A-04 베이스라인 신호를 신호일 하나에 대해 수동으로 만든다")
    ap.add_argument("--date", required=True, help="신호일 YYYY-MM-DD (universe에 있어야 한다)")
    a = ap.parse_args(argv)
    ctx = _IoCtx(a.date)
    run_step(ctx)
    print(json.dumps({"date": a.date, "path": f"signals/{a.date}.parquet",
                      "io_available": io_shim.IO_AVAILABLE}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
