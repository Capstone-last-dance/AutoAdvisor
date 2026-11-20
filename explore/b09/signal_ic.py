r"""신호 예측력 확인: 연도별 rank IC를 교차 적합·검증·시험 칸으로 나눠 비교한다 (11/20 사용자 결정, A 11/19 답장 후속).

결과 읽기다 (학습·엔진 없음, 1~2분).
  .\.venv\Scripts\python.exe -I explore\b09\run.py signal_ic.py

- 신호: 시험 = A v7 시험 신호(20261117_A_wfnews, 2016~2025, walk-forward).
  폴드 Y의 학습용 신호 = 20261117_A_rlsig_y{Y} (2011 ~ Y−1년 11~12월). A 설명(11/19)에 따라 칸을 나눈다:
  연도 y ≤ Y−3은 '교차'(leave-one-year-out 교차 적합: y를 뺀 나머지 학습 연도로 적합한 모델이 예측),
  y = Y−2·Y−1은 '검증'(학습 연도 밖, 폴드 Y의 하이퍼파라미터를 고른 해).
- 라벨: C 공식 라벨 `files/C02_labels_20261109_preholdout/labels.parquet` (2011~2025, 2026 없음). y_rank = U_t 안에서
  초과수익 ER(t,h)의 백분위. h = 20(주 예측 기간, 신호 다음 날 종가부터 20거래일)과 5.
- rank IC = 신호일마다 rank_score와 y_rank의 순위 상관(Spearman). 작은 종목 쏠림 = rank_score와 시가총액 크기
  (universe.parquet mcap_rank의 반대)의 순위 상관 (음수면 작은 종목을 높게 본다).
  상위 20 초과 = rank_score 상위 20종목의 5일 수익 평균 − 그날 라벨이 있는 전체 평균 (버퍼 규칙이 사는 자리).
- 20일 라벨은 주 단위 신호일끼리 기간이 겹치므로, 주별 IC의 평균만 보고 통계 검정은 하지 않는다.
- 결과: signal_ic_dates.parquet (신호일별), signal_ic_summary.csv (칸 × 연도), signal_ic.json.
"""
from __future__ import annotations

import json

from common import HERE, REPO, SIG_RUN
import numpy as np
import pandas as pd
from src.common import io

RLSIG = "20261117_A_rlsig"
LABELS = REPO / "files" / "C02_labels_20261109_preholdout" / "labels.parquet"
YEARS = tuple(range(2016, 2026))


def load_sources() -> pd.DataFrame:
    frames = []
    s = io.read_table("signals", run_id=SIG_RUN)
    frames.append(s.assign(src="시험", fold=0)[["src", "fold", "date", "ticker", "rank_score"]])
    for y in YEARS:
        s = io.read_table("signals", run_id=f"{RLSIG}_y{y}")
        frames.append(s.assign(src=f"y{y}", fold=y)[["src", "fold", "date", "ticker", "rank_score"]])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str)
    yr = out["date"].dt.year
    out["cell"] = np.where(out["src"] == "시험", "시험", np.where(yr <= out["fold"] - 3, "교차", "검증"))
    return out


def per_date(df: pd.DataFrame) -> pd.DataFrame:
    """신호일별 지표. 순위 상관 = 그룹 안 순위끼리의 피어슨 상관."""
    def corr(a: pd.Series, b: pd.Series) -> float:
        m = a.notna() & b.notna()
        if m.sum() < 10:
            return np.nan
        return float(np.corrcoef(a[m].rank(), b[m].rank())[0, 1])

    rows = []
    for (src, date), g in df.groupby(["src", "date"], sort=False):
        top = g.nsmallest(20, "rank_order") if "rank_order" in g else g.nlargest(20, "rank_score")
        lab5 = g["ret_5"].notna()
        rows.append({"src": src, "fold": int(g["fold"].iloc[0]), "cell": g["cell"].iloc[0], "date": date,
                     "n": int(len(g)), "ic20": corr(g["rank_score"], g["y_rank_20"]), "ic5": corr(g["rank_score"], g["y_rank_5"]),
                     "size_tilt": corr(g["rank_score"], -g["mcap_rank"]),
                     "top20_excess5": float(top.loc[top["ret_5"].notna(), "ret_5"].mean() - g.loc[lab5, "ret_5"].mean())
                     if lab5.sum() >= 10 else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    src = load_sources()
    lab = pd.read_parquet(LABELS, columns=["date", "ticker", "h", "ret", "y_rank"])
    lab["date"] = pd.to_datetime(lab["date"])
    lab["ticker"] = lab["ticker"].astype(str)
    wide = lab[lab["h"].isin([5, 20])].pivot_table(index=["date", "ticker"], columns="h", values=["ret", "y_rank"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    u = pd.read_parquet(REPO / "data" / "processed" / "universe.parquet", columns=["date", "ticker", "mcap_rank"])
    u["date"] = pd.to_datetime(u["date"])
    u["ticker"] = u["ticker"].astype(str)
    df = src.merge(wide, on=["date", "ticker"], how="left").merge(u, on=["date", "ticker"], how="left")
    d = per_date(df)
    d.to_parquet(HERE / "signal_ic_dates.parquet", index=False)
    d["year"] = d["date"].dt.year

    summ = d.groupby(["cell", "year"]).agg(ic20=("ic20", "mean"), ic5=("ic5", "mean"), size_tilt=("size_tilt", "mean"),
                                           top20_excess5=("top20_excess5", "mean"), folds=("fold", "nunique"),
                                           weeks=("date", "nunique")).reset_index()
    summ.to_csv(HERE / "signal_ic_summary.csv", index=False)
    same = range(2016, 2023)  # 세 칸이 모두 있는 해
    by_cell = {c: {k: float(d[(d.cell == c) & d.year.isin(same)][k].mean()) for k in ("ic20", "ic5", "size_tilt", "top20_excess5")}
               for c in ("교차", "검증", "시험")}
    early = {c: {k: float(d[(d.cell == c) & (d.year <= 2015)][k].mean()) for k in ("ic20", "size_tilt", "top20_excess5")}
             for c in ("교차", "검증")}
    by_fold = d[(d.cell == "교차") & (d.year <= 2013)].groupby("fold")["ic20"].mean()
    (HERE / "signal_ic.json").write_text(json.dumps({"same_years_2016_2022": by_cell, "years_2011_2015": early,
                                                     "cross_ic20_2011_2013_by_fold": {int(k): float(v) for k, v in by_fold.items()},
                                                     "test_ic20_by_year": {int(y): float(v) for y, v in
                                                                           d[d.cell == "시험"].groupby("year")["ic20"].mean().items()}},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")

    pd.set_option("display.width", 250)
    pv = summ.pivot(index="year", columns="cell", values="ic20")[["교차", "검증", "시험"]]
    pv5 = summ.pivot(index="year", columns="cell", values="top20_excess5")[["교차", "검증", "시험"]]
    pz = summ.pivot(index="year", columns="cell", values="size_tilt")[["교차", "검증", "시험"]]
    print("[연도별 rank IC (20일), 칸 평균] 교차 = 교차 적합, 검증 = 하이퍼파라미터를 고른 해, 시험 = walk-forward")
    print(pv.round(3).to_string())
    print("\n[연도별 상위 20종목 5일 초과 수익 %, 칸 평균]")
    print((pv5 * 100).round(2).to_string())
    print("\n[연도별 작은 종목 쏠림 (rank_score와 시가총액 크기의 순위 상관, 음수 = 작은 종목을 높게 봄)]")
    print(pz.round(3).to_string())
    print("\n[같은 해 2016~2022 평균]")
    for c, v in by_cell.items():
        print(f"  {c}: IC20 {v['ic20']:+.3f}, IC5 {v['ic5']:+.3f}, 상위 20 5일 초과 {v['top20_excess5'] * 100:+.2f}%, 작은 종목 쏠림 {v['size_tilt']:+.3f}")
    print("[2011~2015 평균] " + ", ".join(f"{c}: IC20 {v['ic20']:+.3f}, 쏠림 {v['size_tilt']:+.3f}" for c, v in early.items()))
    print("[같은 2011~2013을 교차 적합으로 예측한 IC20, 폴드별 (학습 연도가 늘어나는 순)] "
          + ", ".join(f"y{k} {v:+.3f}" for k, v in by_fold.items()))
    print("done")


if __name__ == "__main__":
    main()
