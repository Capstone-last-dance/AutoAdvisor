"""A-06·A-07 walk-forward 러너 (6.3·6.4절). 10개 폴드의 M3a·M3b 예측을 한 파일로 묶는다.

    python3 -m src.models.walk_forward --run-id 20261109_A_wf --features data/interim/features.parquet \
        --labels data/interim/labels.parquet --end-dates data/interim/label_end_dates.parquet

산출물
- `signals/wf_{run_id}.parquet` — 9.3절 signals 열. 시험 연도 2016~2025의 모든 신호일 × U_t 종목.
  `model_version`은 `{run_id}_y{시험연도}`라 어느 폴드 모델이 낸 행인지 남는다.
- `reports/{run_id}/metrics.json` — 폴드별 선택 하이퍼파라미터와 검증 점수, 연도별·전체 ROC-AUC·lift@10%
  (6.3절 완료 기준), 연도별·전체 rank IC와 HAC t값, 베이스라인 B1·B2·B3 비교 (6.4절 완료 기준).
  지표는 전부 C의 M5 함수(`src.backtest.metrics`)로 계산한다.
- `reports/{run_id}/wf_diagnostics.parquet` — 공용 스키마 밖의 진단 열(raw_prob, 게이트 베이스라인,
  B1·B2·B3 점수). signals 스키마에 없는 열은 여기 둔다.
- `reports/{run_id}/figures/calibration_{연도}.png` — 보정 곡선 (6.3절 완료 기준)
- `reports/{run_id}/models/fold_{연도}/` — 폴드별 모델 파일 (M12 SHAP 입력)

라벨은 2025-12-31까지만 쓴다. 홀드아웃(2026-01-01~)은 C의 가드가 막고 이 러너도 읽지 않는다 (R-04).
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.backtest import metrics as m5
from src.common import io
from src.models import gate, ranker
from src.models.splits import label_end_dates, make_folds

SIGNAL_COLUMNS = ["date", "ticker", "rank_score", "rank", "crash_prob", "crash_flag", "model_version"]
DIAGNOSTIC_COLUMNS = ["test_year", "date", "ticker", "raw_prob", "base_prob", "base_flag", "b1_score", "b2_score", "b3_score"]


def model_hashes(directory: Path) -> dict:
    """저장한 모델 파일의 SHA-256. 가중치는 커밋하지 않고 팀 드라이브에 두므로 해시를 여기 남긴다
    (PM 2026-11-10 결정). 키는 directory 기준 상대 경로다."""
    out = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            out[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def fold_signals(g: gate.FoldResult, r: ranker.RankerResult, model_version: str) -> tuple:
    """한 폴드의 게이트·순위 예측을 9.3절 signals 행과 진단 행으로 나눈다."""
    key = ["date", "ticker"]
    both = g.predictions.merge(r.predictions.drop(columns="test_year"), on=key, validate="one_to_one")
    if len(both) != len(g.predictions) or len(both) != len(r.predictions):
        raise ValueError(f"게이트 {len(g.predictions)}행과 순위 {len(r.predictions)}행의 (date, ticker)가 다르다")
    signals = both[["date", "ticker", "rank_score", "rank", "crash_prob", "crash_flag"]].copy()
    signals["model_version"] = model_version
    return signals[SIGNAL_COLUMNS], both[DIAGNOSTIC_COLUMNS]


def fold_metrics(g: gate.FoldResult, r: ranker.RankerResult, labels: pd.DataFrame, h: int) -> dict:
    """6.3·6.4절 완료 기준의 지표를 C의 M5 함수로 계산한다. 라벨이 없는 시험 연도는 None이 된다."""
    key = ["date", "ticker"]
    crash = gate.crash_labels(labels)[key + ["y_crash"]]
    rank = ranker.rank_labels(labels, h)[key + ["er"]]
    gm = g.predictions.merge(crash, on=key, how="inner")
    rm = r.predictions.merge(rank, on=key, how="inner")
    ic = m5.rank_ic_with_hac(rm.rename(columns={"rank_score": "rank_score"})[["date", "rank_score", "er"]]) if len(rm) else None
    out = {
        "test_year": g.test_year,
        "gate": {"params": g.params, "val_roc_auc": g.val_auc, "labelled_rows": len(gm),
                 "test_roc_auc": m5.roc_auc(gm["y_crash"], gm["crash_prob"]) if len(gm) else None,
                 "test_lift_at_10pct": m5.lift_at_10pct(gm["crash_flag"], gm["y_crash"]) if len(gm) else None,
                 "baseline_roc_auc": m5.roc_auc(gm["y_crash"], gm["base_prob"]) if len(gm) else None,
                 "baseline_lift_at_10pct": m5.lift_at_10pct(gm["base_flag"], gm["y_crash"]) if len(gm) else None},
        "ranker": {"params": r.params, "val_mean_rank_ic": r.val_ic, "ridge_alpha": r.ridge_alpha,
                   "labelled_rows": len(rm),
                   "test_mean_rank_ic": None if ic is None else ic.mean_ic,
                   "test_hac_t": None if ic is None else ic.hac_t_statistic},
    }
    for name, col in (("B1", "b1_score"), ("B2", "b2_score"), ("B3", "b3_score")):
        if len(rm):
            b = m5.rank_ic_with_hac(rm[["date", col, "er"]].rename(columns={col: "rank_score"}))
            out["ranker"][f"{name}_mean_rank_ic"], out["ranker"][f"{name}_hac_t"] = b.mean_ic, b.hac_t_statistic
    return out


def overall_metrics(signals: pd.DataFrame, diagnostics: pd.DataFrame, labels: pd.DataFrame, h: int) -> dict:
    """전체 구간(시험 연도 전부) 지표. 연도별 표와 함께 6.3·6.4절 완료 기준을 채운다."""
    key = ["date", "ticker"]
    gm = signals.merge(gate.crash_labels(labels)[key + ["y_crash"]], on=key, how="inner")
    rm = signals.merge(ranker.rank_labels(labels, h)[key + ["er"]], on=key, how="inner")
    ic = m5.rank_ic_with_hac(rm[["date", "rank_score", "er"]]) if len(rm) else None
    base = diagnostics.merge(gate.crash_labels(labels)[key + ["y_crash"]], on=key, how="inner")
    out = {"rows": len(signals), "labelled_rows": len(gm),
           "gate_roc_auc": m5.roc_auc(gm["y_crash"], gm["crash_prob"]) if len(gm) else None,
           "gate_lift_at_10pct": m5.lift_at_10pct(gm["crash_flag"], gm["y_crash"]) if len(gm) else None,
           "gate_baseline_roc_auc": m5.roc_auc(base["y_crash"], base["base_prob"]) if len(base) else None,
           "gate_baseline_lift_at_10pct": m5.lift_at_10pct(base["base_flag"], base["y_crash"]) if len(base) else None,
           "ranker_mean_rank_ic": None if ic is None else ic.mean_ic,
           "ranker_hac_t": None if ic is None else ic.hac_t_statistic}
    br = diagnostics.merge(ranker.rank_labels(labels, h)[key + ["er"]], on=key, how="inner")   # 6.4절 베이스라인 비교 (전체)
    for name, col in (("B1", "b1_score"), ("B2", "b2_score"), ("B3", "b3_score")):
        if len(br):
            b = m5.rank_ic_with_hac(br[["date", col, "er"]].rename(columns={col: "rank_score"}))
            out[f"{name}_mean_rank_ic"], out[f"{name}_hac_t"] = b.mean_ic, b.hac_t_statistic
    return out


def calibration_table(crash_prob, y_crash, bins: int = 10) -> pd.DataFrame:
    """보정 곡선용 표: 예측 확률 분위별 평균 예측 확률과 실제 급락 비율."""
    df = pd.DataFrame({"p": pd.to_numeric(crash_prob), "y": pd.to_numeric(y_crash)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    out = df.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return out.reset_index()


def _save_calibration_figure(table: pd.DataFrame, path: Path, title: str) -> bool:
    try:
        import matplotlib
        import matplotlib.font_manager
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    for family in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):   # 한글 글꼴이 없으면 영문으로 떨어진다
        if any(family == f.name for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = family
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, table["mean_pred"].max()], [0, table["mean_pred"].max()], "--", color="gray", linewidth=1)
    ax.plot(table["mean_pred"], table["observed"], "o-")
    ax.set_xlabel("mean predicted crash prob"); ax.set_ylabel("observed crash rate"); ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def run(features: pd.DataFrame, labels: pd.DataFrame, trading_days, *, run_id: str, h: int = 20,
        grid=None, fixed=None, n_jobs: int = 1, actual_end_dates=None, test_years=None,
        reports_root: Path | None = None, write_signals: bool = True, log=print) -> dict:
    """10개 폴드를 돌려 signals/wf_{run_id}.parquet과 리포트를 만든다."""
    io._check_run_id_arg(run_id)
    sig = pd.DatetimeIndex(sorted(features["date"].unique()))
    kw = {} if test_years is None else {"test_years": test_years}
    folds = make_folds(sig, trading_days, h=h, actual_end_dates=actual_end_dates, **kw)
    reports = Path(reports_root or (io.ROOT / "reports")) / run_id
    (reports / "models").mkdir(parents=True, exist_ok=True)

    all_signals, all_diag, per_fold = [], [], []
    for fold in folds:
        log(f"폴드 {fold.test_year}: 학습 {len(fold.train)} / 검증 {len(fold.val)} / 시험 {len(fold.test)} 신호일")
        g = gate.fit_fold(features, labels, fold, grid=grid, fixed=fixed, n_jobs=n_jobs)
        r = ranker.fit_fold(features, labels, fold, h=h, grid=grid, fixed=fixed, n_jobs=n_jobs)
        version = f"{run_id}_y{fold.test_year}"
        s, d = fold_signals(g, r, version)
        all_signals.append(s)
        all_diag.append(d)
        gate.save_fold(g, reports / "models" / f"fold_{fold.test_year}" / "gate")
        ranker.save_fold(r, reports / "models" / f"fold_{fold.test_year}" / "ranker")
        fm = fold_metrics(g, r, labels, h)
        fm["model_sha256"] = model_hashes(reports / "models" / f"fold_{fold.test_year}")
        per_fold.append(fm)
        log(f"  게이트 검증 AUC {g.val_auc:.4f} / 시험 AUC {fm['gate']['test_roc_auc']} "
            f"lift {fm['gate']['test_lift_at_10pct']} | 순위 검증 IC {r.val_ic:.4f} / 시험 IC {fm['ranker']['test_mean_rank_ic']}")
        labelled = g.predictions.merge(gate.crash_labels(labels)[["date", "ticker", "y_crash"]], on=["date", "ticker"], how="inner")
        if len(labelled):
            table = calibration_table(labelled["crash_prob"], labelled["y_crash"])
            table.to_csv(reports / f"calibration_{fold.test_year}.csv", index=False)
            _save_calibration_figure(table, reports / "figures" / f"calibration_{fold.test_year}.png",
                                     f"calibration {fold.test_year}")

    signals = pd.concat(all_signals, ignore_index=True).sort_values(["date", "rank"]).reset_index(drop=True)
    diagnostics = pd.concat(all_diag, ignore_index=True)
    diagnostics.to_parquet(reports / "wf_diagnostics.parquet", index=False)
    summary = {"run_id": run_id, "h": h, "folds": per_fold,
               "overall": overall_metrics(signals, diagnostics, labels, h),
               "signal_dates": int(signals["date"].nunique()), "rows": len(signals)}
    if write_signals:
        summary["signals_path"] = str(io.write_table("signals", signals, run_id=run_id))
    (reports / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="A-06·A-07 walk-forward 러너")
    ap.add_argument("--run-id", required=True, help="{YYYYMMDD}_{역할ID}_{짧은설명}")
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--end-dates", help="C의 end_dates. 주면 실제 라벨 종료일로 purge한다 (DEC-09 (3))")
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args(argv)
    features, labels = pd.read_parquet(a.features), pd.read_parquet(a.labels)
    index = io.read_table("index")
    days = pd.DatetimeIndex(sorted(index.loc[index["index_code"] == "KOSPI200", "date"].unique()))
    ends = label_end_dates(pd.read_parquet(a.end_dates), a.h) if a.end_dates else None
    summary = run(features, labels, days, run_id=a.run_id, h=a.h, n_jobs=a.n_jobs, actual_end_dates=ends)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
