"""DEC-08 RL 학습 구간 교차 적합 신호 (A-06·A-07 산출물). B의 RL 학습 구간 관측값을 만든다.

    python3 -m src.models.cross_fit --run-id 20261122_A_rlsig --features ... --labels ... --params reports/{run}/metrics.json

폴드 Y 하나가 파일 하나다: `signals/wf_{run_id}_y{시험연도}.parquet` (9.3절 signals 열).
한 파일에는 **2011년~(Y−1)년 모든 신호일 × U_t 종목**이 들어간다 (DEC-08).

행이 만들어지는 방식은 구간마다 다르다.
- **학습 연도 y (2011~Y−3)**: y를 뺀 나머지 학습 연도로 적합한 모델이 y를 예측한다. 하이퍼파라미터는
  폴드 Y에서 고른 값 그대로라 그리드가 없다. y의 앞뒤 경계를 실제 라벨 종료일 기준으로 purge한다
  (`splits.cross_fits`). 추가 적합은 모델당 75회다.
- **검증 연도 (Y−2, Y−1)**: 폴드 Y 모델(학습 구간으로 적합)의 예측에 폴드 Y 보정기를 적용한다.
- **시험 연도 Y**: 기본값은 넣지 않는다. `--include-test`로 넣으면 본 walk-forward 신호와 같아야 한다.

`crash_prob` 보정기(2026-11-09 PM 확정): 교차 적합 모델마다 **폴드 Y의 검증 연도에 낸 예측으로 isotonic을
적합**해 빼 둔 연도 y에 적용한다. 검증 연도는 폴드 Y 모델의 학습 구간 밖이고 y도 포함하지 않으므로, 빼 둔
연도의 정보가 보정기에 들어가지 않는다. 구현은 `gate.fit_fold`에 (학습 = y를 뺀 구간, 검증 = 폴드 Y의 검증
연도, 시험 = y)를 넘기는 것으로 끝난다 — 그 함수가 검증 구간 예측으로 isotonic을 적합한다.

내부 교차 적합은 예측 대상보다 뒤의 학습 연도를 쓸 수 있다. 그 시점의 실운영 재현이 아니라 RL 학습 구간
관측값을 만드는 절차다 (DEC-08).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import io
from src.models import gate, ranker
from src.models.splits import Fold, cross_fits, label_end_dates, make_folds

SIGNAL_COLUMNS = ["date", "ticker", "rank_score", "rank", "crash_prob", "crash_flag", "model_version"]


def single_point(params: dict) -> dict:
    """폴드에서 고른 하이퍼파라미터 하나만 담은 그리드 (DEC-08: 교차 적합에는 그리드가 없다)."""
    return {key: [value] for key, value in params.items()}


def gate_predictions(result: gate.FoldResult, features: pd.DataFrame) -> pd.DataFrame:
    """폴드 모델과 그 보정기로 features 행을 예측한다 (검증 연도용)."""
    raw = result.model.predict_proba(features[result.features], num_iteration=result.model.best_iteration_)[:, 1]
    out = features[["date", "ticker"]].copy()
    out["raw_prob"] = raw
    out["crash_prob"] = result.calibrator.predict(raw).astype("float64")
    out["crash_flag"] = gate.flag_top(out, "crash_prob", "raw_prob")
    return out.reset_index(drop=True)


def ranker_predictions(result: ranker.RankerResult, features: pd.DataFrame) -> pd.DataFrame:
    out = features[["date", "ticker"]].copy()
    out["rank_score"] = result.model.predict(features[result.features],
                                             num_iteration=result.model.best_iteration_).astype("float64")
    out["rank"] = ranker.rank_in_universe(out, "rank_score")
    return out.reset_index(drop=True)


def _signals(gate_rows: pd.DataFrame, ranker_rows: pd.DataFrame, model_version: str) -> pd.DataFrame:
    both = gate_rows.merge(ranker_rows, on=["date", "ticker"], validate="one_to_one")
    if len(both) != len(gate_rows) or len(both) != len(ranker_rows):
        raise ValueError(f"게이트 {len(gate_rows)}행과 순위 {len(ranker_rows)}행의 (date, ticker)가 다르다")
    both["model_version"] = model_version
    return both[SIGNAL_COLUMNS]


def run_fold(features: pd.DataFrame, labels: pd.DataFrame, fold: Fold, trading_days, *, run_id: str,
             gate_params: dict, ranker_params: dict, h: int = 20, actual_end_dates=None, n_jobs: int = 1,
             fixed=None, include_test: bool = False, log=print) -> tuple:
    """폴드 Y 하나의 RL 신호 표와 메타를 만든다. (signals, meta)."""
    grid_g, grid_r = single_point(gate_params), single_point(ranker_params)
    frames, meta_years = [], []

    log(f"폴드 {fold.test_year}: 폴드 모델 적합 (학습 {len(fold.train)} / 검증 {len(fold.val)} 신호일)")
    fold_gate = gate.fit_fold(features, labels, fold, grid=grid_g, fixed=fixed, n_jobs=n_jobs,
                              require_test=include_test)
    fold_ranker = ranker.fit_fold(features, labels, fold, h=h, grid=grid_r, fixed=fixed, n_jobs=n_jobs,
                                  require_test=include_test)

    val_rows = features[features["date"].isin(fold.val)]
    frames.append(_signals(gate_predictions(fold_gate, val_rows), ranker_predictions(fold_ranker, val_rows),
                           f"{run_id}_fold"))
    meta_years.append({"segment": "val", "years": sorted({d.year for d in fold.val}), "rows": len(val_rows),
                       "source": "폴드 모델 + 폴드 보정기"})

    for cross in cross_fits(fold, trading_days, h=h, actual_end_dates=actual_end_dates):
        part = Fold(fold.test_year, cross.train, fold.val, cross.predict)   # 검증 = 폴드 Y 검증 연도 (보정기 적합용)
        g = gate.fit_fold(features, labels, part, grid=grid_g, fixed=fixed, n_jobs=n_jobs)
        r = ranker.fit_fold(features, labels, part, h=h, grid=grid_r, fixed=fixed, n_jobs=n_jobs)
        frames.append(_signals(g.predictions.drop(columns="test_year"), r.predictions.drop(columns="test_year"),
                               f"{run_id}_cross"))
        meta_years.append({"segment": "train", "year": cross.year, "train_signal_dates": len(cross.train),
                           "rows": len(g.predictions), "val_roc_auc": g.val_auc, "val_mean_rank_ic": r.val_ic})
        log(f"  {cross.year}년: 학습 {len(cross.train)} 신호일 → {len(g.predictions)}행 "
            f"(검증 AUC {g.val_auc:.4f} / IC {r.val_ic:.4f})")

    if include_test:
        test_rows = features[features["date"].isin(fold.test)]
        frames.append(_signals(gate_predictions(fold_gate, test_rows), ranker_predictions(fold_ranker, test_rows),
                               f"{run_id}_fold"))
        meta_years.append({"segment": "test", "years": [fold.test_year], "rows": len(test_rows),
                           "source": "폴드 모델 + 폴드 보정기 (본 walk-forward와 같아야 한다)"})

    signals = pd.concat(frames, ignore_index=True).sort_values(["date", "rank"]).reset_index(drop=True)
    _check_complete(signals, features, fold, include_test)
    meta = {"test_year": fold.test_year, "run_id": f"{run_id}_y{fold.test_year}", "rows": len(signals),
            "signal_dates": int(signals["date"].nunique()), "gate_params": gate_params,
            "ranker_params": ranker_params, "segments": meta_years}
    return signals, meta


def _check_complete(signals: pd.DataFrame, features: pd.DataFrame, fold: Fold, include_test: bool) -> None:
    """폴드 파일에 2011~(Y−1)년 모든 신호일이 있고, 신호일마다 U_t 종목이 빠짐없이 있어야 한다 (DEC-08)."""
    want = pd.DatetimeIndex(fold.train).append(pd.DatetimeIndex(fold.val))
    if include_test:
        want = want.append(pd.DatetimeIndex(fold.test))
    missing_dates = sorted(set(want) - set(signals["date"]))
    if missing_dates:
        raise ValueError(f"폴드 {fold.test_year}: 신호일 {len(missing_dates)}개가 빠졌다 "
                         f"(첫 번째 {missing_dates[0]:%Y-%m-%d})")
    expected = features[features["date"].isin(want)][["date", "ticker"]]
    merged = expected.merge(signals[["date", "ticker"]], on=["date", "ticker"], how="left", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError(f"폴드 {fold.test_year}: U_t 종목 {(merged['_merge'] != 'both').sum()}개 행이 빠졌다")
    if len(signals) != len(expected):
        raise ValueError(f"폴드 {fold.test_year}: 행 수가 U_t 합계와 다르다 ({len(signals)} vs {len(expected)})")


def fold_params(metrics: dict) -> dict:
    """본 walk-forward의 metrics.json에서 폴드별로 고른 하이퍼파라미터를 읽는다."""
    return {int(f["test_year"]): {"gate": f["gate"]["params"], "ranker": f["ranker"]["params"]}
            for f in metrics["folds"]}


def run(features: pd.DataFrame, labels: pd.DataFrame, trading_days, *, run_id: str, params: dict, h: int = 20,
        actual_end_dates=None, n_jobs: int = 1, fixed=None, include_test: bool = False, test_years=None,
        out_dir: Path | None = None, log=print) -> dict:
    """폴드 전부에 대해 RL 신호 파일을 쓴다."""
    io._check_run_id_arg(run_id)
    sig = pd.DatetimeIndex(sorted(features["date"].unique()))
    kw = {} if test_years is None else {"test_years": test_years}
    folds = make_folds(sig, trading_days, h=h, actual_end_dates=actual_end_dates, **kw)
    out = []
    for fold in folds:
        chosen = params[fold.test_year]
        signals, meta = run_fold(features, labels, fold, trading_days, run_id=run_id, gate_params=chosen["gate"],
                                 ranker_params=chosen["ranker"], h=h, actual_end_dates=actual_end_dates,
                                 n_jobs=n_jobs, fixed=fixed, include_test=include_test, log=log)
        fold_run_id = f"{run_id}_y{fold.test_year}"
        if out_dir is None:
            meta["path"] = str(io.write_table("signals", signals, run_id=fold_run_id))
        else:
            path = Path(out_dir) / f"wf_{fold_run_id}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            signals.to_parquet(path, index=False)
            meta["path"] = str(path)
        log(f"폴드 {fold.test_year}: {meta['rows']}행 / {meta['signal_dates']}신호일 → {meta['path']}")
        out.append(meta)
    return {"run_id": run_id, "folds": out, "total_rows": sum(m["rows"] for m in out),
            "cross_fits": sum(len([s for s in m["segments"] if s["segment"] == "train"]) for m in out)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="DEC-08 RL 학습 구간 교차 적합 신호")
    ap.add_argument("--run-id", required=True, help="{YYYYMMDD}_{역할ID}_{짧은설명} (폴드마다 _y{연도}가 붙는다)")
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--params", required=True, help="본 walk-forward의 reports/{run_id}/metrics.json")
    ap.add_argument("--end-dates")
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--include-test", action="store_true")
    a = ap.parse_args(argv)
    features, labels = pd.read_parquet(a.features), pd.read_parquet(a.labels)
    index = io.read_table("index")
    days = pd.DatetimeIndex(sorted(index.loc[index["index_code"] == "KOSPI200", "date"].unique()))
    ends = label_end_dates(pd.read_parquet(a.end_dates), a.h) if a.end_dates else None
    params = fold_params(json.loads(Path(a.params).read_text(encoding="utf-8")))
    summary = run(features, labels, days, run_id=a.run_id, params=params, h=a.h, actual_end_dates=ends,
                  n_jobs=a.n_jobs, include_test=a.include_test)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
