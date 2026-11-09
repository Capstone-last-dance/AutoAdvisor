"""A-08 최종 모델과 주간 신호 생성 함수 `generate_signals(date)` (4.7절).

최종 모델: 학습 2011~2023년, 검증 2024~2025년 신호일 (splits.final_split, purge 포함). M3a와 M3b를 각각 적합해 한 묶음(FinalModels)으로 저장한다.
홀드아웃(2026-01-01~)은 학습·검증·튜닝에 쓰지 않는다 (R-04). 신호 생성(추론)은 신호일 t의 피처 행만 쓴다 (R-03).
generate_signals는 모델 파일만으로 돌아간다. 매주 실행(D의 run_weekly)은 load_final로 읽은 모델에 t의 피처를 넘기면 된다.
model_version 값은 명세에 정의가 없어 호출자가 정한다 (baseline은 "baseline").
"""
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data.schemas import validate
from src.models import gate, ranker
from src.models.splits import final_split


@dataclass
class FinalModels:
    gate_booster: lgb.Booster
    gate_features: list
    calibrator: IsotonicRegression
    ranker_booster: lgb.Booster
    ranker_features: list
    h: int
    meta: dict
    gate_iteration: int | None = None      # 학습 직후에는 best_iteration, 파일에서 읽으면 None (저장 때 이미 잘랐다)
    ranker_iteration: int | None = None


def train_final(features: pd.DataFrame, labels: pd.DataFrame, signal_dates, trading_days, h: int = 20, **kw) -> FinalModels:
    """features·labels는 2025-12-31 이하 신호일만 쓰인다(그 이후 행이 있어도 final_split이 구간에 넣지 않는다).
    kw는 grid, fixed, seed, n_jobs를 gate·ranker 양쪽에 넘긴다."""
    fold = final_split(signal_dates, trading_days, h=h)
    g = gate.fit_fold(features, labels, fold, require_test=False, **kw)
    r = ranker.fit_fold(features, labels, fold, h=h, require_test=False, **kw)
    meta = {"h": h, "train_dates": [f"{fold.train.min():%Y-%m-%d}", f"{fold.train.max():%Y-%m-%d}"],
            "val_dates": [f"{fold.val.min():%Y-%m-%d}", f"{fold.val.max():%Y-%m-%d}"],
            "gate_params": g.params, "gate_val_auc": g.val_auc, "ranker_params": r.params, "ranker_val_mean_rank_ic": r.val_ic}
    return FinalModels(g.model.booster_, g.features, g.calibrator, r.model.booster_, r.features, h, meta,
                       int(g.model.best_iteration_), int(r.model.best_iteration_))


def save_final(models: FinalModels, directory: Path):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    models.gate_booster.save_model(str(d / "final_gate.txt"), num_iteration=models.gate_iteration)
    models.ranker_booster.save_model(str(d / "final_ranker.txt"), num_iteration=models.ranker_iteration)
    joblib.dump({"calibrator": models.calibrator, "gate_features": models.gate_features,
                 "ranker_features": models.ranker_features, "h": models.h}, d / "final_aux.joblib")
    (d / "final_meta.json").write_text(json.dumps(models.meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_final(directory: Path) -> FinalModels:
    d = Path(directory)
    aux = joblib.load(d / "final_aux.joblib")
    return FinalModels(lgb.Booster(model_file=str(d / "final_gate.txt")), aux["gate_features"], aux["calibrator"],
                       lgb.Booster(model_file=str(d / "final_ranker.txt")), aux["ranker_features"], aux["h"],
                       json.loads((d / "final_meta.json").read_text(encoding="utf-8")))


def generate_signals(date, features: pd.DataFrame, models: FinalModels, model_version: str) -> pd.DataFrame:
    """신호일 date의 U_t 종목 전체에 대해 signals(9.3절) 행을 만든다. features는 compute_feature_matrix 형식이고, date가 아닌 행은 쓰지 않는다."""
    t = pd.Timestamp(date)
    x = features[features["date"] == t].reset_index(drop=True)
    if x.empty:
        raise ValueError(f"{t:%Y-%m-%d} 신호일의 피처 행이 없다")
    need = set(models.gate_features) | set(models.ranker_features)
    if need - set(x.columns):
        raise ValueError(f"모델이 쓰는 피처가 입력에 없다: {sorted(need - set(x.columns))}")

    out = x[["date", "ticker"]].copy()
    raw = models.gate_booster.predict(x[models.gate_features], num_iteration=models.gate_iteration)
    out["raw_prob"] = raw
    out["crash_prob"] = models.calibrator.predict(raw).astype("float64")
    out["crash_flag"] = gate.flag_top(out, "crash_prob", "raw_prob")
    out["rank_score"] = models.ranker_booster.predict(x[models.ranker_features], num_iteration=models.ranker_iteration).astype("float64")
    out["rank"] = ranker.rank_in_universe(out, "rank_score")
    out["model_version"] = model_version
    out = out[["date", "ticker", "rank_score", "rank", "crash_prob", "crash_flag", "model_version"]]
    out = out.sort_values("rank").reset_index(drop=True)
    validate("signals", out)   # io.py(D-02)가 생기면 io.validate로 바꾼다 (R-18). 지금은 로컬 스키마 검증
    return out
