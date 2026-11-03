"""A-06 M3a 급락 게이트 (6.3절). 폴드별로 LightGBM 이진분류를 학습하고 검증 구간으로 고르고 보정한다.

폴드 하나의 절차 (모두 부록 1 models 값):
 1. 학습 구간으로 그리드(num_leaves × learning_rate × min_child_samples) 조합마다 적합한다. early stopping은 검증 ROC-AUC 기준.
 2. 검증 ROC-AUC가 가장 높은 조합을 고른다. 테스트 구간은 다시 적합하지 않는다 (4.7절).
 3. 고른 모델의 검증 예측으로 isotonic regression을 적합해 테스트 예측에 적용한다 (crash_prob).
 4. 신호일마다 U_t에서 crash_prob 상위 ceil(0.10 × N_t)개를 crash_flag = 1로 한다.
    isotonic은 같은 값이 많이 생기므로 동점은 보정 전 확률, 그다음 종목코드 순으로 정해 결과를 고정한다.
베이스라인은 같은 폴드의 학습 구간에서 vol_60 순위 하나로 적합한 로지스틱 회귀다. 같은 방식으로 플래그를 만든다.

라벨은 C의 labels.parquet(date, ticker, h, ..., y_crash; y_crash는 h = 20 행)를 입력으로 받는다 (R-17).
AUC·lift 표와 보정 곡선은 M5 함수로 만드는 A-10의 일이라 여기서는 예측과 모델만 낸다.
"""
import json
import math
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 부록 1 models
SEED = 42
GRID = {"num_leaves": [15, 31, 63], "learning_rate": [0.02, 0.05], "min_child_samples": [50, 200]}
FIXED = {"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "n_estimators_max": 2000, "early_stopping_rounds": 100}
FLAG_FRACTION = 0.10
BASELINE_FEATURE = "vol_60"
LAST_TRAINABLE_DATE = pd.Timestamp("2025-12-31")   # 4.7절. 이보다 늦은 신호일은 학습·튜닝에 쓰지 않는다 (C의 guards.py가 정식 차단)
NON_FEATURES = {"date", "ticker", "y_crash", "y_rank", "h"}


@dataclass
class FoldResult:
    test_year: int | None
    params: dict
    val_auc: float
    grid_val_auc: list                   # [(params, 검증 AUC, best_iteration)]
    features: list
    model: lgb.LGBMClassifier
    calibrator: IsotonicRegression
    baseline: LogisticRegression
    predictions: pd.DataFrame = field(repr=False)


def crash_labels(labels: pd.DataFrame) -> pd.DataFrame:
    x = labels[labels["h"] == 20] if "h" in labels.columns else labels
    return x[["date", "ticker", "y_crash"]].dropna(subset=["y_crash"])


def feature_columns(features: pd.DataFrame) -> list:
    return [c for c in features.columns if c not in NON_FEATURES]


def labelled(features, labels, dates):
    """dates의 피처 행 중 라벨이 있는 행만 (거래정지 등으로 라벨이 없는 행은 학습에서 빠진다, 4.6절)."""
    return features[features["date"].isin(dates)].merge(labels, on=["date", "ticker"], how="inner")


def flag_top(df: pd.DataFrame, prob: str, tiebreak: str | None = None, fraction=FLAG_FRACTION) -> pd.Series:
    """날짜별 상위 ceil(fraction × N_t)개에 1. 동점은 tiebreak 값(내림차순), 종목코드(오름차순) 순."""
    keys = [prob] + ([tiebreak] if tiebreak else []) + ["ticker"]
    asc = [False] * (len(keys) - 1) + [True]
    flag = pd.Series(0, index=df.index, dtype="int8")
    for _, g in df.groupby("date", sort=False):
        k = math.ceil(fraction * len(g))
        flag.loc[g.sort_values(keys, ascending=asc).index[:k]] = 1
    return flag


def check_dates(*date_sets):
    late = [d for s in date_sets for d in pd.DatetimeIndex(s) if d > LAST_TRAINABLE_DATE]
    if late:
        raise ValueError(f"2025-12-31 이후 신호일은 학습·검증에 쓸 수 없다 (4.7절): {late[0]:%Y-%m-%d} 외 {len(late) - 1}건")


def _auc(y, p, what):
    if y.nunique() < 2:
        raise ValueError(f"{what} 구간의 라벨이 한 종류뿐이라 ROC-AUC를 계산할 수 없다")
    return float(roc_auc_score(y, p))


def fit_fold(features: pd.DataFrame, labels: pd.DataFrame, fold, grid=None, fixed=None, seed=SEED, n_jobs=1,
             require_test=True) -> FoldResult:
    """features: date, ticker, 피처 열 (6.2절 `_na` 열 포함). labels: labels.parquet 형식. fold: splits.Fold."""
    grid, fixed = grid or GRID, {**FIXED, **(fixed or {})}
    check_dates(fold.train, fold.val)
    cols = feature_columns(features)
    lab = crash_labels(labels)
    tr, va = labelled(features, lab, fold.train), labelled(features, lab, fold.val)
    te = features[features["date"].isin(fold.test)].copy()
    if not len(tr) or not len(va) or (require_test and not len(te)):
        raise ValueError("학습, 검증, 테스트 중 행이 없는 구간이 있다")

    results = []
    for nl, lr, mcs in product(grid["num_leaves"], grid["learning_rate"], grid["min_child_samples"]):
        params = {"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
        m = lgb.LGBMClassifier(
            objective="binary", metric="auc", n_estimators=fixed["n_estimators_max"], colsample_bytree=fixed["feature_fraction"],
            subsample=fixed["bagging_fraction"], subsample_freq=fixed["bagging_freq"], random_state=seed,
            n_jobs=n_jobs, deterministic=True, force_row_wise=True, verbose=-1, **params)
        m.fit(tr[cols], tr["y_crash"], eval_set=[(va[cols], va["y_crash"])],
              callbacks=[lgb.early_stopping(fixed["early_stopping_rounds"], first_metric_only=True, verbose=False)])
        p = m.predict_proba(va[cols], num_iteration=m.best_iteration_)[:, 1]
        results.append((params, _auc(va["y_crash"], p, "검증"), int(m.best_iteration_ or fixed["n_estimators_max"]), m))
    best = max(results, key=lambda r: r[1])      # 동률이면 먼저 나온 조합
    params, val_auc, _, model = best

    raw_val = model.predict_proba(va[cols], num_iteration=model.best_iteration_)[:, 1]
    cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw_val, va["y_crash"])
    base = LogisticRegression().fit(tr[[BASELINE_FEATURE]], tr["y_crash"])
    if len(te):   # 최종 모델(테스트 구간 없음)은 모델만 만든다
        out = te[["date", "ticker"]].copy()
        out["raw_prob"] = model.predict_proba(te[cols], num_iteration=model.best_iteration_)[:, 1]
        out["crash_prob"] = cal.predict(out["raw_prob"]).astype("float64")
        out["crash_flag"] = flag_top(out, "crash_prob", "raw_prob")
        out["base_prob"] = base.predict_proba(te[[BASELINE_FEATURE]])[:, 1]
        out["base_flag"] = flag_top(out, "base_prob")
    else:
        out = pd.DataFrame({c: [] for c in ["date", "ticker", "raw_prob", "crash_prob", "crash_flag", "base_prob", "base_flag"]})
    out.insert(0, "test_year", fold.test_year)
    return FoldResult(fold.test_year, params, val_auc, [(r[0], r[1], r[2]) for r in results], cols, model, cal, base,
                      out.reset_index(drop=True))


def walk_forward_gate(features, labels, folds, **kw) -> list:
    return [fit_fold(features, labels, f, **kw) for f in folds]


def save_fold(res: FoldResult, directory: Path):
    """M12(SHAP)가 읽을 모델 파일. LightGBM은 텍스트, 보정기와 베이스라인은 joblib."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    tag = res.test_year if res.test_year is not None else "final"
    res.model.booster_.save_model(str(d / f"gate_{tag}.txt"), num_iteration=res.model.best_iteration_)
    joblib.dump({"calibrator": res.calibrator, "baseline": res.baseline, "features": res.features}, d / f"gate_{tag}_aux.joblib")
    (d / f"gate_{tag}_meta.json").write_text(json.dumps(
        {"params": res.params, "val_auc": res.val_auc, "best_iteration": int(res.model.best_iteration_),
         "grid": [{"params": p, "val_auc": a, "best_iteration": i} for p, a, i in res.grid_val_auc]},
        ensure_ascii=False, indent=2), encoding="utf-8")
