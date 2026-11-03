"""A-07 M3b 진입 순위 모델 (6.4절)과 A-09 `train_ranker`.

폴드 하나의 절차 (부록 1 models.ranker):
 1. 학습 구간으로 LightGBM 회귀(목표 y_rank)를 그리드 12조합마다 적합한다. early stopping은 검증 l2 기준이다.
 2. 검증 구간 평균 rank IC가 가장 높은 조합을 고른다. 테스트 구간은 다시 적합하지 않는다 (4.7절).
 3. 베이스라인: B1 rank_score = mom_12_1, B2 = -rev_1m, B3 = 같은 피처의 릿지 회귀(alpha ∈ {0.1, 1, 10, 100}, 검증 평균 rank IC로 선택).
    릿지에는 피처 표준화를 넣지 않았다(명세에 없음, R-08). 순위 피처는 이미 0~1 범위다.
출력은 신호일별 rank_score(예측값)와 rank(U_t 안의 내림차순 순위, 1이 최고, 동점은 종목코드 순).

rank IC = 신호일별 Spearman(rank_score, ER(t,20)) 의 신호일 평균 (6.6절). 여기 구현은 모델 선택용이다. 보고용 표(연도별 IC, Newey-West t)는
M5 함수(C 소유)로 계산한다 (A-10). 라벨(labels.parquet: date, ticker, h, ret, er, y_rank, y_crash)은 C의 산출물을 입력으로 받는다 (R-17).
"""
import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.gate import FIXED, GRID, SEED, check_dates, feature_columns, labelled

RIDGE_ALPHAS = [0.1, 1, 10, 100]   # 부록 1 models.ranker.baselines.B3
IC_MIN_NAMES = 3                   # 한 신호일에 이보다 적은 종목이면 그날의 IC는 계산하지 않는다


@dataclass
class RankerResult:
    test_year: int | None
    h: int
    params: dict
    val_ic: float
    grid_val_ic: list                    # [(params, 검증 평균 rank IC, best_iteration)]
    ridge_alpha: float
    ridge_val_ic: list                   # [(alpha, 검증 평균 rank IC)]
    features: list
    model: lgb.LGBMRegressor
    ridge: Ridge
    predictions: pd.DataFrame = field(repr=False)


def rank_labels(labels: pd.DataFrame, h: int) -> pd.DataFrame:
    x = labels[labels["h"] == h]
    if x.empty:
        raise ValueError(f"labels에 h = {h} 행이 없다")
    return x[["date", "ticker", "y_rank", "er"]].dropna(subset=["y_rank", "er"])


def mean_rank_ic(df: pd.DataFrame, score: str, er: str = "er") -> float:
    """신호일별 Spearman(score, er)의 평균. 종목이 3개 미만이거나 한쪽이 상수인 날은 뺀다."""
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < IC_MIN_NAMES:
            continue
        c = g[score].rank().corr(g[er].rank())
        if pd.notna(c):
            ics.append(c)
    if not ics:
        raise ValueError("rank IC를 계산할 수 있는 신호일이 없다")
    return float(np.mean(ics))


def rank_in_universe(df: pd.DataFrame, score: str) -> pd.Series:
    """날짜별 내림차순 순위(1이 최고). 동점은 종목코드 오름차순."""
    out = pd.Series(0, index=df.index, dtype="int32")
    for _, g in df.groupby("date", sort=False):
        order = g.sort_values([score, "ticker"], ascending=[False, True]).index
        out.loc[order] = np.arange(1, len(g) + 1, dtype="int32")
    return out


def fit_fold(features: pd.DataFrame, labels: pd.DataFrame, fold, h: int = 20, exclude_features=None, grid=None, fixed=None,
             seed=SEED, n_jobs=1, require_test=True) -> RankerResult:
    grid, fixed = grid or GRID, {**FIXED, **(fixed or {})}
    check_dates(fold.train, fold.val)
    all_cols = feature_columns(features)
    drop = set(exclude_features or [])
    unknown = drop - set(all_cols)
    if unknown:
        raise ValueError(f"exclude_features에 피처가 아닌 이름이 있다: {sorted(unknown)}")
    cols = [c for c in all_cols if c not in drop]
    if not cols:
        raise ValueError("남은 피처가 없다")
    lab = rank_labels(labels, h)
    tr, va = labelled(features, lab, fold.train), labelled(features, lab, fold.val)
    te = features[features["date"].isin(fold.test)].copy()
    if not len(tr) or not len(va) or (require_test and not len(te)):
        raise ValueError("학습, 검증, 테스트 중 행이 없는 구간이 있다")

    results = []
    for nl, lr, mcs in product(grid["num_leaves"], grid["learning_rate"], grid["min_child_samples"]):
        params = {"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
        m = lgb.LGBMRegressor(
            objective="regression", metric="l2", n_estimators=fixed["n_estimators_max"], colsample_bytree=fixed["feature_fraction"],
            subsample=fixed["bagging_fraction"], subsample_freq=fixed["bagging_freq"], random_state=seed,
            n_jobs=n_jobs, deterministic=True, force_row_wise=True, verbose=-1, **params)
        m.fit(tr[cols], tr["y_rank"], eval_set=[(va[cols], va["y_rank"])],
              callbacks=[lgb.early_stopping(fixed["early_stopping_rounds"], first_metric_only=True, verbose=False)])
        ic = mean_rank_ic(va.assign(s=m.predict(va[cols], num_iteration=m.best_iteration_)), "s")
        results.append((params, ic, int(m.best_iteration_ or fixed["n_estimators_max"]), m))
    params, val_ic, _, model = max(results, key=lambda r: r[1])

    ridge_scores = []
    for a in RIDGE_ALPHAS:
        r = Ridge(alpha=a).fit(tr[cols], tr["y_rank"])
        ridge_scores.append((a, mean_rank_ic(va.assign(s=r.predict(va[cols])), "s"), r))
    alpha, _, ridge = max(ridge_scores, key=lambda r: r[1])

    if len(te):   # 최종 모델(테스트 구간 없음)은 모델만 만든다
        out = te[["date", "ticker"]].copy()
        out["rank_score"] = model.predict(te[cols], num_iteration=model.best_iteration_).astype("float64")
        out["rank"] = rank_in_universe(out, "rank_score")
        out["b1_score"] = te["mom_12_1"].to_numpy(dtype="float64")
        out["b2_score"] = -te["rev_1m"].to_numpy(dtype="float64")
        out["b3_score"] = ridge.predict(te[cols])
    else:
        out = pd.DataFrame({c: [] for c in ["date", "ticker", "rank_score", "rank", "b1_score", "b2_score", "b3_score"]})
    out.insert(0, "test_year", fold.test_year)
    return RankerResult(fold.test_year, h, params, val_ic, [(r[0], r[1], r[2]) for r in results], alpha,
                        [(a, ic) for a, ic, _ in ridge_scores], cols, model, ridge, out.reset_index(drop=True))


def train_ranker(h: int, exclude_features=None, *, features: pd.DataFrame | None = None, labels: pd.DataFrame | None = None,
                 folds=None, **kw) -> list:
    """공개 함수 (6.4절, A-09). h는 예측 기간, exclude_features는 기여도 실험에서 뺄 피처 목록. 폴드마다 RankerResult를 돌려준다.

    features·labels·folds는 호출자가 넘긴다. src/common/io.py(D-02)가 생기면 그 함수로 읽도록 바꾼다 (R-18).
    G6(C-08)의 h = 1, 5 실험은 그 h의 신호일 주기에 맞춘 features·folds(splits.make_folds)를 넘기면 된다."""
    if features is None or labels is None or folds is None:
        raise ValueError("features, labels, folds를 넘겨야 한다 (io.py가 생기기 전까지 train_ranker는 파일을 읽지 않는다)")
    return [fit_fold(features, labels, f, h=h, exclude_features=exclude_features, **kw) for f in folds]


def save_fold(res: RankerResult, directory: Path):
    """M12(SHAP)가 읽을 모델 파일."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{res.test_year if res.test_year is not None else 'final'}_h{res.h}"
    res.model.booster_.save_model(str(d / f"ranker_{tag}.txt"), num_iteration=res.model.best_iteration_)
    joblib.dump({"ridge": res.ridge, "features": res.features}, d / f"ranker_{tag}_aux.joblib")
    (d / f"ranker_{tag}_meta.json").write_text(json.dumps(
        {"h": res.h, "params": res.params, "val_mean_rank_ic": res.val_ic, "best_iteration": int(res.model.best_iteration_),
         "ridge_alpha": res.ridge_alpha, "ridge_val_ic": res.ridge_val_ic,
         "grid": [{"params": p, "val_mean_rank_ic": a, "best_iteration": i} for p, a, i in res.grid_val_ic]},
        ensure_ascii=False, indent=2), encoding="utf-8")
