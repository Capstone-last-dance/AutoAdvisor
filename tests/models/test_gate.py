import math

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.models.gate import crash_labels, feature_columns, fit_fold, flag_top, save_fold, walk_forward_gate
from src.models.splits import Fold


def test_ss_allowed_excluded_from_v1_model_input():
    """구성종목 원천 없음(이슈 #5), PM 결정(2026-11-04): ss_allowed는 v1 모델 입력에서 뺀다. ss_regime은 남긴다."""
    df = pd.DataFrame({"date": [0], "ticker": ["a"], "mom_12_1": [0.5], "ss_allowed": [np.nan], "ss_regime": [1]})
    cols = feature_columns(df)
    assert "ss_allowed" not in cols and "ss_regime" in cols and "mom_12_1" in cols

GRID = {"num_leaves": [7, 15], "learning_rate": [0.1], "min_child_samples": [20]}   # 테스트용 축소 그리드
FIXED = {"n_estimators_max": 80, "early_stopping_rounds": 10}


def _data(seed=0, n_tk=40):
    """2012~2016년 주간 신호일. y_crash는 vol_60과 mom_12_1에 의존한다 (모델이 베이스라인보다 낫도록)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-01", "2016-12-31", freq="W-FRI")
    rows = []
    for d in dates:
        f = pd.DataFrame({"date": d, "ticker": [f"{i:06d}" for i in range(n_tk)]})
        for c in ["vol_60", "mom_12_1", "noise_a", "noise_b"]:
            f[c] = rng.permutation(n_tk) / (n_tk - 1)
        f["vol_60_na"] = np.int8(0)
        rows.append(f)
    feat = pd.concat(rows, ignore_index=True)
    z = -3.0 + 3.0 * feat["vol_60"] + 3.0 * (feat["mom_12_1"] < 0.25) * feat["vol_60"]
    y = (rng.random(len(feat)) < 1 / (1 + np.exp(-z))).astype("int8")
    lab = feat[["date", "ticker"]].assign(h=np.int8(20), y_crash=y)
    junk = feat[["date", "ticker"]].assign(h=np.int8(5), y_crash=rng.integers(0, 2, len(feat)).astype("int8"))
    return feat, pd.concat([lab, junk], ignore_index=True)


def _fold(feat):
    d = pd.DatetimeIndex(sorted(feat["date"].unique()))
    return Fold(2016, d[d.year <= 2014], d[d.year == 2015], d[d.year == 2016])


@pytest.fixture(scope="module")
def run():
    feat, lab = _data()
    return feat, lab, fit_fold(feat, lab, _fold(feat), grid=GRID, fixed=FIXED)


def test_crash_labels_use_h20_rows_only():
    feat, lab = _data()
    assert set(crash_labels(lab)["date"]) == set(feat["date"]) and len(crash_labels(lab)) == len(feat)
    assert crash_labels(lab.drop(columns="h")).shape[0] == len(lab)       # h 열이 없으면 그대로 쓴다


def test_predictions_cover_test_dates_with_valid_probabilities(run):
    feat, _, r = run
    p = r.predictions
    te = feat[feat["date"].dt.year == 2016]
    assert len(p) == len(te) and not p.duplicated(["date", "ticker"]).any()
    assert p["crash_prob"].between(0, 1).all() and p["crash_flag"].dtype == np.int8
    assert (p["test_year"] == 2016).all()


def test_flag_count_is_ceil_10_percent_per_date_even_with_isotonic_ties(run):
    p = run[2].predictions
    assert p.groupby("date")["crash_prob"].nunique().max() < 40                  # 보정으로 동점이 실제로 생겼다
    for _, g in p.groupby("date"):
        assert g["crash_flag"].sum() == math.ceil(0.10 * len(g)) == 4
        assert g["base_flag"].sum() == 4


def test_flagged_are_the_highest_probabilities_with_raw_prob_tiebreak(run):
    for _, g in run[2].predictions.groupby("date"):
        top = g.sort_values(["crash_prob", "raw_prob", "ticker"], ascending=[False, False, True]).head(4)
        assert set(top["ticker"]) == set(g.loc[g["crash_flag"] == 1, "ticker"])


def test_model_beats_single_variable_baseline_on_test(run):
    feat, lab, r = run
    m = r.predictions.merge(crash_labels(lab), on=["date", "ticker"])
    auc_model, auc_base = roc_auc_score(m["y_crash"], m["crash_prob"]), roc_auc_score(m["y_crash"], m["base_prob"])
    assert auc_base > 0.6 and auc_model > auc_base


def test_selected_params_have_max_validation_auc(run):
    r = run[2]
    assert r.val_auc == max(a for _, a, _ in r.grid_val_auc) and len(r.grid_val_auc) == 2
    assert r.params in [p for p, _, _ in r.grid_val_auc]


def test_calibrator_is_fit_on_validation_predictions_only(run):
    feat, lab, r = run
    va = feat[feat["date"].dt.year == 2015]
    raw_val = r.model.predict_proba(va[r.features], num_iteration=r.model.best_iteration_)[:, 1]
    assert set(np.round(r.calibrator.X_thresholds_, 12)) <= set(np.round(raw_val, 12))


def test_test_labels_do_not_influence_predictions(run):
    feat, lab, r = run
    bad = lab.copy()
    bad.loc[bad["date"].dt.year == 2016, "y_crash"] = 1 - bad.loc[bad["date"].dt.year == 2016, "y_crash"]
    r2 = fit_fold(feat, bad, _fold(feat), grid=GRID, fixed=FIXED)
    pd.testing.assert_frame_equal(r.predictions, r2.predictions)


def test_deterministic_and_rows_without_labels_are_dropped_from_training(run):
    feat, lab, r = run
    again = fit_fold(feat, lab, _fold(feat), grid=GRID, fixed=FIXED)
    pd.testing.assert_frame_equal(r.predictions, again.predictions)
    fewer = lab.drop(lab[(lab["h"] == 20) & (lab["date"].dt.year == 2013)].index)   # 한 해 라벨이 전부 없어도 동작한다
    assert len(fit_fold(feat, fewer, _fold(feat), grid=GRID, fixed=FIXED).predictions) == len(r.predictions)


def test_holdout_dates_are_rejected_for_training_and_validation():
    feat, lab = _data()
    d = pd.DatetimeIndex(sorted(feat["date"].unique()))
    late = Fold(2016, d[d.year <= 2014].append(pd.DatetimeIndex(["2026-01-02"])), d[d.year == 2015], d[d.year == 2016])
    with pytest.raises(ValueError, match="2025-12-31 이후"):
        fit_fold(feat, lab, late, grid=GRID, fixed=FIXED)


def test_single_class_validation_raises():
    feat, lab = _data()
    lab.loc[(lab["h"] == 20) & (lab["date"].dt.year == 2015), "y_crash"] = 0
    with pytest.raises(ValueError, match="한 종류뿐"):
        fit_fold(feat, lab, _fold(feat), grid=GRID, fixed=FIXED)


def test_walk_forward_returns_one_result_per_fold():
    feat, lab = _data()
    d = pd.DatetimeIndex(sorted(feat["date"].unique()))
    f1 = Fold(2015, d[d.year <= 2013], d[d.year == 2014], d[d.year == 2015])
    res = walk_forward_gate(feat, lab, [f1, _fold(feat)], grid=GRID, fixed=FIXED)
    assert [r.test_year for r in res] == [2015, 2016]


def test_flag_top_small_universe():
    df = pd.DataFrame({"date": pd.Timestamp("2020-01-03"), "ticker": ["000001", "000002", "000003"], "p": [0.5, 0.5, 0.1]})
    assert flag_top(df, "p").tolist() == [1, 0, 0]          # ceil(0.3) = 1개, 동점은 종목코드 순


def test_saved_model_reproduces_predictions(run, tmp_path):
    feat, _, r = run
    save_fold(r, tmp_path)
    booster = lgb.Booster(model_file=str(tmp_path / "gate_2016.txt"))
    aux = joblib.load(tmp_path / "gate_2016_aux.joblib")
    te = feat[feat["date"].dt.year == 2016].reset_index(drop=True)
    raw = booster.predict(te[aux["features"]])
    np.testing.assert_allclose(aux["calibrator"].predict(raw), r.predictions["crash_prob"].to_numpy(), atol=1e-12)
    assert (tmp_path / "gate_2016_meta.json").exists()
