import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.models.gate import load_booster
from src.models.ranker import (RIDGE_ALPHAS, mean_rank_ic, rank_in_universe, rank_labels, save_fold, train_ranker)
from src.models.splits import Fold

GRID = {"num_leaves": [7, 15], "learning_rate": [0.1], "min_child_samples": [20]}   # 테스트용 축소 그리드
FIXED = {"n_estimators_max": 80, "early_stopping_rounds": 10}


def _data(seed=0, n_tk=40):
    """er = 0.6 mom_12_1 + 0.4 (1 - rev_1m) + 상호작용 + 잡음. h = 5 행은 무관한 잡음 라벨이다."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-01", "2016-12-31", freq="W-FRI")
    rows = []
    for d in dates:
        f = pd.DataFrame({"date": d, "ticker": [f"{i:06d}" for i in range(n_tk)]})
        for c in ["mom_12_1", "rev_1m", "noise_a", "noise_b"]:
            f[c] = rng.permutation(n_tk) / (n_tk - 1)
        f["mom_12_1_na"] = np.int8(0)
        rows.append(f)
    feat = pd.concat(rows, ignore_index=True)
    er = (0.6 * feat["mom_12_1"] + 0.4 * (1 - feat["rev_1m"]) + 0.5 * feat["mom_12_1"] * feat["noise_a"]
          + rng.normal(0, 0.15, len(feat)))
    base = feat[["date", "ticker"]]
    l20 = base.assign(h=np.int8(20), er=er, y_rank=er.groupby(feat["date"]).rank(pct=True))
    junk = rng.normal(size=len(feat))
    l5 = base.assign(h=np.int8(5), er=junk, y_rank=pd.Series(junk).groupby(feat["date"]).rank(pct=True))
    return feat, pd.concat([l20, l5], ignore_index=True)


def _fold(feat):
    d = pd.DatetimeIndex(sorted(feat["date"].unique()))
    return Fold(2016, d[d.year <= 2014], d[d.year == 2015], d[d.year == 2016])


def _run(feat, lab, h=20, **kw):
    return train_ranker(h, features=feat, labels=lab, folds=[_fold(feat)], grid=GRID, fixed=FIXED, **kw)[0]


@pytest.fixture(scope="module")
def run():
    feat, lab = _data()
    return feat, lab, _run(feat, lab)


def _test_ic(r, lab, col):
    m = r.predictions.merge(rank_labels(lab, r.h), on=["date", "ticker"])
    return mean_rank_ic(m, col)


def test_mean_rank_ic_matches_scipy_with_ties_and_skips_degenerate_days():
    d1, d2, d3 = pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-17")
    df = pd.DataFrame({"date": [d1] * 5 + [d2] * 5 + [d3] * 2,
                       "s": [1, 2, 2, 4, 5, 5, 4, 3, 1, 1, 1, 2],
                       "er": [0.1, 0.3, 0.2, 0.5, 0.4, 0.9, 0.1, 0.2, 0.3, 0.8, 0.1, 0.2]})
    exp = np.mean([spearmanr(g["s"], g["er"])[0] for _, g in df[df["date"] != d3].groupby("date")])   # d3는 종목 2개라 제외
    assert mean_rank_ic(df, "s") == pytest.approx(exp)
    with pytest.raises(ValueError, match="계산할 수 있는"):
        mean_rank_ic(df[df["date"] == d3], "s")


def test_rank_is_descending_within_date_with_ticker_tiebreak():
    df = pd.DataFrame({"date": pd.Timestamp("2020-01-03"), "ticker": ["000003", "000001", "000002", "000004"], "s": [0.9, 0.5, 0.5, 0.1]})
    r = rank_in_universe(df, "s")
    assert r.tolist() == [1, 2, 3, 4] and r.dtype == np.int32       # 000001이 000002보다 앞 (동점은 종목코드 순)


def test_predictions_schema_and_complete_rank_per_date(run):
    feat, _, r = run
    p = r.predictions
    assert len(p) == (feat["date"].dt.year == 2016).sum() and not p.duplicated(["date", "ticker"]).any()
    for _, g in p.groupby("date"):
        assert sorted(g["rank"]) == list(range(1, len(g) + 1))
        assert g.sort_values("rank")["rank_score"].is_monotonic_decreasing


def test_model_beats_b1_b2_baselines_on_test_ic(run):
    _, lab, r = run
    ic_model = _test_ic(r, lab, "rank_score")
    ic1, ic2, ic3 = (_test_ic(r, lab, c) for c in ("b1_score", "b2_score", "b3_score"))
    assert ic1 > 0.2 and ic2 > 0.1                                  # B2는 부호가 맞게 -rev_1m으로 들어갔다
    assert ic_model > max(ic1, ic2) and ic3 > ic1 - 0.1


def test_b1_b2_are_taken_from_the_named_features(run):
    feat, _, r = run
    te = feat[feat["date"].dt.year == 2016].reset_index(drop=True)
    np.testing.assert_array_equal(r.predictions["b1_score"], te["mom_12_1"])
    np.testing.assert_array_equal(r.predictions["b2_score"], -te["rev_1m"])


def test_selection_uses_validation_mean_rank_ic(run):
    r = run[2]
    assert r.val_ic == max(a for _, a, _ in r.grid_val_ic) and len(r.grid_val_ic) == 2
    assert r.ridge_alpha in RIDGE_ALPHAS and len(r.ridge_val_ic) == 4
    assert dict(r.ridge_val_ic)[r.ridge_alpha] == max(v for _, v in r.ridge_val_ic)
    assert r.model.best_iteration_ >= 1


def test_test_labels_do_not_influence_predictions(run):
    feat, lab, r = run
    bad = lab.copy()
    m = bad["date"].dt.year == 2016
    bad.loc[m, ["y_rank", "er"]] = 1 - bad.loc[m, ["y_rank", "er"]].to_numpy()
    pd.testing.assert_frame_equal(r.predictions, _run(feat, bad).predictions)


def test_h_selects_label_rows_and_missing_h_raises(run):
    feat, lab, r20 = run
    r5 = _run(feat, lab, h=5)                                         # h = 5 라벨은 잡음이라 IC가 0 근처다
    assert abs(_test_ic(r5, lab, "rank_score")) < 0.1 < _test_ic(r20, lab, "rank_score")
    with pytest.raises(ValueError, match="h = 1 행이 없다"):
        _run(feat, lab, h=1)


def test_exclude_features_removes_columns_and_validates_names(run):
    feat, lab, r = run
    r2 = _run(feat, lab, exclude_features=["mom_12_1", "mom_12_1_na"])
    assert "mom_12_1" not in r2.features and "mom_12_1" in r.features and len(r2.features) == len(r.features) - 2
    assert _test_ic(r2, lab, "rank_score") < _test_ic(r, lab, "rank_score")        # 핵심 피처를 빼면 나빠진다
    pd.testing.assert_series_equal(r2.predictions["b1_score"], r.predictions["b1_score"])   # B1은 모델 피처 제외와 무관
    with pytest.raises(ValueError, match="피처가 아닌"):
        _run(feat, lab, exclude_features=["nope"])
    with pytest.raises(ValueError, match="남은 피처가 없다"):
        _run(feat, lab, exclude_features=list(r.features))


def test_train_ranker_requires_data_arguments():
    with pytest.raises(ValueError, match="넘겨야"):
        train_ranker(20)


def test_deterministic(run):
    feat, lab, r = run
    pd.testing.assert_frame_equal(r.predictions, _run(feat, lab).predictions)


def test_holdout_dates_rejected():
    feat, lab = _data()
    d = pd.DatetimeIndex(sorted(feat["date"].unique()))
    late = Fold(2016, d[d.year <= 2014].append(pd.DatetimeIndex(["2026-01-02"])), d[d.year == 2015], d[d.year == 2016])
    with pytest.raises(ValueError, match="2025-12-31 이후"):
        train_ranker(20, features=feat, labels=lab, folds=[late], grid=GRID, fixed=FIXED)


@pytest.mark.parametrize("folder", ["models", "보고서 모델"])   # 비 ASCII 경로에서도 저장·로드가 되어야 한다
def test_saved_model_reproduces_predictions(run, tmp_path, folder):
    feat, _, r = run
    d = tmp_path / folder
    save_fold(r, d)
    booster = load_booster(d / "ranker_2016_h20.txt")
    aux = joblib.load(d / "ranker_2016_h20_aux.joblib")
    te = feat[feat["date"].dt.year == 2016].reset_index(drop=True)
    np.testing.assert_allclose(booster.predict(te[aux["features"]]), r.predictions["rank_score"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(aux["ridge"].predict(te[aux["features"]]), r.predictions["b3_score"].to_numpy(), atol=1e-12)


def _data_all_h(seed=0, n_tk=40):
    """A-09: h = 1, 5, 20 라벨이 모두 있는 데이터(`_data()`는 h = 1을 일부러 빼 뒀다, 다른 테스트가 그 부재로
    에러 경로를 확인한다)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-01", "2016-12-31", freq="W-FRI")
    rows = []
    for d in dates:
        f = pd.DataFrame({"date": d, "ticker": [f"{i:06d}" for i in range(n_tk)]})
        for c in ["mom_12_1", "rev_1m", "noise_a", "noise_b"]:
            f[c] = rng.permutation(n_tk) / (n_tk - 1)
        f["mom_12_1_na"] = np.int8(0)
        rows.append(f)
    feat = pd.concat(rows, ignore_index=True)
    base = feat[["date", "ticker"]]
    labs = []
    for h in (1, 5, 20):
        er = rng.normal(size=len(feat))
        labs.append(base.assign(h=np.int8(h), er=er, y_rank=pd.Series(er).groupby(feat["date"]).rank(pct=True)))
    return feat, pd.concat(labs, ignore_index=True)


@pytest.mark.parametrize("h", [1, 5, 20])
def test_a09_train_ranker_runs_for_h_1_5_20(h):
    """A-09 완료 기준: h = 1, 5, 20으로 실행 확인."""
    feat, lab = _data_all_h()
    r = _run(feat, lab, h=h)
    assert r.h == h and len(r.predictions) > 0 and r.predictions["date"].dt.year.eq(2016).all()


def test_a09_exclude_features_runs_for_h_1_5_20():
    """A-09 완료 기준: 피처 제외 옵션으로 실행 확인(기여도 실험, PM-12)."""
    feat, lab = _data_all_h()
    for h in (1, 5, 20):
        r = _run(feat, lab, h=h, exclude_features=["mom_12_1", "mom_12_1_na"])
        assert "mom_12_1" not in r.features and len(r.predictions) > 0
