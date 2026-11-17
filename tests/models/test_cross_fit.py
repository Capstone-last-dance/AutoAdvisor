import numpy as np
import pandas as pd
import pytest

from src.data.calendar import signal_dates
from src.data.schemas import validate
from src.models.cross_fit import _check_complete, run, run_fold, single_point
from src.models.splits import make_folds

DAYS = pd.bdate_range("2010-01-04", "2018-12-31")
SIG = pd.DatetimeIndex(signal_dates(DAYS))
N_TK = 12
PARAMS = {"num_leaves": 7, "learning_rate": 0.1, "min_child_samples": 20}
FIXED = {"n_estimators_max": 25, "early_stopping_rounds": 5}


def _data(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in SIG[SIG.year >= 2011]:
        f = pd.DataFrame({"date": d, "ticker": [f"{i:06d}" for i in range(N_TK)]})
        for c in ["vol_60", "mom_12_1", "rev_1m"]:
            f[c] = rng.permutation(N_TK) / (N_TK - 1)
        f["vol_60_na"] = np.int8(0)
        rows.append(f)
    feat = pd.concat(rows, ignore_index=True)
    er = 0.6 * feat["mom_12_1"] + rng.normal(0, 0.2, len(feat))
    z = -3.0 + 3.0 * feat["vol_60"]
    lab = feat[["date", "ticker"]].assign(
        h=np.int8(20), er=er, y_rank=er.groupby(feat["date"]).rank(pct=True),
        y_crash=(rng.random(len(feat)) < 1 / (1 + np.exp(-z))).astype("int8"))
    return feat, lab


@pytest.fixture(scope="module")
def fold_result():
    feat, lab = _data()
    fold = make_folds(SIG, DAYS, h=20, test_years=(2016,))[0]
    signals, meta = run_fold(feat, lab, fold, DAYS, run_id="20261110_A_rltest", gate_params=PARAMS,
                             ranker_params=PARAMS, fixed=FIXED, n_jobs=2, log=lambda s: None)
    return feat, fold, signals, meta


def test_single_point_grid_has_one_value_per_key():
    assert single_point({"num_leaves": 31, "learning_rate": 0.05}) == {"num_leaves": [31], "learning_rate": [0.05]}


def test_file_covers_every_train_and_validation_signal_date(fold_result):
    feat, fold, signals, _ = fold_result
    want = set(fold.train) | set(fold.val)
    assert set(signals["date"]) == want
    assert set(signals["date"]).isdisjoint(set(fold.test))        # 시험 연도는 기본값에서 빠진다
    expected = feat[feat["date"].isin(want)][["date", "ticker"]]
    assert len(signals) == len(expected)                          # 신호일마다 U_t 전 종목
    validate("signals", signals)


def test_train_years_come_from_cross_fits_and_validation_from_the_fold_model(fold_result):
    _, fold, signals, meta = fold_result
    train_years = sorted({d.year for d in fold.train})
    assert [s["year"] for s in meta["segments"] if s["segment"] == "train"] == train_years
    cross = signals[signals["model_version"].str.endswith("_cross")]
    fold_rows = signals[signals["model_version"].str.endswith("_fold")]
    assert sorted(set(cross["date"].dt.year)) == train_years
    assert sorted(set(fold_rows["date"].dt.year)) == sorted({d.year for d in fold.val})


def test_cross_fit_never_trains_on_the_year_it_predicts(fold_result):
    _, fold, _, meta = fold_result
    for segment in [s for s in meta["segments"] if s["segment"] == "train"]:
        year = segment["year"]
        in_year = sum(1 for d in fold.train if d.year == year)
        assert segment["train_signal_dates"] <= len(fold.train) - in_year   # 그 해를 빼고, 경계도 purge한다
        assert segment["rows"] > 0


def test_rank_is_contiguous_per_signal_date(fold_result):
    _, _, signals, _ = fold_result
    for _, g in signals.groupby("date"):
        assert sorted(g["rank"]) == list(range(1, len(g) + 1))
        assert g["crash_flag"].isin([0, 1]).all()


def test_missing_universe_row_or_signal_date_is_rejected(fold_result):
    feat, fold, signals, _ = fold_result
    drop_one = signals[~((signals["date"] == fold.train[0]) & (signals["ticker"] == "000000"))]
    with pytest.raises(ValueError, match="U_t 종목"):
        _check_complete(drop_one, feat, fold, include_test=False)
    drop_date = signals[signals["date"] != fold.train[0]]
    with pytest.raises(ValueError, match="신호일"):
        _check_complete(drop_date, feat, fold, include_test=False)
    _check_complete(signals, feat, fold, include_test=False)     # 원본은 통과한다


def test_run_writes_one_file_per_fold(tmp_path):
    feat, lab = _data(seed=1)
    params = {2016: {"gate": PARAMS, "ranker": PARAMS}}
    summary = run(feat, lab, DAYS, run_id="20261110_A_rltest", params=params, test_years=(2016,),
                  fixed=FIXED, n_jobs=2, out_dir=tmp_path, log=lambda s: None)
    assert summary["cross_fits"] == 3                       # 폴드 2016의 학습 연도 2011~2013
    path = tmp_path / "wf_20261110_A_rltest_y2016.parquet"
    assert path.is_file()
    validate("signals", pd.read_parquet(path))
