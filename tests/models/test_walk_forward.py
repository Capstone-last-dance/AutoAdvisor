import json

import numpy as np
import pandas as pd
import pytest

from src.data.calendar import signal_dates
from src.models.walk_forward import calibration_table, fold_signals, run

DAYS = pd.bdate_range("2010-01-04", "2018-12-31")
SIG = pd.DatetimeIndex(signal_dates(DAYS))
N_TK = 20
GRID = {"num_leaves": [7], "learning_rate": [0.1], "min_child_samples": [20]}
FIXED = {"n_estimators_max": 30, "early_stopping_rounds": 5}


def _data(seed=0):
    rng = np.random.default_rng(seed)
    dates = SIG[SIG.year >= 2011]
    rows = []
    for d in dates:
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
def result(tmp_path_factory):
    feat, lab = _data()
    out = tmp_path_factory.mktemp("wf")
    summary = run(feat, lab, DAYS, run_id="20261109_A_test", test_years=(2016, 2017), grid=GRID, fixed=FIXED,
                  reports_root=out, write_signals=False, log=lambda s: None)
    return summary, out, feat, lab


def test_runs_every_test_year_and_keeps_signals_schema(result):
    summary, out, feat, _ = result
    assert [f["test_year"] for f in summary["folds"]] == [2016, 2017]
    signals = pd.read_parquet(out / "20261109_A_test" / "wf_diagnostics.parquet")
    assert set(signals["test_year"]) == {2016, 2017}
    assert summary["rows"] == len(feat[feat["date"].dt.year.isin([2016, 2017])])


def test_metrics_cover_both_completion_criteria(result):
    summary, out, _, _ = result
    meta = json.loads((out / "20261109_A_test" / "metrics.json").read_text(encoding="utf-8"))
    for fold in meta["folds"]:
        assert fold["gate"]["test_roc_auc"] is not None and fold["gate"]["test_lift_at_10pct"] is not None
        assert fold["ranker"]["test_mean_rank_ic"] is not None and fold["ranker"]["test_hac_t"] is not None
        for name in ("B1", "B2", "B3"):                       # 6.4절 베이스라인 비교 표
            assert f"{name}_mean_rank_ic" in fold["ranker"]
    assert meta["overall"]["gate_roc_auc"] is not None and meta["overall"]["ranker_hac_t"] is not None


def test_fold_models_and_calibration_figures_are_saved(result):
    _, out, _, _ = result
    root = out / "20261109_A_test"
    for year in (2016, 2017):
        assert (root / "models" / f"fold_{year}" / "gate").is_dir()
        assert (root / "models" / f"fold_{year}" / "ranker").is_dir()
        assert (root / f"calibration_{year}.csv").is_file()


def test_runs_reproduce_on_a_second_seed(result):
    _, out, _, _ = result
    feat2, lab2 = _data(seed=1)
    s = run(feat2, lab2, DAYS, run_id="20261109_A_test2", test_years=(2016,), grid=GRID, fixed=FIXED,
            reports_root=out, write_signals=False, log=lambda s: None)
    assert s["rows"] > 0


def test_fold_signals_rejects_mismatched_predictions():
    class _G:
        predictions = pd.DataFrame({"test_year": [2016], "date": [pd.Timestamp("2016-01-08")], "ticker": ["000001"],
                                    "raw_prob": [0.1], "crash_prob": [0.1], "crash_flag": [np.int8(0)],
                                    "base_prob": [0.1], "base_flag": [np.int8(0)]})

    class _R:
        predictions = pd.DataFrame({"test_year": [2016], "date": [pd.Timestamp("2016-01-08")], "ticker": ["999999"],
                                    "rank_score": [0.5], "rank": [np.int32(1)], "b1_score": [0.1], "b2_score": [0.1],
                                    "b3_score": [0.1]})

    with pytest.raises(ValueError, match="date, ticker"):
        fold_signals(_G(), _R(), "v")


def test_calibration_table_bins_predictions():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype("int8")
    table = calibration_table(p, y, bins=5)
    assert len(table) == 5 and table["n"].sum() == 500
    assert table["mean_pred"].is_monotonic_increasing
