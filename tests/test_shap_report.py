"""C-06 시험 연도 표본과 SHAP 출력 계약."""

import numpy as np
import pandas as pd
import pytest
import sys
from types import SimpleNamespace

from src.analysis import shap_report
from src.analysis.shap_report import _values, run, sample_test_year


def test_sample_is_deterministic_and_excludes_other_years():
    dates = pd.to_datetime(["2025-01-03"] * 5_100 + ["2026-01-02"] * 10)
    rows = pd.DataFrame({"date": dates, "ticker": [f"{i:06d}" for i in range(len(dates))]})
    a = sample_test_year(rows, 2025)
    b = sample_test_year(rows.sample(frac=1, random_state=9), 2025)
    assert len(a) == 5_000 and a[["date", "ticker"]].equals(b[["date", "ticker"]])
    assert (a.date.dt.year == 2025).all() and a.ticker.is_unique
    with pytest.raises(ValueError, match="중복"):
        sample_test_year(pd.concat([rows.iloc[:1], rows.iloc[:1]]), 2025)


def test_gate_uses_positive_class_shap_values():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    class Explainer:
        def shap_values(self, _):
            return np.stack([np.zeros((2, 2)), np.full((2, 2), 7.0)], axis=-1)

    np.testing.assert_array_equal(_values(Explainer(), frame, binary=True), np.full((2, 2), 7.0))
    with pytest.raises(ValueError, match="SHAP 배열"):
        _values(Explainer(), frame, binary=False)


def test_missing_fold_model_fails_before_creating_report(tmp_path):
    features = tmp_path / "features.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2016-01-08"]), "ticker": ["000001"]}).to_parquet(features)
    output = tmp_path / "report"
    with pytest.raises(FileNotFoundError, match="폴드 모델"):
        run(features, tmp_path / "models", output, years=[2016])
    assert not output.exists()
    with pytest.raises(ValueError, match="2016~2025"):
        run(features, tmp_path / "models", output, years=[2026])


def test_run_writes_two_model_reports_without_v7_artifacts(tmp_path, monkeypatch):
    features = tmp_path / "features.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2016-01-08", "2016-01-15", "2017-01-06", "2017-01-13"]),
                  "ticker": ["000001"] * 4, "a": [1.0, 2.0, 3.0, 4.0],
                  "b": [2.0, 1.0, 1.0, 2.0], "c": [1.0, 1.0, 2.0, 2.0]}).to_parquet(features)
    models = tmp_path / "models"
    for year in (2016, 2017):
        for kind in ("gate", "ranker"):
            path = shap_report._model_path(models, year, kind, 20)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fake model", encoding="utf-8")

    class Booster:
        def feature_name(self):
            return ["a", "b", "c"]

    class Explainer:
        def shap_values(self, frame):
            return frame.to_numpy(dtype=float)

    monkeypatch.setattr(shap_report, "load_booster", lambda _: Booster())
    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(TreeExplainer=lambda _: Explainer()))
    monkeypatch.setattr(shap_report, "_write_dependence", lambda frame, values, feature, path: path.write_bytes(b"figure"))
    out = tmp_path / "report"
    record = run(features, models, out, years=[2016, 2017])
    assert record["sample_rows"] == {"2016": 2, "2017": 2}
    assert len(pd.read_csv(out / "top10_by_year.csv")) == 12
    assert len(pd.read_csv(out / "rank_stability.csv")) == 2
    assert len(list((out / "figures").glob("*.png"))) == 6
    assert (out / "config.yaml").is_file() and (out / "metrics.json").is_file()
