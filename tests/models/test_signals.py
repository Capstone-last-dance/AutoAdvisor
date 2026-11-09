import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.calendar import signal_dates
from src.data.schemas import validate
from src.models import signals as signals_module
from src.models.signals import generate_signals, load_final, m3_step, save_final, train_final

DAYS = pd.bdate_range("2010-01-04", "2026-10-30")
SIG = pd.DatetimeIndex(signal_dates(DAYS))
GRID = {"num_leaves": [7], "learning_rate": [0.1], "min_child_samples": [20]}
FIXED = {"n_estimators_max": 40, "early_stopping_rounds": 5}
N_TK = 25


def _data(seed=0):
    """2011~2025년 주간 신호일 + 2026년 일부 신호일(라벨 없음). 라벨은 vol_60, mom_12_1에 의존한다."""
    rng = np.random.default_rng(seed)
    dates = SIG[(SIG.year >= 2011) & ((SIG.year <= 2025) | (SIG >= "2026-10-01"))]
    rows = []
    for d in dates:
        f = pd.DataFrame({"date": d, "ticker": [f"{i:06d}" for i in range(N_TK)]})
        for c in ["vol_60", "mom_12_1", "rev_1m", "noise"]:
            f[c] = rng.permutation(N_TK) / (N_TK - 1)
        f["vol_60_na"] = np.int8(0)
        rows.append(f)
    feat = pd.concat(rows, ignore_index=True)
    er = 0.6 * feat["mom_12_1"] + 0.4 * (1 - feat["rev_1m"]) + rng.normal(0, 0.15, len(feat))
    z = -3.0 + 3.0 * feat["vol_60"]
    lab = feat[["date", "ticker"]].assign(
        h=np.int8(20), er=er, y_rank=er.groupby(feat["date"]).rank(pct=True), y_crash=(rng.random(len(feat)) < 1 / (1 + np.exp(-z))).astype("int8"))
    return feat, lab[lab["date"] <= "2025-12-31"]


@pytest.fixture(scope="module")
def trained():
    feat, lab = _data()
    return feat, lab, train_final(feat, lab, SIG, DAYS, grid=GRID, fixed=FIXED)


def test_final_models_use_2011_2023_train_and_2024_2025_validation(trained):
    m = trained[2].meta
    assert m["train_dates"][0] == f"{SIG[SIG.year == 2011][0]:%Y-%m-%d}" and m["train_dates"][1] < "2024-01-01"
    assert m["val_dates"][0] >= "2024-01-01" and m["val_dates"][1] <= "2025-12-31"


def test_generate_signals_passes_schema_for_arbitrary_signal_dates(trained):
    feat, _, models = trained
    for d in ["2012-03-30", "2019-06-28", "2025-12-26", "2026-10-02", "2026-10-30"]:
        s = generate_signals(d, feat, models, "final_test")
        validate("signals", s)
        assert len(s) == N_TK and s["rank"].tolist() == list(range(1, N_TK + 1))
        assert s["crash_flag"].sum() == math.ceil(0.10 * N_TK) == 3
        assert s["crash_prob"].between(0, 1).all() and (s["model_version"] == "final_test").all()
        assert (s["date"] == pd.Timestamp(d)).all()


def test_only_that_dates_features_are_used(trained):
    feat, _, models = trained
    d = pd.Timestamp("2026-10-30")
    a = generate_signals(d, feat, models, "v")
    other = feat.copy()
    other.loc[other["date"] != d, ["vol_60", "mom_12_1", "rev_1m"]] = 0.123          # 다른 날의 값을 바꿔도
    pd.testing.assert_frame_equal(a, generate_signals(d, other, models, "v"))
    pd.testing.assert_frame_equal(a, generate_signals(d, feat[feat["date"] <= d], models, "v"))   # 이후 행을 지워도 같다


@pytest.mark.parametrize("folder", ["models", "보고서 모델"])   # 비 ASCII 경로에서도 저장·로드가 되어야 한다
def test_saved_models_reproduce_signals_exactly(trained, tmp_path, folder):
    feat, _, models = trained
    d = tmp_path / folder
    save_final(models, d)
    loaded = load_final(d)
    for d in ["2015-01-30", "2026-10-30"]:
        pd.testing.assert_frame_equal(generate_signals(d, feat, models, "v"), generate_signals(d, feat, loaded, "v"))
    assert loaded.meta == models.meta and loaded.h == 20


def test_holdout_period_rows_do_not_affect_training():
    feat, lab = _data()
    a = train_final(feat, lab, SIG, DAYS, grid=GRID, fixed=FIXED)
    junk = feat.copy()
    late = junk["date"] >= "2026-01-01"
    junk.loc[late, ["vol_60", "mom_12_1", "rev_1m", "noise"]] = 0.5                 # 2026년 피처를 바꿔도 학습은 같다
    b = train_final(junk, lab, SIG, DAYS, grid=GRID, fixed=FIXED)
    d = pd.Timestamp("2025-12-26")
    pd.testing.assert_frame_equal(generate_signals(d, feat, a, "v"), generate_signals(d, feat, b, "v"))


def test_missing_date_and_missing_feature_raise(trained):
    feat, _, models = trained
    with pytest.raises(ValueError, match="피처 행이 없다"):
        generate_signals("2026-10-31", feat, models, "v")
    with pytest.raises(ValueError, match="피처가 입력에 없다"):
        generate_signals("2026-10-30", feat.drop(columns=["noise"]), models, "v")


# ---------------------------------------------------------------------------
# run_weekly 3단계 (m3_step)
# ---------------------------------------------------------------------------

class _FakeCtx:
    """D의 StepContext 중 3단계가 쓰는 부분만 흉내 낸다 (run_weekly.py는 쓰지 않는다)."""

    def __init__(self, date, tables, stub_tables=()):
        self.date = date
        self.stub_tables = set(stub_tables)
        self._tables = tables
        self.written = {}
        self.log = logging.getLogger("test.m3_step")

    def read_table(self, name, date=None):
        return self._tables[name]

    def write_table(self, name, df, date=None):
        self.written[name] = (date, df)
        return f"{name}/{date}.parquet"


def _universe(date, tickers):
    return pd.DataFrame({"date": pd.Timestamp(date), "ticker": list(tickers),
                         "mcap_rank": range(1, len(tickers) + 1), "in_universe": True})


def _prices_to(date, n_tk=N_TK, seed=5):
    """date까지의 일별 시세(long). 베이스라인이 mom_12_1·vol_60을 계산할 만큼 길게 만든다."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp(date), periods=400)
    cols = [f"{i:06d}" for i in range(n_tk)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(idx), n_tk)), axis=0)), index=idx, columns=cols)
    long = close.stack().rename("close").reset_index()
    long.columns = ["date", "ticker", "close"]
    return close, long


def test_m3_step_before_ml_date_uses_baseline():
    t = "2026-11-20"   # positions는 이 날부터지만 신호는 아직 베이스라인이다 (6.8절)
    close, prices = _prices_to(t)
    ctx = _FakeCtx(t, {"prices": prices, "universe": _universe(t, close.columns)})
    m3_step(ctx)
    date, df = ctx.written["signals"]
    assert date == t
    validate("signals", df)
    assert (df["model_version"] == "baseline").all()
    assert sorted(df["ticker"]) == sorted(close.columns)


def test_m3_step_from_ml_date_uses_final_models(trained, tmp_path, monkeypatch):
    feat, _, models = trained
    t = "2026-11-27"
    x = feat[feat["date"] == feat["date"].max()].assign(date=pd.Timestamp(t))
    directory = tmp_path / "final_20261127"
    save_final(models, directory)
    monkeypatch.setattr(signals_module, "FINAL_MODEL_DIR", directory)
    ctx = _FakeCtx(t, {"features": x, "universe": _universe(t, x["ticker"])})
    m3_step(ctx)
    date, df = ctx.written["signals"]
    assert date == t
    validate("signals", df)
    assert (df["model_version"] == "final_20261127").all()   # "baseline"이 아니어야 D가 ML 기록으로 가른다
    pd.testing.assert_frame_equal(df, generate_signals(t, x, load_final(directory), "final_20261127"))


def test_m3_step_rejects_signals_that_differ_from_universe(trained, tmp_path, monkeypatch):
    feat, _, models = trained
    t = "2026-11-27"
    x = feat[feat["date"] == feat["date"].max()].assign(date=pd.Timestamp(t))
    save_final(models, tmp_path / "final_x")
    monkeypatch.setattr(signals_module, "FINAL_MODEL_DIR", tmp_path / "final_x")
    universe = pd.concat([_universe(t, x["ticker"]), _universe(t, ["999999"])], ignore_index=True)
    ctx = _FakeCtx(t, {"features": x, "universe": universe})
    with pytest.raises(ValueError, match="U_t와 다르다"):
        m3_step(ctx)
    assert "signals" not in ctx.written   # 틀린 파일을 쓰고 나서 멈추면 안 된다


def test_m3_step_writes_empty_stub_when_inputs_are_stubbed(monkeypatch):
    monkeypatch.setattr(signals_module, "FINAL_MODEL_DIR", Path("/없는경로/final"))   # 모델을 읽지 않고 끝나야 한다
    ctx = _FakeCtx("2026-11-27", {}, stub_tables={"features"})
    m3_step(ctx)
    date, df = ctx.written["signals"]
    assert date == "2026-11-27" and df.empty
    assert list(df.columns) == ["date", "ticker", "rank_score", "rank", "crash_prob", "crash_flag", "model_version"]


def test_m3_step_rejects_final_model_named_baseline(trained, tmp_path, monkeypatch):
    feat, _, models = trained
    directory = tmp_path / "baseline"
    save_final(models, directory)
    monkeypatch.setattr(signals_module, "FINAL_MODEL_DIR", directory)
    x = feat[feat["date"] == feat["date"].max()].assign(date=pd.Timestamp("2026-11-27"))
    ctx = _FakeCtx("2026-11-27", {"features": x, "universe": _universe("2026-11-27", x["ticker"])})
    with pytest.raises(ValueError, match="model_version"):
        m3_step(ctx)
