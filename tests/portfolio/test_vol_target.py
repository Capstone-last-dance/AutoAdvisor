"""작업 B-02: risk_layer.vol_target_annual(Q-12)과 compute_regime_mc (6.9절)."""
from __future__ import annotations

import importlib.util
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.portfolio import vol_target_report
from src.portfolio.params import RiskLayerParams, vol_target_years
from src.portfolio.risk_layer import (
    InsufficientHistory,
    compute_regime_mc,
    compute_vol_target_annual,
    market_state,
    market_state_table,
)
from tests.portfolio.fakes import bm_crash_rebound_path

HAS_IO = importlib.util.find_spec("src.common.io") is not None


def _bm(start: str = "2009-01-02", end: str = "2017-12-29", seed: int = 0) -> pd.Series:
    """코스피 200 대용 무작위 경로. 거래일은 영업일로 근사한다."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    return pd.Series(250.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(days)))), index=days)


def _reference(bm: pd.Series, first: int, last: int, p: RiskLayerParams) -> float:
    """6.9절 정의를 그대로 옮긴 느린 구현: first~last년 매 거래일 sigma_t의 산술평균."""
    values = bm.to_numpy()
    sigmas = []
    for i, d in enumerate(bm.index):
        if first <= d.year <= last:
            r = [math.log(values[s] / values[s - 1]) for s in range(i - p.vol_lookback_days + 1, i + 1)]
            sigmas.append(statistics.stdev(r) * math.sqrt(p.annualization_daily))
    return statistics.fmean(sigmas)


@pytest.fixture
def years(cfg) -> tuple[int, int]:
    return vol_target_years(cfg)


def test_vol_target_matches_definition(risk_params, years):
    bm = _bm()
    expected = _reference(bm, *years, risk_params)
    assert compute_vol_target_annual(bm, risk_params, years) == pytest.approx(expected, rel=1e-12)


def test_vol_target_does_not_use_2016_or_later(risk_params, years):
    """B-02 완료 기준: 2016년 이후 종가를 어떻게 바꿔도 값이 같다."""
    bm = _bm()
    later = bm.index > pd.Timestamp("2015-12-31")
    changed = bm.copy()
    changed[later] = changed[later] * np.linspace(0.3, 3.0, later.sum())
    assert compute_vol_target_annual(changed, risk_params, years) == compute_vol_target_annual(bm, risk_params, years)


def test_vol_target_uses_2010_only_for_the_first_window(risk_params, years):
    """2010년 종가는 2011년 첫 거래일 sigma_t의 창(종가 21개)에 들어가는 것만 쓴다."""
    bm = _bm()
    i = int(np.flatnonzero(bm.index >= pd.Timestamp("2011-01-01"))[0])
    n = risk_params.vol_lookback_days
    base = compute_vol_target_annual(bm, risk_params, years)

    older = bm.copy()
    older.iloc[: i - n] *= 1.7  # 창보다 앞선 종가
    assert compute_vol_target_annual(older, risk_params, years) == pytest.approx(base, rel=1e-12)

    inside = bm.copy()
    inside.iloc[i - n] *= 1.05  # 2011년 첫 거래일 sigma_t 창의 첫 종가
    assert compute_vol_target_annual(inside, risk_params, years) != pytest.approx(base, rel=1e-9)


def test_vol_target_needs_the_whole_period(risk_params, years):
    bm = _bm()
    with pytest.raises(InsufficientHistory):  # 2011년 첫 거래일의 sigma_t 창이 없다
        compute_vol_target_annual(bm.loc["2011-01-01":], risk_params, years)
    with pytest.raises(ValueError, match="2015"):  # 2015년이 끝까지 들어왔는지 알 수 없다
        compute_vol_target_annual(bm.loc[:"2015-12-31"], risk_params, years)


def test_regime_mc_is_the_same_as_m8_and_needs_no_vol_target(cfg):
    unfilled = RiskLayerParams.from_prereg(cfg)  # vol_target_annual이 FILL_이어도 Step 5는 계산한다
    bm = bm_crash_rebound_path()
    table = market_state_table(bm, unfilled, mode="no_exposure")["regime_mc"].dropna()
    got = {d: compute_regime_mc(d, bm_close=bm, params=unfilled) for d in table.index[::3]}
    assert all(v == int(table[d]) for d, v in got.items())
    assert set(got.values()) == {0, 1}


def test_regime_mc_ignores_prices_after_t(cfg):
    params = RiskLayerParams.from_prereg(cfg)
    bm = bm_crash_rebound_path()
    t = bm.index[470]
    changed = bm.copy()
    changed.iloc[471:] *= 0.5
    assert compute_regime_mc(t, bm_close=changed, params=params) == compute_regime_mc(t, bm_close=bm, params=params)


def _path_from_2010(rebound: float) -> tuple[pd.Series, pd.Timestamp]:
    """data.start_date(2010-01-04)부터 잔잔하다가 2011년 첫 거래일까지 20거래일 동안 rebound만큼 오르는 경로.

    2011년 첫 거래일에는 bear_t의 창(종가 311개)이 차지 않는다 (D4).
    """
    days = pd.bdate_range("2010-01-04", "2011-06-30")
    t = days[days >= pd.Timestamp("2011-01-01")][0]
    i = days.get_loc(t)
    r = np.random.default_rng(0).normal(0.0, 0.003, len(days))
    r[i - 19 : i + 1] += math.log1p(rebound) / 20
    return pd.Series(250.0 * np.exp(np.cumsum(r)), index=days), t


def test_regime_mc_is_zero_without_bear_history_if_no_rebound(cfg, risk_params):
    """D4: bear_t의 창이 차지 않아도 rebound_t = 0이면 regime_mc_t = bear_t × 0 = 0이다."""
    params = RiskLayerParams.from_prereg(cfg)
    bm, t = _path_from_2010(rebound=0.0)
    row = market_state_table(bm, params, mode="no_exposure").loc[t]
    assert np.isnan(row["bear"]) and row["rebound"] == 0.0
    assert compute_regime_mc(t, bm_close=bm, params=params) == 0
    state = market_state(bm, t, risk_params)  # M8 Step 4~5도 이 날 계산된다
    assert state.regime_mc == 0 and risk_params.exposure_min <= state.exposure <= risk_params.exposure_max


def test_regime_mc_is_missing_without_bear_history_if_rebound(cfg):
    """rebound_t = 1인데 bear_t를 계산할 수 없으면 값을 정할 수 없다 (D4. 실제 2011년 초에는 이런 날이 없다)."""
    params = RiskLayerParams.from_prereg(cfg)
    bm, t = _path_from_2010(rebound=0.15)
    assert market_state_table(bm, params, mode="no_exposure").loc[t, "rebound"] == 1.0
    with pytest.raises(InsufficientHistory):
        compute_regime_mc(t, bm_close=bm, params=params)


@pytest.mark.skipif(HAS_IO, reason="io.py가 있으면 test_regime_mc_reads_kospi200_through_io가 확인한다")
def test_regime_mc_without_io_says_what_is_missing(cfg):
    with pytest.raises(RuntimeError, match="D-02"):
        compute_regime_mc("2015-06-30", params=RiskLayerParams.from_prereg(cfg))


@pytest.mark.skipif(not HAS_IO, reason="io.py(D-02)가 병합되면 실행한다")
def test_regime_mc_reads_kospi200_through_io(cfg, tmp_path, monkeypatch):
    """bm_close를 넘기지 않으면 io.read_table("index")의 KOSPI200 종가로 계산한다 (R-18)."""
    import src.common.io as io

    monkeypatch.setattr(io, "ROOT", tmp_path)
    params = RiskLayerParams.from_prereg(cfg)
    bm = bm_crash_rebound_path()
    io.write_table("index", pd.DataFrame({"date": bm.index, "index_code": "KOSPI200", "close": bm.to_numpy()}))
    table = market_state_table(bm, params, mode="no_exposure")["regime_mc"].dropna()
    days = table.index[::25]
    assert [compute_regime_mc(d, params=params) for d in days] == [int(table[d]) for d in days]


def test_report_metrics_and_data_fingerprint(cfg, risk_params, years):
    bm = _bm()
    m = vol_target_report.vol_target_metrics(bm, cfg)
    assert m["vol_target_annual"] == compute_vol_target_annual(bm, risk_params, years)
    used = m["kospi200_used"]
    assert used["n_closes"] == m["n_sigma_days"] + risk_params.vol_lookback_days
    assert used["first_date"] < "2011-01-01" and used["last_date"] <= "2015-12-31"
    changed = bm.copy()
    changed[changed.index > pd.Timestamp("2015-12-31")] *= 2.0  # 2016년 이후는 지문에도 들어가지 않는다
    assert vol_target_report.vol_target_metrics(changed, cfg)["kospi200_used"] == used


def test_report_writes_run_folder_once(tmp_path, monkeypatch):
    monkeypatch.setattr(vol_target_report, "ROOT", tmp_path)
    monkeypatch.setattr(vol_target_report, "load_kospi200", _bm)
    monkeypatch.setattr(vol_target_report, "_git_commit", lambda: "abc\n")
    config = str(Path(__file__).parent / "prereg_subset.yaml")
    out = vol_target_report.main(["--run-id", "20261103_B_test", "--config", config])
    assert out == tmp_path / "reports" / "20261103_B_test"
    assert sorted(p.name for p in out.iterdir()) == ["config.yaml", "git_commit.txt", "metrics.json"]
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert (metrics["run_id"], metrics["task"]) == ("20261103_B_test", "B-02")
    with pytest.raises(SystemExit, match="R-11"):  # 같은 run_id는 덮어쓰지 않는다
        vol_target_report.main(["--run-id", "20261103_B_test", "--config", config])
    with pytest.raises(SystemExit, match="run_id"):
        vol_target_report.main(["--run-id", "20261103_A_test", "--config", config])
