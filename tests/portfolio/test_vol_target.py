"""작업 B-02: risk_layer.vol_target_annual(Q-12)과 compute_regime_mc (6.9절)."""
from __future__ import annotations

import importlib.util
import math
import statistics

import numpy as np
import pandas as pd
import pytest

from src.portfolio.params import RiskLayerParams, vol_target_years
from src.portfolio.risk_layer import (
    InsufficientHistory,
    compute_regime_mc,
    compute_vol_target_annual,
    market_state_table,
)
from tests.portfolio.fakes import bm_crash_rebound_path


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


def test_regime_mc_early_2011_is_short_of_history(cfg):
    """data.start_date(2010-01-04)부터의 이력으로는 2011년 초 창을 채우지 못한다 ([결정 필요] D4)."""
    bm = _bm(start="2010-01-04")
    first_2011 = bm.index[bm.index >= pd.Timestamp("2011-01-01")][0]
    with pytest.raises(InsufficientHistory):
        compute_regime_mc(first_2011, bm_close=bm, params=RiskLayerParams.from_prereg(cfg))


@pytest.mark.skipif(importlib.util.find_spec("src.common.io") is not None, reason="io.py가 생기면 실제 경로로 확인한다")
def test_regime_mc_without_io_says_what_is_missing(cfg):
    with pytest.raises(RuntimeError, match="D-02"):
        compute_regime_mc("2015-06-30", params=RiskLayerParams.from_prereg(cfg))
