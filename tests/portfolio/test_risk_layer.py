"""M8 리스크 레이어 (6.9절). 경계 사례 묶음과 2011~2025년 그림은 작업 B-04에서 추가한다."""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import statistics

import numpy as np
import pandas as pd
import pytest

from src.portfolio.params import PreregValueMissing, RiskLayerParams
from src.portfolio.risk_layer import (
    CASH,
    InsufficientHistory,
    RiskLayerViolation,
    apply_risk_layer,
    check_invariants,
    market_state,
    market_state_table,
    sectors_at,
)
from tests.portfolio.fakes import bm_crash_rebound_path as _bm_path


def _reference(bm: np.ndarray, i: int, p: RiskLayerParams) -> tuple[float, int, float]:
    """6.9절 Step 4~5 식을 그대로 옮긴 느린 구현."""
    rp = p.momentum_crash_regime
    r = [math.log(bm[s] / bm[s - 1]) for s in range(i - p.vol_lookback_days + 1, i + 1)]
    sigma = statistics.stdev(r) * math.sqrt(p.annualization_daily)

    def dd(s: int) -> float:
        return 1.0 - bm[s] / max(bm[s - rp.drawdown_high_lookback_days + 1 : s + 1])

    bear = int(max(dd(s) for s in range(i - rp.drawdown_recent_window_days + 1, i + 1)) >= rp.drawdown_threshold)
    rebound = int(bm[i] / bm[i - rp.rebound_lookback_days] - 1.0 >= rp.rebound_threshold)
    regime = bear * rebound
    e = min(p.exposure_max, max(p.exposure_min, p.vol_target_annual / sigma))
    if regime == 1:
        e = min(e, rp.exposure_cap)
    return sigma, regime, e


def test_step4_5_match_formula(risk_params):
    bm = _bm_path()
    table = market_state_table(bm, risk_params)
    first = 252 + 60 - 1 - 1  # 창을 처음으로 모두 채우는 행 번호
    assert table.iloc[:first].isna().any(axis=1).all()
    regimes = set()
    for i in range(first, len(bm), 3):
        sigma, regime, e = _reference(bm.to_numpy(), i, risk_params)
        row = table.iloc[i]
        assert row["sigma"] == pytest.approx(sigma, rel=1e-12)
        assert int(row["regime_mc"]) == regime
        assert row["exposure"] == pytest.approx(e, rel=1e-12)
        regimes.add(regime)
    assert regimes == {0, 1}


def test_regime_mc_caps_exposure(risk_params):
    params = dataclasses.replace(risk_params, vol_target_annual=10.0)  # 변동성 항은 1.00이 되게
    bm = _bm_path()
    table = market_state_table(bm, params).dropna()
    on, off = table[table["regime_mc"] == 1], table[table["regime_mc"] == 0]
    assert len(on) > 0 and len(off) > 0
    assert (on["exposure"] == params.momentum_crash_regime.exposure_cap).all()
    assert (off["exposure"] == params.exposure_max).all()


def test_no_bm_after_t_is_used(risk_params):
    bm = _bm_path()
    t = bm.index[400]
    future_changed = bm.copy()
    future_changed.iloc[401:] *= 3.0
    expected = market_state(bm.loc[:t], t, risk_params)
    assert market_state(bm, t, risk_params) == expected
    assert market_state(future_changed, t, risk_params) == expected


def test_short_history_and_fill_value_stop(cfg, risk_params):
    bm = _bm_path()
    with pytest.raises(InsufficientHistory):
        market_state(bm, bm.index[200], risk_params)
    unfilled = RiskLayerParams.from_prereg(cfg)  # vol_target_annual = FILL_
    with pytest.raises(PreregValueMissing):
        market_state(bm, bm.index[400], unfilled)
    assert market_state(bm, bm.index[400], unfilled, mode="no_exposure").exposure == 1.0


@pytest.fixture
def case():
    """종목 1~4는 업종 X, 5는 업종 정보 없음, 6은 crash_flag = 1, 7은 쿨다운, 8은 U_t 밖."""
    tickers = [f"{i:06d}" for i in range(1, 9)]
    w_raw = pd.Series([0.12, 0.12, 0.12, 0.12, 0.08, 0.05, 0.05, 0.05], index=tickers)
    universe = tickers[:7]
    signals = pd.DataFrame({"ticker": universe, "crash_flag": [0, 0, 0, 0, 0, 1, 0]})
    sectors = {"000001": "X", "000002": "X", "000003": "X", "000004": "X", "000006": "Y", "000007": "Y"}
    bm = _bm_path()
    return {
        "w_raw": w_raw,
        "signals": signals,
        "sectors": sectors,
        "bm_close": bm,
        "cooldown": ["000007"],
        "universe": universe,
        "date": bm.index[400],
    }


def _weights(pos: pd.DataFrame) -> dict[str, float]:
    return dict(zip(pos["ticker"], pos["target_weight"]))


def test_steps_1_to_3_and_6(case, risk_params):
    pos = apply_risk_layer(**case, params=risk_params, constructor="buffer", mode="no_exposure")
    w = _weights(pos)
    # Step 1: 6, 7, 8 제외. Step 2: 0.12 → 0.10. Step 3: 업종 X 합 0.40 → × 0.75
    assert set(w) == {"000001", "000002", "000003", "000004", "000005", CASH}
    for tk in ("000001", "000002", "000003", "000004"):
        assert w[tk] == pytest.approx(0.075, abs=1e-15)
    assert w["000005"] == pytest.approx(0.08, abs=1e-15)
    assert w[CASH] == pytest.approx(0.62, abs=1e-15)
    assert list(pos.columns) == ["date", "ticker", "target_weight", "constructor", "exposure", "regime_mc"]
    assert pos["date"].iloc[0] == case["date"].date()
    assert (pos["constructor"] == "buffer").all() and (pos["exposure"] == 1.0).all()
    assert pos["target_weight"].dtype == np.float64 and pos["regime_mc"].dtype == np.int8

    low_target = dataclasses.replace(risk_params, vol_target_annual=0.08)  # E_t < 1이 되게 하는 테스트 값
    full = apply_risk_layer(**case, params=low_target, constructor="buffer")
    e_t = market_state(case["bm_close"], case["date"], low_target).exposure
    assert 0.30 <= e_t < 1.0
    assert full["target_weight"].iloc[:-1].tolist() == pytest.approx(
        (pos["target_weight"].iloc[:-1] * e_t).tolist(), abs=1e-15
    )
    assert full["target_weight"].sum() == pytest.approx(1.0, abs=1e-12)


def test_unknown_sector_is_one_group(case, risk_params):
    case["sectors"] = {}
    w = _weights(apply_risk_layer(**case, params=risk_params, constructor="rl", mode="no_exposure"))
    # 종목 1~5가 모두 '미분류': 0.10 × 4 + 0.08 = 0.48 → × 0.30 / 0.48
    assert w["000001"] == pytest.approx(0.10 * 0.30 / 0.48, abs=1e-15)
    assert w["000005"] == pytest.approx(0.08 * 0.30 / 0.48, abs=1e-15)


def test_mode_off_keeps_only_crash_and_universe_filters(case, risk_params):
    pos = apply_risk_layer(**case, params=risk_params, constructor="buffer", mode="off")
    w = _weights(pos)
    assert w["000001"] == pytest.approx(0.12) and w["000007"] == pytest.approx(0.05)
    assert "000006" not in w and "000008" not in w
    assert (pos["exposure"] == 1.0).all()


def test_invariants_on_random_inputs(risk_params):
    rng = np.random.default_rng(1)
    bm = _bm_path()
    t = bm.index[450]
    for _ in range(200):
        n = int(rng.integers(1, 60))
        tickers = [f"{i:06d}" for i in range(1, n + 1)]
        w_raw = pd.Series(rng.dirichlet(np.full(n + 1, 0.3))[:n], index=tickers)
        sectors = {tk: f"S{rng.integers(0, 4)}" for tk in tickers if rng.random() > 0.2}
        universe = [tk for tk in tickers if rng.random() > 0.1]
        signals = pd.DataFrame({"ticker": universe, "crash_flag": (rng.random(len(universe)) < 0.1).astype(int)})
        cooldown = [tk for tk in universe if rng.random() < 0.05]
        pos = apply_risk_layer(w_raw, signals, sectors, bm, cooldown, universe, t, risk_params, "rl")

        stocks = pos[pos["ticker"] != CASH]
        blocked = set(signals.loc[signals["crash_flag"] == 1, "ticker"]) | set(cooldown) | (set(tickers) - set(universe))
        assert not blocked & set(stocks["ticker"])
        assert (pos["target_weight"] >= 0.0).all()
        assert (stocks["target_weight"] <= risk_params.stock_cap + 1e-9).all()
        by_sector = stocks.groupby(stocks["ticker"].map(lambda tk: sectors.get(tk, "미분류")))["target_weight"].sum()
        assert (by_sector <= risk_params.sector_cap + 1e-9).all()
        assert pos["target_weight"].sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("w", "codes"),
    [
        ([0.2], [0]),  # 종목 한도
        ([0.1, 0.1, 0.1, 0.1], [0, 0, 0, 0]),  # 업종 한도
        ([0.1] * 11, list(range(11))),  # 현금 < 0
        ([-0.01], [0]),
        ([np.nan], [0]),
    ],
)
def test_violation_raises(risk_params, w, codes):
    with pytest.raises(RiskLayerViolation):
        check_invariants(np.array(w), np.array(codes), risk_params)


def test_mode_off_checks_only_basic_invariants(risk_params):
    assert check_invariants(np.array([0.5]), np.array([0]), risk_params, mode="off") == pytest.approx(0.5)
    with pytest.raises(RiskLayerViolation):
        check_invariants(np.array([0.7, 0.4]), np.array([0, 1]), risk_params, mode="off")


def test_sectors_at_uses_valid_rows_only():
    sector = pd.DataFrame(
        {
            "ticker": ["000001", "000001", "000002", "000003"],
            "sector": ["A", "B", "C", None],
            "valid_from": [dt.date(2010, 1, 1), dt.date(2015, 1, 1), dt.date(2010, 1, 1), dt.date(2010, 1, 1)],
            "valid_to": [dt.date(2014, 12, 31), None, None, None],
        }
    )
    assert sectors_at(sector, "2014-06-30") == {"000001": "A", "000002": "C"}
    assert sectors_at(sector, "2015-01-02") == {"000001": "B", "000002": "C"}
    overlapping = sector.assign(valid_to=None)
    with pytest.raises(ValueError):
        sectors_at(overlapping, "2015-01-02")


def test_signals_of_other_dates_are_rejected(case, risk_params):
    case["signals"] = case["signals"].assign(date=case["date"])
    apply_risk_layer(**case, params=risk_params, constructor="rl")
    case["signals"].loc[0, "date"] = case["date"] + pd.Timedelta(days=7)
    with pytest.raises(ValueError, match="R-03"):
        apply_risk_layer(**case, params=risk_params, constructor="rl")
