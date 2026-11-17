"""M8 리스크 레이어 (6.9절). 완료 기준 (1) 불변식, (2) 경계 사례 (작업 B-04). (3) 그림은 test_market_state_report.py."""
from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.util
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
    with pytest.raises(InsufficientHistory):  # sigma_t·rebound_t 창(종가 21개)이 차지 않는다
        market_state(bm, bm.index[10], risk_params)
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
    assert pos["date"].dtype == "datetime64[ns]" and (pos["date"] == case["date"].normalize()).all()
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


@pytest.mark.skipif(importlib.util.find_spec("src.common.io") is None, reason="io.py(D-02)가 병합되면 실행한다")
def test_positions_round_trip_through_io(case, risk_params, tmp_path, monkeypatch):
    """positions가 9.3절 스키마(io.validate)를 통과하고, date·run_id 파일로 쓰고 읽어도 그대로다 (R-18)."""
    import src.common.io as io

    monkeypatch.setattr(io, "ROOT", tmp_path)
    pos = apply_risk_layer(**case, params=risk_params, constructor="buffer")
    day = case["date"].strftime("%Y-%m-%d")
    io.write_table("positions", pos, date=day)
    io.write_table("positions", pos, run_id="20261116_B_io_check")
    pd.testing.assert_frame_equal(io.read_table("positions", date=day), pos)
    pd.testing.assert_frame_equal(io.read_table("positions", run_id="20261116_B_io_check"), pos)


# 6.9절 완료 기준 (2) 경계 사례: 한 업종에 20종목 집중, 보유 종목 5개, regime_mc = 1, sigma_t = 3 × vol_target_annual


def _alternating(n: int) -> np.ndarray:
    return np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def _boundary_bm(sigma_t: float, regime: int, p: RiskLayerParams) -> pd.Series:
    """마지막 날의 sigma_t와 regime_mc_t를 정해 둔 코스피 200 대용 경로 (종가 361개, 무작위 없음).

    마지막 20개 로그수익률은 ±a 교대(합 0)에 일정한 추세를 더한 것이라 표본표준편차가 a × √(20/19)다.
    regime = 1이면 그 앞 40일의 25% 하락으로 bear_t = 1, 마지막 20일의 12% 반등으로 rebound_t = 1이다.
    regime = 0이면 ±0.2% 교대뿐이라 bear_t = rebound_t = 0이다.
    """
    n = p.vol_lookback_days
    assert p.momentum_crash_regime.rebound_lookback_days == n  # 마지막 n일이 sigma_t와 rebound_t의 창을 함께 채운다
    a = sigma_t / math.sqrt(p.annualization_daily) / math.sqrt(n / (n - 1))
    if regime:
        fall = np.full(40, math.log(0.75) / 40)
        r = np.concatenate([0.002 * _alternating(300), fall, math.log(1.12) / n + a * _alternating(n)])
    else:
        r = np.concatenate([0.002 * _alternating(340), a * _alternating(n)])
    closes = 1000.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return pd.Series(closes, index=pd.bdate_range("2012-01-02", periods=len(closes)))


def _tickers(n: int) -> list[str]:
    return [f"{i:06d}" for i in range(1, n + 1)]


def _apply_last_day(w_raw: pd.Series, sectors: dict[str, str], bm: pd.Series, p: RiskLayerParams) -> pd.DataFrame:
    """w_raw의 종목이 모두 U_t 안에 있고 crash_flag = 0, 쿨다운 없음일 때 경로 마지막 날의 M8 (mode = full)."""
    tickers = list(w_raw.index)
    signals = pd.DataFrame({"ticker": tickers, "crash_flag": 0})
    return apply_risk_layer(w_raw, signals, sectors, bm, [], tickers, bm.index[-1], p, "buffer")


@pytest.mark.parametrize("labeled", [True, False], ids=["sector_X", "unknown_sector"])
def test_boundary_20_stocks_in_one_sector(risk_params, labeled):
    """버퍼 규칙의 20종목 동일가중(각 0.05)이 모두 한 업종이면 Step 3으로 0.30만 남고 0.70은 현금이다."""
    bm = _boundary_bm(0.5 * risk_params.vol_target_annual, 0, risk_params)  # 변동성 항 2 → E_t = 1
    tk = _tickers(20)
    sectors = {t: "X" for t in tk} if labeled else {}  # 업종 정보가 없으면 '미분류' 하나로 묶인다
    pos = _apply_last_day(pd.Series(0.05, index=tk), sectors, bm, risk_params)
    w = _weights(pos)
    assert (pos["exposure"] == 1.0).all()
    assert all(w[t] == pytest.approx(0.015, abs=1e-15) for t in tk) and w[CASH] == pytest.approx(0.70, abs=1e-12)

    # RL처럼 한 종목에 몰려도 Step 2(0.10)를 먼저 자르고 Step 3 배율 0.30 ÷ (0.10 + 19 × 0.03)을 곱한다
    w = _weights(_apply_last_day(pd.Series([0.43] + [0.03] * 19, index=tk), sectors, bm, risk_params))
    scale = 0.30 / 0.67
    assert w[tk[0]] == pytest.approx(0.10 * scale, abs=1e-15)
    assert all(w[t] == pytest.approx(0.03 * scale, abs=1e-15) for t in tk[1:])
    assert w[CASH] == pytest.approx(0.70, abs=1e-12)


def test_boundary_five_holdings(risk_params):
    """보유 예정 5종목이면 1/5 = 0.20이 종목 한도 0.10으로 잘리고 0.50은 현금이다 (다른 종목에 재배분하지 않는다)."""
    bm = _boundary_bm(0.5 * risk_params.vol_target_annual, 0, risk_params)
    tk = _tickers(5)
    w_raw = pd.Series(0.20, index=tk)
    w = _weights(_apply_last_day(w_raw, {t: f"S{i}" for i, t in enumerate(tk)}, bm, risk_params))
    assert all(w[t] == pytest.approx(0.10, abs=1e-15) for t in tk) and w[CASH] == pytest.approx(0.50, abs=1e-12)
    # 3종목이 한 업종이면 업종 합이 한도와 같은 0.30이라 Step 3이 더 줄이지 않는다
    w = _weights(_apply_last_day(w_raw, dict(zip(tk, "XXXYY")), bm, risk_params))
    assert all(w[t] == pytest.approx(0.10, abs=1e-12) for t in tk) and w[CASH] == pytest.approx(0.50, abs=1e-12)


@pytest.mark.parametrize(("vol_term", "expected"), [(2.0, 0.70), (0.6, 0.60)])
def test_boundary_regime_mc_caps_exposure(risk_params, vol_term, expected):
    """regime_mc_t = 1이면 E_t ≤ 0.70이다. 변동성 항이 0.70보다 크면 0.70으로 낮추고, 작으면 그대로 둔다."""
    bm = _boundary_bm(risk_params.vol_target_annual / vol_term, 1, risk_params)
    row = market_state_table(bm, risk_params).iloc[-1]
    assert (row["bear"], row["rebound"], row["regime_mc"]) == (1.0, 1.0, 1.0)
    tk = _tickers(10)
    pos = _apply_last_day(pd.Series(0.10, index=tk), {t: f"S{i}" for i, t in enumerate(tk)}, bm, risk_params)
    w = _weights(pos)
    assert (pos["regime_mc"] == 1).all() and pos["exposure"].iloc[0] == pytest.approx(expected, rel=1e-9)
    assert all(w[t] == pytest.approx(0.10 * expected, rel=1e-9) for t in tk)
    assert w[CASH] == pytest.approx(1.0 - expected, abs=1e-9)


@pytest.mark.parametrize(("multiple", "expected"), [(3.0, 1 / 3), (4.0, 0.30)])
def test_boundary_sigma_multiple_of_target(risk_params, multiple, expected):
    """sigma_t = 3 × vol_target_annual이면 E_t = 1/3이다. 4배면 1/4이 하한 0.30에 걸린다."""
    bm = _boundary_bm(multiple * risk_params.vol_target_annual, 0, risk_params)
    state = market_state(bm, bm.index[-1], risk_params)
    assert state.sigma == pytest.approx(multiple * risk_params.vol_target_annual, rel=1e-9)
    assert state.regime_mc == 0 and state.exposure == pytest.approx(expected, rel=1e-9)
    tk = _tickers(20)
    w = _weights(_apply_last_day(pd.Series(0.05, index=tk), {t: f"S{i}" for i, t in enumerate(tk)}, bm, risk_params))
    assert all(w[t] == pytest.approx(0.05 * expected, rel=1e-9) for t in tk)
    assert w[CASH] == pytest.approx(1.0 - expected, abs=1e-9)


def test_boundary_cases_together(risk_params):
    """한 업종 20종목(→ 0.30)에 regime_mc = 1과 sigma_t = 3배가 겹치면 E_t = min(1/3, 0.70)이라 종목 합 0.10이다."""
    bm = _boundary_bm(3.0 * risk_params.vol_target_annual, 1, risk_params)
    tk = _tickers(20)
    pos = _apply_last_day(pd.Series(0.05, index=tk), {t: "X" for t in tk}, bm, risk_params)
    w = _weights(pos)
    assert (pos["regime_mc"] == 1).all() and pos["exposure"].iloc[0] == pytest.approx(1 / 3, rel=1e-9)
    assert sum(w[t] for t in tk) == pytest.approx(0.10, rel=1e-9) and w[CASH] == pytest.approx(0.90, rel=1e-9)
