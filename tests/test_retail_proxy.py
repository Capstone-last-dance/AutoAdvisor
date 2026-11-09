"""DEC-22 코호트 선택의 시점과 G2 d_t 결측·HAC 검증."""

from datetime import date

import pandas as pd
import pytest

from src.backtest.retail_proxy import cohort_effect_hac, select_retail_cohort
from src.backtest.retail_proxy import CohortEffect, evaluate_g2, simulate_retail_proxy
from src.backtest.accounting import CostRates
from src.backtest.engine import ExecutionEngine
from src.common.guards import DataAccessPolicy, GuardViolation


POLICY = DataAccessPolicy(date(2025, 12, 31), date(2026, 1, 1), date(2026, 10, 31), date(2026, 12, 11))
GUARD = {"policy": POLICY, "run_date": date(2026, 11, 9)}


def test_cohort_uses_only_signal_day_and_reports_missing_history():
    days = pd.bdate_range("2024-01-01", periods=67)
    values = pd.DataFrame({
        "A": [10.0] * 60 + [20.0] * 5 + [999.0, 999.0],
        "B": [10.0] * 60 + [15.0] * 5 + [999.0, 999.0],
        "C": [0.0] * 67,
        "D": [10.0] * 64 + [float("nan")] + [10.0] * 2,
    }, index=days)
    first = select_retail_cohort(values, ["B", "A", "C", "D"], str(days[64].date()), top_n=2)
    changed = values.copy()
    changed.loc[days[65]:, ["A", "B"]] = 0.0
    second = select_retail_cohort(changed, ["B", "A", "C", "D"], str(days[64].date()), top_n=2)

    assert first.cohort["ticker"].tolist() == ["A", "B"]
    assert first.cohort["proxy_score"].tolist() == pytest.approx([2.0, 1.5])
    assert first.cohort.equals(second.cohort)
    assert first.missing_history == 1
    assert first.zero_baseline == 1


def test_g2_daily_effect_drops_missing_pairs_and_undefined_days():
    cohorts = pd.DataFrame([
        {"date": day, "ticker": ticker}
        for day in pd.to_datetime(["2023-01-06", "2023-01-13", "2023-01-20"])
        for ticker in ("A", "B", "C")
    ])
    signals = cohorts.assign(crash_flag=[1, 0, 0, 1, 0, 0, 0, 0, 0])
    labels = cohorts.assign(h=20, er=[-0.2, 0.1, 0.1, -0.1, 0.1, None, 0.1, 0.2, 0.3])

    result = cohort_effect_hac(cohorts, signals, labels, **GUARD)
    assert result.daily_d.tolist() == pytest.approx([-0.3, -0.2])
    assert result.mean_d == pytest.approx(-0.25)
    assert result.hac_t_statistic is not None and result.hac_t_statistic < 0
    assert result.signal_days_undefined == 1
    assert result.signal_days_no_flags == 1
    assert result.signal_days_missing_group == 0
    assert result.missing_pairs == 1


def test_gate_avoids_flagged_loss_with_overlapping_cohorts():
    days = pd.bdate_range("2020-01-01", periods=30)
    closes = pd.DataFrame({"000001": [100.0] * 4 + [50.0] * 26,
                           "000002": [100.0] * 30}, index=days)
    cohorts = pd.DataFrame({"date": [days[0], days[0], days[5], days[5]],
                            "ticker": ["000001", "000002", "000001", "000002"]})
    signals = cohorts.assign(crash_flag=[1, 0, 1, 0])
    engine = ExecutionEngine(closes, pd.Series(100.0, index=days), CostRates(0, 0, 0))
    end_date = days[-1].strftime("%Y-%m-%d")

    baseline = simulate_retail_proxy(engine, cohorts, signals, end_date=end_date, gate=False, **GUARD)
    guarded = simulate_retail_proxy(engine, cohorts, signals, end_date=end_date, gate=True, **GUARD)
    assert baseline.mdd > 0
    assert guarded.mdd == pytest.approx(0)
    assert baseline.daily_nav.iloc[-1] < guarded.daily_nav.iloc[-1]
    assert guarded.daily_nav.iloc[-1] == pytest.approx(1.0)
    assert baseline.cagr < guarded.cagr
    effect = CohortEffect(pd.Series([-0.2, -0.1]), -0.15, -4.0, 0, 0)
    decision = evaluate_g2(baseline, guarded, effect)
    assert decision.mdd_reduction_relative == pytest.approx(1.0)
    assert decision.cagr_gate_not_lower is True
    assert decision.passed is True


def test_cohort_exit_retries_after_lower_limit_and_buy_cost_scales_cash():
    days = pd.bdate_range("2020-01-01", periods=25)
    closes = pd.DataFrame({"000001": [100.0] * 21 + [70.5, 75.0, 75.0, 75.0]}, index=days)
    cohorts = pd.DataFrame({"date": [days[0]], "ticker": ["000001"]})
    signals = cohorts.assign(crash_flag=0)
    engine = ExecutionEngine(closes, pd.Series(100.0, index=days),
                             CostRates(0.002, 0.01, 0))
    result = simulate_retail_proxy(engine, cohorts, signals,
                                   end_date=days[-1].strftime("%Y-%m-%d"),
                                   gate=False, cohort_fraction=1.0, **GUARD)
    assert result.daily_nav.iloc[0] == pytest.approx(1 / 1.01)
    assert result.ledger.loc[result.ledger["date"] == days[21].strftime("%Y-%m-%d"),
                             "rejected"].iloc[0]["000001"] == "near_lower_limit"
    assert result.ledger.loc[result.ledger["date"] == days[22].strftime("%Y-%m-%d"),
                             "cost"].iloc[0] > 0


def test_g2_does_not_claim_success_without_drawdown_or_hac_sample():
    days = pd.bdate_range("2020-01-01", periods=25)
    closes = pd.DataFrame({"000001": [100.0] * 25}, index=days)
    engine = ExecutionEngine(closes, pd.Series(100.0, index=days), CostRates(0, 0, 0))
    cohort = pd.DataFrame({"date": [days[0]], "ticker": ["000001"]})
    run = simulate_retail_proxy(engine, cohort, cohort.assign(crash_flag=0),
                                end_date=days[-1].strftime("%Y-%m-%d"), gate=False, **GUARD)
    effect = CohortEffect(pd.Series(dtype="float64"), None, None, 1, 0)
    decision = evaluate_g2(run, run, effect)
    assert decision.mdd_reduction_relative is None
    assert decision.passed is None


def test_g2_outcomes_reject_unopened_holdout():
    days = pd.bdate_range("2026-10-01", periods=30)
    cohorts = pd.DataFrame({"date": [days[0]], "ticker": ["000001"]})
    signals = cohorts.assign(crash_flag=1)
    labels = cohorts.assign(h=20, er=-0.2)
    with pytest.raises(GuardViolation):
        cohort_effect_hac(cohorts, signals, labels, **GUARD)
    closes = pd.DataFrame({"000001": [100.0] * len(days)}, index=days)
    engine = ExecutionEngine(closes, pd.Series(100.0, index=days), CostRates(0, 0, 0))
    with pytest.raises(GuardViolation):
        simulate_retail_proxy(engine, cohorts, signals,
                              end_date=days[-1].strftime("%Y-%m-%d"), gate=False, **GUARD)
