"""DEC-22 코호트 선택의 시점과 G2 d_t 결측·HAC 검증."""

import pandas as pd
import pytest

from src.backtest.retail_proxy import cohort_effect_hac, select_retail_cohort


def test_cohort_uses_only_signal_day_and_reports_missing_history():
    days = pd.bdate_range("2024-01-01", periods=67)
    volumes = pd.DataFrame({
        "A": [10.0] * 60 + [20.0] * 5 + [999.0, 999.0],
        "B": [10.0] * 60 + [15.0] * 5 + [999.0, 999.0],
        "C": [0.0] * 67,
        "D": [10.0] * 64 + [float("nan")] + [10.0] * 2,
    }, index=days)
    first = select_retail_cohort(volumes, ["B", "A", "C", "D"], str(days[64].date()), top_n=2)
    changed = volumes.copy()
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

    result = cohort_effect_hac(cohorts, signals, labels)
    assert result.daily_d.tolist() == pytest.approx([-0.3, -0.2])
    assert result.mean_d == pytest.approx(-0.25)
    assert result.hac_t_statistic is not None and result.hac_t_statistic < 0
    assert result.signal_days_undefined == 1
    assert result.missing_pairs == 1
