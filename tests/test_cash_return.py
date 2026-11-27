"""C 평가 보정이 B의 고정 탐색 결과와 같은 규칙을 쓰는지 확인한다."""

import pandas as pd
import pytest

from src.backtest.cash_return import cash_fraction, cd91_adjust


def test_cd91_uses_previous_cash_and_month_across_weekend():
    days = pd.DatetimeIndex(["2026-01-30", "2026-02-02", "2026-02-03"])
    nav = pd.Series([1.0, 0.98, 1.02], index=days)
    ledger = [
        {"date": str(day.date()), "cash": c, "nav_after": n}
        for day, c, n in zip(days, [0.5, 0.196, 0.204], nav)
    ]
    weights = cash_fraction(ledger)
    monthly = pd.Series([0.0365, 0.073], index=pd.period_range("2026-01", periods=2, freq="M"))
    result = cd91_adjust(nav, weights, monthly)
    first_cash = (1 + 0.0365) ** (3 / 365) - 1
    assert result.iloc[1] == pytest.approx(0.98 + 0.5 * first_cash)
    second_cash = (1 + 0.073) ** (1 / 365) - 1
    assert result.iloc[2] == pytest.approx(result.iloc[1] * (1.02 / 0.98 + 0.2 * second_cash))
    assert result.iloc[0] == 1.0


def test_missing_rate_and_misaligned_cash_fail_closed():
    days = pd.to_datetime(["2026-01-30", "2026-02-02", "2026-02-03"])
    nav = pd.Series([1.0, 1.0, 1.0], index=days)
    weights = pd.Series([1.0, 1.0, 1.0], index=days)
    jan = pd.Series([0.03], index=pd.PeriodIndex(["2026-01"], freq="M"))
    with pytest.raises(KeyError, match="2026-02"):
        cd91_adjust(nav, weights, jan)
    with pytest.raises(ValueError, match="같은 중복 없는"):
        cd91_adjust(nav, weights.iloc[:-1], jan)


def test_zero_cash_preserves_engine_nav():
    days = pd.DatetimeIndex(pd.to_datetime(["2025-12-29", "2025-12-30"])).as_unit("ns")
    nav = pd.Series([1.0, 0.9], index=days)
    weights = pd.Series([0.0, 0.0], index=days)
    monthly = pd.Series([0.03], index=pd.PeriodIndex(["2025-12"], freq="M"))
    pd.testing.assert_series_equal(cd91_adjust(nav, weights, monthly), nav, check_freq=False)
