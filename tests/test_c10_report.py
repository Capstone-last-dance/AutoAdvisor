"""C-10 window boundaries and official first-fill convention."""

import pandas as pd
import pytest

from src.backtest.c10_report import _period_end, _window


def test_window_uses_previous_close_and_excludes_future_dates():
    nav = pd.DataFrame({
        "KOSPI200": [100.0, 90.0, 99.0, 198.0],
        "buffer_full": [1.0, 0.8, 0.88, 1.76],
    }, index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2026-01-02"]))
    result = _window(nav, "2025-01-03", "2025-01-06", "buffer_full")
    assert result["base_day"] == "2025-01-02"
    assert result["daily_returns"] == 2
    assert result["return"] == pytest.approx(-0.12)
    assert result["bm_return"] == pytest.approx(-0.01)
    assert result["mdd"] == pytest.approx(0.2)


def test_first_fill_cagr_starts_at_pretrade_nav_one():
    nav = pd.DataFrame({"KOSPI200": [100.0, 110.0, 121.0],
                        "buffer_full": [0.99, 1.10, 1.21]},
                       index=pd.to_datetime(["2016-01-11", "2016-01-12", "2016-01-13"]))
    result = _window(nav, "2016-01-01", "2016-12-31", "buffer_full")
    assert result["base_day"] == "first_fill_pretrade"
    assert result["daily_returns"] == 2
    assert result["return"] == pytest.approx(0.21)
    assert result["bm_return"] == pytest.approx(0.21)


def test_last_signal_target_has_remaining_2025_evaluation_days():
    days = pd.to_datetime(["2025-12-26", "2025-12-29", "2025-12-30", "2026-01-02"])
    assert _period_end(days, "2025-12-26") == ("2025-12-29", "2025-12-30")
