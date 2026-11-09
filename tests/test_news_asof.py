"""날짜만 아는 뉴스 집계의 t 시점 누수와 0건·결측 분리."""

import pandas as pd
import pytest

from src.backtest.news_asof import load_news_windows, news_count_spike_at


@pytest.fixture
def windows(tmp_path):
    config = tmp_path / "prereg.yaml"
    config.write_text("features:\n  news_window_offsets: [-5, -1]\n  news_baseline_offsets: [-65, -6]\n", encoding="utf-8")
    return load_news_windows(config)


def table():
    days = pd.bdate_range("2026-01-01", periods=66)
    rows = pd.DataFrame({"date": days[:65], "ticker": ["005930"] * 65,
                         "news_count": [2] * 60 + [4] * 5})
    rows["available_at"] = rows["date"].map(
        lambda day: pd.Timestamp(f"{day:%Y-%m-%d} 23:59:59", tz="Asia/Seoul")
    )
    return days, rows


def test_complete_count_window_and_confirmed_zero_are_available(windows):
    days, rows = table()
    result = news_count_spike_at(rows, days, str(days[-1].date()), "005930", windows=windows)
    assert result.spike == pytest.approx((20 + 1) / (120 * 5 / 60 + 1))
    rows["news_count"] = 0
    assert news_count_spike_at(rows, days, str(days[-1].date()), "005930", windows=windows).spike == 1.0


def test_missing_and_late_aggregate_are_not_replaced_with_zero(windows):
    days, rows = table()
    missing_day = days[20]
    late_day = days[64]
    rows = rows.loc[rows["date"] != missing_day].copy()
    rows.loc[rows["date"] == late_day, "available_at"] = pd.Timestamp(
        f"{days[-1]:%Y-%m-%d} 16:00:00", tz="Asia/Seoul"
    )
    result = news_count_spike_at(rows, days, str(days[-1].date()), "005930", windows=windows)
    assert result.spike is None
    assert result.missing_days == (missing_day.strftime("%Y-%m-%d"),)
    assert result.late_days == (late_day.strftime("%Y-%m-%d"),)


def test_signal_day_news_cannot_change_past_window(windows):
    days, rows = table()
    before = news_count_spike_at(rows, days, str(days[-1].date()), "005930", windows=windows)
    signal_day = pd.DataFrame([{
        "date": days[-1], "ticker": "005930", "news_count": 100000,
        "available_at": pd.Timestamp(f"{days[-1]:%Y-%m-%d} 23:59:59", tz="Asia/Seoul"),
    }])
    after = news_count_spike_at(pd.concat([rows, signal_day]), days, str(days[-1].date()), "005930", windows=windows)
    assert after == before


def test_null_count_row_is_schema_error_not_missing_day(windows):
    days, rows = table()
    rows.loc[0, "news_count"] = None
    with pytest.raises(ValueError, match="news_count 결측"):
        news_count_spike_at(rows, days, str(days[-1].date()), "005930", windows=windows)


def test_weekend_aggregate_must_be_available_before_signal(windows):
    days, rows = table()
    prior = days[-2]
    rows.loc[rows["date"] == prior, "available_at"] = pd.Timestamp(
        f"{days[-1]:%Y-%m-%d} 08:00:00", tz="Asia/Seoul"
    )
    assert news_count_spike_at(rows, days, str(days[-1].date()), "005930", windows=windows).spike is not None
    rows.loc[rows["date"] == prior, "available_at"] = pd.Timestamp(
        f"{days[-1]:%Y-%m-%d} 16:00:00", tz="Asia/Seoul"
    )
    result = news_count_spike_at(rows, days, str(days[-1].date()), "005930", windows=windows)
    assert result.spike is None
    assert result.late_days == (prior.strftime("%Y-%m-%d"),)
