"""M5 지표의 단위·부호·연율화가 손계산 예제와 맞는지 검증한다."""

import math

import pandas as pd
import pytest

from src.backtest.metrics import (
    annual_excess_return,
    annual_turnover,
    cagr,
    capture_ratio,
    information_ratio,
    lift_at_10pct,
    max_drawdown,
    rank_ic_with_hac,
    rebound_capture,
    roc_auc,
    turnover,
    validation_ir,
    weekly_active_returns,
)


def test_cagr_annual_excess_and_drawdown():
    assert cagr([100, 110], trading_days=252) == pytest.approx(0.10)
    assert annual_excess_return(
        [100, 110], [100, 105], trading_days=252
    ) == pytest.approx(0.05)
    assert max_drawdown([100, 120, 90, 110]) == pytest.approx(0.25)


def test_weekly_active_returns_ir_and_undefined_zero_variance():
    active = weekly_active_returns([100, 110, 115.5], [100, 105, 110.25])
    assert active.tolist() == pytest.approx([0.05, 0.0])
    expected = (0.025 / math.sqrt(0.00125)) * math.sqrt(52)
    assert information_ratio(active) == pytest.approx(expected)
    assert information_ratio([0.01, 0.01]) is None


def test_turnover_ignores_cash_and_annualizes_by_trading_days():
    assert turnover(
        {"005930": 0.4, "000660": 0.4, "CASH": 0.2},
        {"005930": 0.2, "000660": 0.5, "CASH": 0.3},
    ) == pytest.approx(0.15)
    assert annual_turnover([0.1, 0.2], trading_days=252) == pytest.approx(0.3)


def test_rank_ic_uses_signal_day_spearman_and_hac_t():
    frame = pd.DataFrame(
        [
            {"date": day, "rank_score": score, "er": er}
            for day, excesses in (
                ("2023-01-06", [0.1, 0.2, 0.3]),
                ("2023-01-13", [0.3, 0.2, 0.1]),
                ("2023-01-20", [0.2, 0.3, 0.1]),
            )
            for score, er in zip((1, 2, 3), excesses)
        ]
    )
    result = rank_ic_with_hac(frame)
    assert result.signal_days_used == 3
    assert result.signal_days_undefined == 0
    assert result.missing_pairs == 0
    assert result.mean_ic == pytest.approx((1 - 1 - 0.5) / 3)
    assert result.hac_t_statistic is not None


def test_auc_lift_capture_and_rebound():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert lift_at_10pct([0, 1, 0, 0], [0, 1, 0, 1]) == pytest.approx(2.0)
    assert capture_ratio(-0.08, -0.10) == pytest.approx(0.8)
    assert rebound_capture(
        [0.1, 0.5, -0.05], [0.05, -0.5, 0.02], [1, 0, 1]
    ) == pytest.approx(((1.1 * 0.95) - 1) / ((1.05 * 1.02) - 1))


def test_undefined_metrics_are_not_written_as_zero():
    assert roc_auc([0, 0], [0.1, 0.2]) is None
    assert lift_at_10pct([0, 0], [0, 1]) is None
    assert capture_ratio(0.1, 0.0) is None
    assert rebound_capture([0.1], [0.1], [0]) is None
    assert math.isnan(validation_ir([0.01, 0.01]))
    assert validation_ir([0.01, 0.02]) == pytest.approx(information_ratio([0.01, 0.02]))


def test_rank_ic_counts_signal_day_with_only_missing_pairs():
    result = rank_ic_with_hac(
        pd.DataFrame(
            {
                "date": ["2023-01-06", "2023-01-06"],
                "rank_score": [None, None],
                "er": [0.1, 0.2],
            }
        )
    )
    assert result.mean_ic is None
    assert result.signal_days_undefined == 1
    assert result.missing_pairs == 2
