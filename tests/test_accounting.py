"""M5 공통 회계의 손계산 예제와 미체결 경계를 검증한다."""

import pytest

from src.backtest.accounting import (
    CostRates,
    blocked_reason,
    execute_orders,
    portfolio_nav,
    trade_cost,
)

RATES = CostRates(
    sell_tax=0.0020,
    commission_per_side=0.00015,
    slippage_per_side=0.0010,
)


def test_buy_sell_costs_and_stress_multiplier_match_hand_calculation():
    assert trade_cost(
        buy_notional=400, sell_notional=500, rates=RATES
    ) == pytest.approx(400 * 0.00115 + 500 * 0.00315)
    doubled = CostRates(0.0020, 0.00015, 0.0010, stress_multiplier=2.0)
    assert trade_cost(
        buy_notional=400, sell_notional=500, rates=doubled
    ) == pytest.approx(2 * (400 * 0.00115 + 500 * 0.00315))


def test_sell_proceeds_then_buy_and_daily_nav():
    result = execute_orders(
        orders={"005930": -5.0, "000660": 4.0},
        holdings={"005930": 10.0},
        cash=100.0,
        execution_closes={"005930": 100.0, "000660": 100.0},
        previous_closes={"005930": 100.0, "000660": 100.0},
        halted_tickers=frozenset(),
        rates=RATES,
        limit_move_abs_return=0.295,
    )
    assert result.holdings == {"005930": 5.0, "000660": 4.0}
    assert result.total_cost == pytest.approx(2.035)
    assert result.cash == pytest.approx(197.965)
    assert portfolio_nav(
        holdings=result.holdings,
        closes={"005930": 100.0, "000660": 100.0},
        cash=result.cash,
    ) == pytest.approx(1100.0 - 2.035)


def test_limit_moves_and_halt_reject_without_changing_cash():
    result = execute_orders(
        orders={"005930": -1.0, "000660": 1.0, "035420": 1.0},
        holdings={"005930": 1.0},
        cash=1000.0,
        execution_closes={"005930": 70.5, "000660": 129.5, "035420": 100.0},
        previous_closes={"005930": 100.0, "000660": 100.0, "035420": 100.0},
        halted_tickers=frozenset({"035420"}),
        rates=RATES,
        limit_move_abs_return=0.295,
    )
    assert result.rejected == {
        "005930": "near_lower_limit",
        "000660": "near_upper_limit",
        "035420": "halted",
    }
    assert result.holdings == {"005930": 1.0}
    assert result.cash == 1000.0
    assert result.fills == ()


def test_cash_shortage_is_explicit_instead_of_hidden_partial_fill():
    with pytest.raises(ValueError, match="가용 현금"):
        execute_orders(
            orders={"005930": 1.0},
            holdings={},
            cash=100.0,
            execution_closes={"005930": 100.0},
            previous_closes={"005930": 100.0},
            halted_tickers=frozenset(),
            rates=RATES,
            limit_move_abs_return=0.295,
        )


def test_delisted_zero_mark_reduces_nav_without_inventing_recovery_price():
    assert (
        portfolio_nav(holdings={"005930": 10.0}, closes={"005930": 0.0}, cash=50.0)
        == 50.0
    )
    assert (
        blocked_reason(
            side="buy",
            previous_close=100.0,
            execution_close=129.5,
            is_halted=False,
            limit_move_abs_return=0.295,
        )
        == "near_upper_limit"
    )


def test_zero_cost_full_equal_weight_rebalance_matches_equal_weight_reference():
    zero = CostRates(0.0, 0.0, 0.0)
    first = execute_orders(
        orders={"005930": 5.0, "000660": 5.0},
        holdings={},
        cash=1000.0,
        execution_closes={"005930": 100.0, "000660": 100.0},
        previous_closes={"005930": 100.0, "000660": 100.0},
        halted_tickers=frozenset(),
        rates=zero,
        limit_move_abs_return=0.295,
    )
    second = execute_orders(
        orders={"005930": 500.0 / 110.0 - 5.0, "000660": 500.0 / 90.0 - 5.0},
        holdings=first.holdings,
        cash=first.cash,
        execution_closes={"005930": 110.0, "000660": 90.0},
        previous_closes={"005930": 100.0, "000660": 100.0},
        halted_tickers=frozenset(),
        rates=zero,
        limit_move_abs_return=0.295,
    )
    actual = portfolio_nav(
        holdings=second.holdings,
        closes={"005930": 120.0, "000660": 80.0},
        cash=second.cash,
    )
    reference = 1000.0 * (0.5 * 120.0 / 110.0 + 0.5 * 80.0 / 90.0)
    assert actual == pytest.approx(reference)
    assert second.cash == pytest.approx(0.0)
