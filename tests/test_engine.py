"""M4와 M5가 함께 쓰는 구간 함수의 NAV 경계와 이벤트 우선순위."""

import pandas as pd
import pytest

from src.backtest.accounting import CostRates
from src.backtest.engine import ExecutionEngine


def market(prices, *, rates=None, **kwargs):
    days = pd.bdate_range("2025-01-06", periods=len(prices))
    closes = pd.DataFrame({"005930": prices}, index=days, dtype=float)
    bm = pd.Series([100.0] * len(prices), index=days)
    return ExecutionEngine(closes, bm, rates or CostRates(0, 0, 0), record_ledger=True, **kwargs)


def day(i):
    return pd.bdate_range("2025-01-06", periods=5)[i].strftime("%Y-%m-%d")


def test_period_boundary_and_observation_use_next_signal_close():
    engine = market([100, 110, 120, 130])
    result = engine.run_period(engine.initial_state(), {"005930": 0.5}, day(0), day(2))
    expected_end = 0.5 + 0.5 * 130 / 110
    assert result.nav_start == 1
    assert result.nav_end == pytest.approx(expected_end)
    assert result.r_net == pytest.approx(expected_end - 1)
    assert result.r_bm == 0
    assert result.turnover == pytest.approx(0.25)
    assert result.weights_obs["005930"] == pytest.approx((0.5 * 120 / 110) / (0.5 + 0.5 * 120 / 110))
    assert result.cash_obs + sum(result.weights_obs.values()) == pytest.approx(1)
    assert engine.ledger[0]["nav_before"] == pytest.approx(1)
    assert engine.ledger[0]["nav_after"] == pytest.approx(1)


def test_all_buys_scale_together_when_cost_exceeds_cash():
    engine = market([100, 100, 100, 100], rates=CostRates(0, 0.01, 0))
    result = engine.run_period(engine.initial_state(), {"005930": 1.0}, day(0), day(2))
    assert result.state.cash == pytest.approx(0)
    assert result.nav_end == pytest.approx(1 / 1.01)
    assert engine.ledger[0]["cost"] == pytest.approx(0.01 / 1.01)


def test_event_liquidation_precedes_first_day_target_and_cost_is_once():
    engine = market([100, 100, 100, 100], event_liquidations={day(1): frozenset({"005930"})})
    result = engine.run_period(engine.initial_state(), {"005930": 0.5}, day(0), day(2))
    assert result.state.holdings == {}
    assert result.nav_end == pytest.approx(1)
    assert engine.ledger[0]["fills"] == ()


def test_mid_period_event_retries_sell_after_lower_limit():
    engine = market([100, 100, 70.5, 75, 75], event_liquidations={day(2): frozenset({"005930"})})
    result = engine.run_period(engine.initial_state(), {"005930": 0.5}, day(0), day(3))
    assert engine.ledger[1]["rejected"]["005930"] == "near_lower_limit"
    assert result.state.holdings == {}
    assert result.state.pending_liquidations == frozenset()
    assert result.nav_end == pytest.approx(0.5 + 0.5 * 75 / 100)


def test_next_period_starts_from_previous_pretrade_nav():
    engine = market([100, 110, 120, 130, 140])
    first = engine.run_period(engine.initial_state(), {"005930": 0.5}, day(0), day(2))
    second = engine.run_period(first.state, {}, day(2), day(3))
    assert second.nav_start == pytest.approx(first.nav_end)
    assert second.state.holdings == {}
    assert second.nav_end == pytest.approx(first.nav_end)


def test_cooldown_suppresses_buy_and_records_reason():
    engine = market([100, 100, 100, 100], cooldowns={day(1): frozenset({"005930"})})
    result = engine.run_period(engine.initial_state(), {"005930": 0.5}, day(0), day(2))
    assert result.nav_end == pytest.approx(1)
    assert engine.ledger[0]["rejected"]["005930"] == "cooldown_or_delisted"


def test_delisting_uses_last_price_and_sell_cost():
    engine = market([100, 100, 100, 100], rates=CostRates(0.01, 0, 0),
                    delistings={"005930": (day(2), 80.0)})
    result = engine.run_period(engine.initial_state(), {"005930": 0.5}, day(0), day(2))
    assert result.state.holdings == {}
    assert result.nav_end == pytest.approx(0.5 + 0.5 * 0.8 * 0.99)
    assert engine.ledger[1]["fills"][0].price == 80.0


def test_identical_rl_and_buffer_targets_share_execution_and_costs():
    kwargs = {"rates": CostRates(0.002, 0.00015, 0.001),
              "event_liquidations": {day(2): frozenset({"005930"})}}
    rl_engine = market([100, 100, 102, 103], **kwargs)
    buffer_engine = market([100, 100, 102, 103], **kwargs)
    rl = rl_engine.run_period(rl_engine.initial_state(), {"005930": 0.4}, day(0), day(2))
    buffer = buffer_engine.run_period(buffer_engine.initial_state(), {"005930": 0.4}, day(0), day(2))
    assert rl.nav_end == pytest.approx(buffer.nav_end)
    assert rl.turnover == pytest.approx(buffer.turnover)
    assert [(row["date"], row["cost"], row["fills"]) for row in rl_engine.ledger] == [
        (row["date"], row["cost"], row["fills"]) for row in buffer_engine.ledger
    ]


def test_halt_rejects_initial_buy_and_keeps_cash():
    days = pd.bdate_range("2025-01-06", periods=4)
    halted = pd.DataFrame({"005930": [False, True, False, False]}, index=days)
    engine = market([100, 100, 100, 100], halted=halted)
    result = engine.run_period(engine.initial_state(), {"005930": 0.4}, day(0), day(2))
    assert result.nav_end == pytest.approx(1)
    assert engine.ledger[0]["rejected"]["005930"] == "halted"
