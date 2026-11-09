"""M7 사후 성과의 확정 시점, 원본 신호, 실제 NAV 재생."""

from datetime import date

import pandas as pd
import pytest

from src.backtest.accounting import CostRates
from src.backtest.engine import ExecutionEngine
from src.backtest.forward_outcomes import evaluate_forward_record
from src.common.guards import DataAccessPolicy, GuardViolation


POLICY = DataAccessPolicy(date(2025, 12, 31), date(2026, 1, 1), date(2026, 10, 31), date(2026, 12, 11))


def fixture(start="2026-11-06", *, has_positions=1):
    days = pd.bdate_range(start, periods=23)
    d = [day.strftime("%Y-%m-%d") for day in days]
    prices = pd.DataFrame({"000001": [100.0] * len(days)}, index=days)
    bm = pd.Series([100.0] * 21 + [110.0] * 2, index=days)
    engine = ExecutionEngine(prices, bm, CostRates(0, 0, 0))
    signals = pd.DataFrame({"date": [days[0]] * 20,
                            "ticker": [f"{i:06d}" for i in range(1, 21)],
                            "rank": list(range(1, 21)),
                            "crash_flag": [1, 1] + [0] * 18})
    labels = signals[["date", "ticker"]].assign(h=20, er=[-0.2, -0.2] + [0.1] * 18)
    record = {"signal_date": days[0], "label_ready_date": days[21],
              "has_positions": has_positions, "commit_hash": "original-hash", "signal_source": "baseline"}
    log = pd.DataFrame([record])
    return d, engine, signals, labels, record, log


def test_forward_outcome_waits_for_close_and_preserves_signal_only_nulls():
    d, engine, signals, labels, record, log = fixture(has_positions=0)
    kwargs = dict(engine=engine, record=record, signals=signals, labels=labels,
                  forward_log=log, recorded_targets={}, policy=POLICY)
    with pytest.raises(ValueError, match="장 마감 전"):
        evaluate_forward_record(**kwargs, run_at=pd.Timestamp(f"{d[21]} 15:29:59", tz="Asia/Seoul"))
    result = evaluate_forward_record(**kwargs, run_at=pd.Timestamp(f"{d[21]} 16:00:00", tz="Asia/Seoul"))
    assert result.mean_er_top20 == pytest.approx(0.07)
    assert result.mean_er_flagged == pytest.approx(-0.2)
    assert result.portfolio_return is None and result.bm_return is None
    assert record["commit_hash"] == "original-hash"
    altered = {**record, "commit_hash": "changed-hash"}
    with pytest.raises(ValueError, match="원본 로그"):
        evaluate_forward_record(**{**kwargs, "record": altered},
                                run_at=pd.Timestamp(f"{d[21]} 16:00:00", tz="Asia/Seoul"))


def test_forward_outcome_replays_recorded_positions_and_rejects_missing_week():
    d, engine, signals, labels, record, log = fixture()
    run_at = pd.Timestamp(f"{d[21]} 16:00:00", tz="Asia/Seoul")
    kwargs = dict(engine=engine, record=record, signals=signals, labels=labels,
                  forward_log=log, policy=POLICY, run_at=run_at)
    result = evaluate_forward_record(**kwargs, recorded_targets={d[0]: {"000001": 0.5}})
    assert result.portfolio_return == pytest.approx(0.0)
    assert result.bm_return == pytest.approx(0.1)
    with pytest.raises(ValueError, match="날짜가 다릅니다"):
        evaluate_forward_record(**kwargs, recorded_targets={})


def test_end_date_rebalance_cost_is_excluded_from_forward_return():
    d, _, signals, labels, record, log = fixture()
    days = pd.to_datetime(d)
    prices = pd.DataFrame({"000001": [100.0] * len(days)}, index=days)
    bm = pd.Series([100.0] * len(days), index=days)
    engine = ExecutionEngine(prices, bm, CostRates(0, 0.01, 0))
    late = {"signal_date": days[20], "label_ready_date": days[21],
            "has_positions": 1, "commit_hash": "later-hash", "signal_source": "baseline"}
    log = pd.concat([log, pd.DataFrame([late])], ignore_index=True)
    outcome = evaluate_forward_record(
        engine=engine, record=record, signals=signals, labels=labels,
        forward_log=log, recorded_targets={d[0]: {"000001": 0.5}, d[20]: {"000001": 1.0}},
        policy=POLICY, run_at=pd.Timestamp(f"{d[21]} 16:00:00", tz="Asia/Seoul"),
    )
    assert outcome.portfolio_return == pytest.approx(-0.005)


def test_forward_outcome_respects_holdout_open_guard():
    d, engine, signals, labels, record, log = fixture(start="2026-10-01", has_positions=0)
    with pytest.raises(GuardViolation):
        evaluate_forward_record(engine=engine, record=record, signals=signals, labels=labels,
                                forward_log=log, recorded_targets={}, policy=POLICY,
                                run_at=pd.Timestamp(f"{d[21]} 16:00:00", tz="Asia/Seoul"))
