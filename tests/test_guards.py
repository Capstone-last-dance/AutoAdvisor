"""C-02: 홀드아웃이 실수로 열리지 않는지 경계 날짜로 검증한다."""

from datetime import date, datetime

import pytest

from src.common.guards import (
    DataAccessPolicy,
    GuardViolation,
    assert_outcome_dates_allowed,
    assert_training_dates_allowed,
    purge_signal_dates,
)

POLICY = DataAccessPolicy(
    latest_training_signal_date=date(2025, 12, 31),
    holdout_start=date(2026, 1, 1),
    holdout_end=date(2026, 10, 31),
    holdout_open_date=date(2026, 12, 11),
)


def test_training_rejects_holdout_and_later_signal_dates():
    assert_training_dates_allowed(["2025-12-31"], policy=POLICY)
    with pytest.raises(GuardViolation, match="2026-01-01"):
        assert_training_dates_allowed(["2025-12-31", "2026-01-01"], policy=POLICY)
    with pytest.raises(GuardViolation, match="2026-11-06"):
        assert_training_dates_allowed(["2026-11-06"], policy=POLICY)


def test_holdout_requires_both_explicit_flag_and_open_date():
    holdout = ["2026-01-01", "2026-10-31"]
    with pytest.raises(GuardViolation):
        assert_outcome_dates_allowed(
            holdout, policy=POLICY, allow_holdout=False, run_date=date(2026, 12, 11)
        )
    with pytest.raises(GuardViolation):
        assert_outcome_dates_allowed(
            holdout, policy=POLICY, allow_holdout=True, run_date=date(2026, 12, 10)
        )
    assert_outcome_dates_allowed(
        holdout, policy=POLICY, allow_holdout=True, run_date=date(2026, 12, 11)
    )


def test_non_holdout_dates_do_not_need_open_flag():
    assert_outcome_dates_allowed(
        ["2025-12-31", "2026-11-06"],
        policy=POLICY,
        allow_holdout=False,
        run_date=date(2026, 11, 2),
    )


def test_timestamps_are_not_silently_treated_as_signal_dates():
    with pytest.raises(TypeError):
        assert_training_dates_allowed([datetime(2025, 12, 31, 15, 30)], policy=POLICY)


def test_purge_excludes_labels_reaching_next_segment():
    days = [
        date(2025, 12, 24),
        date(2025, 12, 26),
        date(2025, 12, 29),
        date(2025, 12, 30),
        date(2025, 12, 31),
        date(2026, 1, 2),
        date(2026, 1, 5),
    ]
    signals = [date(2025, 12, 29), date(2025, 12, 30), date(2025, 12, 31)]
    assert purge_signal_dates(
        signals,
        trading_days=days,
        h=1,
        next_segment_first_signal=date(2026, 1, 2),
    ) == (date(2025, 12, 29),)
    assert (
        purge_signal_dates(
            signals,
            trading_days=days,
            h=2,
            next_segment_first_signal=date(2026, 1, 2),
        )
        == ()
    )

    # 명목 종료일은 앞 구간이어도 거래 재개 후 확정된 라벨은 경계를 넘는다.
    assert purge_signal_dates(
        signals,
        trading_days=days,
        h=1,
        next_segment_first_signal=date(2026, 1, 2),
        actual_end_dates={date(2025, 12, 29): date(2026, 1, 2)},
    ) == ()
