"""PM 시각 계약 후보를 원본 보관→정제→시각 선택까지 검증한다."""
from datetime import date, datetime

import pandas as pd
import pytest

from conftest import item, page
from src.data.collect_disclosures import parser
from src.data.dart_client import KST
from src.data.disclosure_archive import RawArchive
from src.data.disclosures import (EXTENDED_COLUMNS, build_disclosures, collect_run,
                                  filter_disclosures_before, plan_run)


@pytest.fixture
def result(tmp_path, client_factory):
    today = date(2026, 11, 2)
    archive = RawArchive.create(tmp_path / "raw", "20100105_PM_contract_test", today)
    client, _, _ = client_factory([page([item(n, day="20100104") for n in (1, 2, 3)])])
    plan_run(date(2010, 1, 1), date(2010, 3, 31), types=["B"], client=client, archive=archive, today=today)
    assert collect_run(client=client, archive=archive, today=today).status == "complete"
    return build_disclosures([archive.root], today=today)


def timestamp_column(values):
    return pd.to_datetime(values, utc=True).tz_convert(KST).astype(pd.DatetimeTZDtype("ns", KST))


def test_default_diagnostic_separates_actual_and_available_time(result):
    assert result.schema_mode == "extended"
    assert list(result.df.columns) == EXTENDED_COLUMNS
    assert result.df.rcept_dt.isna().all()
    assert result.df.available_at.iloc[0] == pd.Timestamp("2010-01-04T23:59:59+09:00")
    assert result.df.ticker.isna().all()
    assert parser().parse_args(["build"]).schema_mode == "extended"


@pytest.mark.parametrize("cutoff,expected", [
    ("2010-01-04T15:29:59+09:00", 0),
    ("2010-01-04T15:30:00+09:00", 0),
    ("2010-01-04T15:30:01+09:00", 0),
    ("2010-01-04T23:59:59+09:00", 0),
    ("2010-01-05T15:30:00+09:00", 3),
])
def test_archived_date_only_rows_wait_for_next_signal(result, cutoff, expected):
    selected = filter_disclosures_before(result.df, cutoff_at=pd.Timestamp(cutoff))
    assert len(selected) == expected
    assert selected.rcept_dt.isna().all()
    assert selected.attrs["diagnostic_only"] and not selected.attrs["public_publish_enabled"]


@pytest.mark.parametrize("cutoff,expected_numbers", [
    ("2010-01-04T15:29:59+09:00", []),
    ("2010-01-04T15:30:00+09:00", [1]),
    ("2010-01-04T15:30:00.000000001+09:00", [1, 2]),
    ("2010-01-04T15:30:01+09:00", [1, 2]),
])
def test_strict_cutoff_keeps_time_precision(result, cutoff, expected_numbers):
    frame = result.df.copy()
    frame["available_at"] = timestamp_column(["2010-01-04T15:29:59+09:00", "2010-01-04T15:30:00+09:00", "2010-01-04T15:30:01+09:00"])
    frame["rcept_dt"] = frame.available_at
    before = frame.copy(deep=True)
    selected = filter_disclosures_before(frame, cutoff_at=pd.Timestamp(cutoff))
    assert selected.rcept_no.tolist() == [f"20100104{number:06d}" for number in expected_numbers]
    pd.testing.assert_frame_equal(frame, before)


def test_utc_input_has_identical_cutoff_and_kst_output(result):
    frame = result.df.copy()
    frame["available_at"] = timestamp_column(["2010-01-04T15:29:59+09:00", "2010-01-04T15:30:00+09:00", "2010-01-04T15:30:01+09:00"])
    expected = filter_disclosures_before(frame, cutoff_at=pd.Timestamp("2010-01-04T15:30:00+09:00"))
    utc = frame.copy()
    for name in ("rcept_dt", "available_at"):
        utc[name] = utc[name].dt.tz_convert("UTC")
    selected = filter_disclosures_before(utc, cutoff_at=pd.Timestamp("2010-01-04T06:30:00+00:00"))
    pd.testing.assert_frame_equal(selected, expected)


@pytest.mark.parametrize("cutoff", [date(2010, 1, 5), datetime(2010, 1, 5, 15, 30), "2010-01-05T15:30:00+09:00", pd.NaT])
def test_cutoff_must_be_aware_datetime(result, cutoff):
    with pytest.raises(ValueError, match="컷오프"):
        filter_disclosures_before(result.df, cutoff_at=cutoff)


@pytest.mark.parametrize("column", ["rcept_dt", "available_at"])
def test_time_fields_cannot_be_naive(result, column):
    frame = result.df.copy()
    frame[column] = frame[column].dt.tz_localize(None)
    with pytest.raises(ValueError, match="시간대"):
        filter_disclosures_before(frame, cutoff_at=pd.Timestamp("2010-01-05T15:30:00+09:00"))


def test_missing_availability_is_not_filled(result):
    frame = result.df.copy()
    frame.loc[0, "available_at"] = pd.NaT
    with pytest.raises(ValueError, match="결측"):
        filter_disclosures_before(frame, cutoff_at=pd.Timestamp("2010-01-05T15:30:00+09:00"))


def test_availability_cannot_precede_known_actual_time(result):
    frame = result.df.copy()
    frame.loc[0, "rcept_dt"] = pd.Timestamp("2010-01-05T00:00:00+09:00")
    with pytest.raises(ValueError, match="빠를 수"):
        filter_disclosures_before(frame, cutoff_at=pd.Timestamp("2010-01-05T15:30:00+09:00"))


def test_legacy_frame_without_separate_availability_is_rejected(result):
    frame = result.df.drop(columns="available_at")
    with pytest.raises(ValueError, match="분리"):
        filter_disclosures_before(frame, cutoff_at=pd.Timestamp("2010-01-05T15:30:00+09:00"))


def test_empty_diagnostic_input_preserves_schema(result):
    frame = result.df.iloc[:0].copy()
    selected = filter_disclosures_before(frame, cutoff_at=pd.Timestamp("2010-01-05T15:30:00+09:00"))
    assert selected.empty and list(selected.columns) == EXTENDED_COLUMNS
