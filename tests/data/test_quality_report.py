import json

import pandas as pd
import pytest

from src.data.quality_report import (SPLIT_CASES, build, check_calendar, check_delisted,
                                     check_reused_delisted, check_splits, check_ticker_format,
                                     check_universe_size, dividend_check, listing_vs_first_price)

DAYS = pd.bdate_range("2010-01-04", "2010-03-31")


def _index(days=DAYS, code="KOSPI200", close=100.0):
    rows = pd.DataFrame({"date": days, "index_code": code, "close": close})
    other = pd.DataFrame({"date": days, "index_code": "KOSDAQ150", "close": -1.0})   # 섞여 있어도 봐서는 안 된다
    return pd.concat([rows, other], ignore_index=True)


def test_calendar_counts_only_kospi200():
    r = check_calendar(_index())
    assert r["n_days"] == len(DAYS) and r["duplicates"] == 0
    assert r["close_nonpositive_or_nan"] == 0            # 코스닥 150의 -1은 세지 않는다
    assert r["first"] == "2010-01-04" and r["last"] == str(DAYS[-1].date())


def test_calendar_flags_duplicates_and_bad_close():
    idx = _index()
    idx = pd.concat([idx, idx.iloc[[0]], pd.DataFrame({"date": [DAYS[1]], "index_code": ["KOSPI200"], "close": [0.0]})])
    r = check_calendar(idx)
    assert r["duplicates"] == 2 and r["close_nonpositive_or_nan"] == 1


def test_delisted_counts_rows_that_survive_in_prices():
    tickers = pd.DataFrame({"ticker": ["000001", "000002", "000003"],
                            "security_type": ["common", "common", "preferred"],
                            "delisting_date": pd.to_datetime(["2015-03-02", "2009-05-01", "2020-06-01"]),
                            "listing_date": pd.to_datetime(["2000-01-04"] * 3)})
    prices = pd.DataFrame({"ticker": ["000001", "000003"], "date": pd.to_datetime(["2015-03-01", "2020-05-29"])})
    delisted = pd.DataFrame({"ticker": ["000001"], "delist_date": pd.to_datetime(["2015-03-02"]), "last_price": [1000.0]})
    r = check_delisted(tickers, prices, delisted)
    assert r["delisted_since_2010_total"] == 2            # 2009년 폐지는 빠진다
    assert r["in_prices"] == 2 and r["common_delisted"] == 1
    assert r["common_delisted_with_last_price"] == 1
    assert r["by_type"]["preferred"] == {"n": 1, "in_prices": 1}
    # delisted_reused가 없으면 "전부 확인"으로 보고하지 않는다 (C 리뷰 #45)
    assert r["complete"] is False and r["reused"]["checked"] is False


def test_ticker_format_catches_bad_codes():
    good = pd.DataFrame({"ticker": ["005930", "00088K"]})
    bad = pd.DataFrame({"ticker": ["5930", None]})
    assert check_ticker_format({"a": good, "b": bad}) == {"a": True, "b": False}


def test_splits_report_continuous_adjusted_close():
    ticker, _, day, _ = SPLIT_CASES[0]
    days = pd.bdate_range("2018-05-01", "2018-05-10")
    raw = [50000.0, 50500.0, 1010.0, 1020.0, 1015.0, 1030.0, 1025.0, 1040.0]      # 분할로 원종가가 끊긴다
    adj = [1000.0, 1010.0, 1010.0, 1020.0, 1015.0, 1030.0, 1025.0, 1040.0]        # 수정종가는 이어진다
    prices = pd.DataFrame({"ticker": ticker, "date": days, "close_raw": raw, "close": adj})
    result, rows = check_splits(prices)
    me = [r for r in result if r["ticker"] == ticker][0]
    assert me["status"] == "연속" and me["date"] == day
    assert me["max_abs_daily_change_raw"] > 0.9 and me["max_abs_daily_change_adjusted"] < 0.31
    assert set(rows["ticker"]) == {ticker}
    missing = [r for r in result if r["ticker"] != ticker]
    assert all(r["status"] == "시세 없음" for r in missing)


def test_universe_size_marks_admin_step():
    universe = pd.DataFrame({"date": pd.to_datetime(["2011-01-07"] * 3 + ["2011-01-14"] * 2),
                             "ticker": ["000001", "000002", "000003", "000001", "000002"],
                             "in_universe": [True, True, False, True, True]})
    missing_admin = pd.DataFrame({"ticker": ["000001"], "is_admin": pd.Series([pd.NA], dtype="boolean")})
    nt, r = check_universe_size(universe, missing_admin)
    assert list(nt) == [2, 2] and r["n_signal_days"] == 2 and r["max"] == 2
    assert r["admin_step_applied"] is False and "DEC-24" in r["universe_note"]
    known_admin = pd.DataFrame({"ticker": ["000001"], "is_admin": pd.Series([False], dtype="boolean")})
    assert check_universe_size(universe, known_admin)[1]["admin_step_applied"] is True


def test_dividend_check_sees_year_end_concentration():
    days = pd.bdate_range("2015-01-01", "2015-12-31")
    prices = pd.DataFrame({"date": days})
    events = pd.DataFrame({"date": pd.to_datetime(["2015-12-28", "2015-12-29", "2015-03-02"])})
    r = dividend_check(prices, events)
    assert r["events_dec24_30"] == 2 and r["ratio_year_end_to_all"] > 1


def test_listing_gap_counts_only_newly_listed():
    prices = pd.DataFrame({"ticker": ["000001", "000001", "000002"],
                           "date": pd.to_datetime(["2010-01-04", "2010-01-05", "2015-06-01"])})
    tickers = pd.DataFrame({"ticker": ["000001", "000002"],
                            "listing_date": pd.to_datetime(["1999-01-04", "2015-05-01"])})
    r = listing_vs_first_price(prices, tickers, pd.Timestamp("2010-01-04"))
    assert r["compared"] == 1 and r["first_price_more_than_7d_after_listing"] == 1
    assert r["first_price_before_listing"] == 0 and r["listing_date_missing"] == 0


def test_build_rejects_bad_run_id_and_existing_folder(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        build("nope", out_root=tmp_path)
    (tmp_path / "20261110_A_dq_v1").mkdir()
    with pytest.raises(FileExistsError):           # R-11: 덮어쓰지 않는다
        build("20261110_A_dq_v1", out_root=tmp_path)


def test_reused_ticker_old_company_is_checked_separately():
    """재사용 코드의 구회사 폐지는 tickers가 아니라 delisted_reused에만 있다 (DEC-28, C 리뷰 #45)."""
    prices = pd.DataFrame({
        "ticker": ["036220"] * 4,
        "date": pd.to_datetime(["2016-05-03", "2016-05-04", "2024-03-13", "2024-03-14"]),  # 구회사 2일 + 신회사 2일
        "close_raw": [3400.0, 3500.0, 20000.0, 20500.0]})
    reused = pd.DataFrame({"ticker": ["036220"], "delist_date": pd.to_datetime(["2016-05-09"]),
                           "last_price": [3500.0]})
    r = check_reused_delisted(prices, reused)
    assert r["checked"] is True and r["n"] == 1
    row = r["detail"][0]
    assert row["rows_before_delist"] == 2 and row["rows_after_delist"] == 2   # 구회사·신회사 구간이 나뉜다
    assert row["last_close_raw_before_delist"] == 3500.0 and row["last_price_matches_close_raw"] is True
    assert r["all_old_company_rows_present"] and r["all_last_price_matches"]
    assert check_reused_delisted(prices, None)["checked"] is False            # 입력이 없으면 미점검
    assert check_reused_delisted(prices, reused.iloc[:0])["checked"] is False
