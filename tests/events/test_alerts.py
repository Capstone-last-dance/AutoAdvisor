"""M9 종목 연결·경보·쿨다운·청산 일정 테스트. 합성 자료만 쓴다."""

import pandas as pd
import pytest

from src.common import io
from src.events.alerts import AlertBook, build_alerts, link_tickers, write_alert_files

KST = "Asia/Seoul"
DAYS = pd.bdate_range("2026-01-05", periods=60)
COOLDOWN = 20

EVENTS = {
    "cooldown_trading_days": COOLDOWN,
    "title_filters": {"strip_prefixes": ["[첨부추가]"], "exclude_prefixes": ["[기재정정]"], "exclude_terms": ["철회"]},
    "triggers": {"T2": {"keywords": ["상장폐지사유발생"]}, "T5": {"keywords": ["부도발생"]},
                 "T6": {"keywords": ["유상증자결정"], "min_amount_to_mcap": 0.10}},
}


def stamp(day):
    return pd.Timestamp(f"{day} 23:59:59", tz=KST)


def disclosures(rows):
    """rows: (rcept_no, day, corp_code, stock_code, title)"""
    frame = pd.DataFrame(rows, columns=["rcept_no", "rcept_date", "corp_code", "stock_code", "report_nm"])
    frame["rcept_date"] = pd.to_datetime(frame["rcept_date"])
    frame["available_at"] = [stamp(day) for day in frame["rcept_date"].dt.strftime("%Y-%m-%d")]
    frame["available_at"] = frame["available_at"].astype(pd.DatetimeTZDtype("ns", KST))
    frame["rcept_dt"] = pd.Series([pd.NaT] * len(frame), dtype=pd.DatetimeTZDtype("ns", KST))
    frame["corp_name"] = "회사"
    frame["report_nm_norm"] = frame["report_nm"]
    return frame


TICKERS = pd.DataFrame({"ticker": ["000001", "000002", "000003"],
                        "listing_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2026-02-01"]),
                        "delisting_date": pd.to_datetime([None, "2026-01-15", None])})


def alerts_frame(rows):
    """rows: (ticker, available_at 문자열)"""
    available = pd.DatetimeIndex([pd.Timestamp(a, tz=KST) for _, a in rows], tz=KST).as_unit("ns")
    return pd.DataFrame({"date": available.tz_localize(None).normalize(), "ticker": [t for t, _ in rows],
                         "trigger_type": "T2", "rcept_no": [str(i) for i in range(len(rows))], "report_nm": "x",
                         "rcept_dt": pd.Series([pd.NaT] * len(rows), dtype=pd.DatetimeTZDtype("ns", KST)),
                         "available_at": available})


def book(rows):
    return AlertBook(alerts_frame(rows), DAYS, COOLDOWN)


class TestLinkTickers:
    def status(self, rows):
        return list(link_tickers(disclosures(rows), TICKERS)["link_status"])

    def test_listed_on_receipt_date(self):
        frame = link_tickers(disclosures([("1", "2026-01-06", "C1", "000001", "t")]), TICKERS)
        assert (frame.loc[0, "link_status"], frame.loc[0, "ticker"]) == ("linked", "000001")

    def test_no_code(self):
        frame = link_tickers(disclosures([("1", "2026-01-06", "C1", "", "t")]), TICKERS)
        assert frame.loc[0, "link_status"] == "no_code" and frame.loc[0, "ticker"] is None

    def test_not_in_tickers(self):
        assert self.status([("1", "2026-01-06", "C1", "999999", "t")]) == ["not_in_tickers"]

    def test_before_listing_or_after_delisting(self):
        assert self.status([("1", "2026-01-06", "C3", "000003", "t")]) == ["outside_listing"]
        assert self.status([("1", "2026-01-16", "C2", "000002", "t")]) == ["outside_listing"]

    def test_delisting_day_itself_is_listed(self):
        assert self.status([("1", "2026-01-15", "C2", "000002", "t")]) == ["linked"]

    def test_code_with_two_companies_is_ambiguous(self):
        rows = [("1", "2026-01-06", "C1", "000001", "t"), ("2", "2026-01-07", "C9", "000001", "t")]
        assert self.status(rows) == ["ambiguous", "ambiguous"]

    def test_company_with_two_codes_is_ambiguous(self):
        rows = [("1", "2026-01-06", "C1", "000001", "t"), ("2", "2026-01-07", "C1", "000002", "t")]
        assert self.status(rows) == ["ambiguous", "ambiguous"]

    def test_input_is_not_changed(self):
        frame = disclosures([("1", "2026-01-06", "C1", "000001", "t")])
        before = frame.copy()
        link_tickers(frame, TICKERS)
        pd.testing.assert_frame_equal(frame, before)


class TestBuildAlerts:
    def build(self, rows):
        return build_alerts(disclosures(rows), TICKERS, EVENTS, DAYS)

    def test_alert_matches_shared_schema(self):
        result = self.build([("20260106000001", "2026-01-06", "C1", "000001", "상장폐지사유발생")])
        assert len(result.alerts) == 1
        io.validate("alerts", result.alerts)
        row = result.alerts.iloc[0]
        assert (row["ticker"], row["trigger_type"], row["date"]) == ("000001", "T2", pd.Timestamp("2026-01-06"))

    def test_unlinked_candidate_is_kept_only_in_decisions(self):
        result = self.build([("1", "2026-01-06", "C1", "", "상장폐지사유발생")])
        assert result.alerts.empty and list(result.decisions["final"]) == ["unlinked"]

    def test_document_dependent_candidate_is_not_an_alert(self):
        result = self.build([("1", "2026-01-06", "C1", "000001", "유상증자결정")])
        assert result.alerts.empty and list(result.decisions["final"]) == ["needs_document"]

    def test_followup_inside_cooldown_keeps_first_event(self):
        rows = [("1", "2026-01-06", "C1", "000001", "상장폐지사유발생"), ("2", "2026-01-12", "C1", "000001", "상장폐지사유발생")]
        result = self.build(rows)
        assert list(result.decisions["final"]) == ["alert", "same_event"]
        assert list(result.alerts["rcept_no"]) == ["1"]

    def test_new_event_after_cooldown_ends(self):
        day_after = DAYS[DAYS.get_loc(pd.Timestamp("2026-01-07")) + COOLDOWN].strftime("%Y-%m-%d")
        rows = [("1", "2026-01-06", "C1", "000001", "상장폐지사유발생"), ("2", day_after, "C1", "000001", "상장폐지사유발생")]
        assert list(self.build(rows).decisions["final"]) == ["alert", "alert"]

    def test_other_type_is_a_separate_event(self):
        rows = [("1", "2026-01-06", "C1", "000001", "상장폐지사유발생"), ("2", "2026-01-07", "C1", "000001", "부도발생")]
        assert list(self.build(rows).alerts["trigger_type"]) == ["T2", "T5"]

    def test_amendment_never_alerts(self):
        result = self.build([("1", "2026-01-06", "C1", "000001", "[기재정정]상장폐지사유발생")])
        assert result.alerts.empty and list(result.decisions["final"]) == ["excluded"]

    def test_no_candidates(self):
        result = self.build([("1", "2026-01-06", "C1", "000001", "분기보고서")])
        assert result.alerts.empty and result.decisions.empty


class TestCooldown:
    def test_same_day_disclosure_is_not_known_at_signal(self):
        assert book([("000001", "2026-01-09 23:59:59")]).cooldown_at("2026-01-09") == set()

    def test_known_after_receipt_date(self):
        assert book([("000001", "2026-01-08 23:59:59")]).cooldown_at("2026-01-09") == {"000001"}

    def test_cutoff_is_strict(self):
        exact = book([("000001", "2026-01-09 15:30:00")]).cooldown_at("2026-01-09")
        earlier = book([("000001", "2026-01-09 15:29:59")]).cooldown_at("2026-01-09")
        assert exact == set() and earlier == {"000001"}

    def test_weekend_disclosure_starts_on_monday(self):
        saturday = book([("000001", "2026-01-10 23:59:59")])
        assert saturday.cooldown_at("2026-01-09") == set()
        assert saturday.cooldown_at("2026-01-12") == {"000001"}

    def test_last_and_first_free_day(self):
        first_exec = DAYS.get_loc(pd.Timestamp("2026-01-07"))
        last_signal = DAYS[first_exec + COOLDOWN - 2].strftime("%Y-%m-%d")
        free_signal = DAYS[first_exec + COOLDOWN - 1].strftime("%Y-%m-%d")
        one = book([("000001", "2026-01-06 23:59:59")])
        assert one.cooldown_at(last_signal) == {"000001"}
        assert one.cooldown_at(free_signal) == set()

    def test_future_alert_does_not_change_past_answer(self):
        base = [("000001", "2026-01-06 23:59:59")]
        future = base + [("000002", "2026-01-20 23:59:59")]
        for signal in DAYS[:8].strftime("%Y-%m-%d"):
            assert book(base).cooldown_at(signal) == book(future).cooldown_at(signal)

    def test_empty_book(self):
        empty = AlertBook(alerts_frame([]).iloc[0:0], DAYS, COOLDOWN)
        assert empty.cooldown_at("2026-01-09") == set()
        assert empty.engine_alerts("2026-01-05", "2026-01-09") == ({}, {})

    def test_calendar_end_is_an_error(self):
        with pytest.raises(ValueError, match="다음 거래일"):
            book([]).cooldown_at(DAYS[-1].strftime("%Y-%m-%d"))


class TestEngineAlerts:
    def test_liquidation_on_next_trading_day_and_cooldown_from_there(self):
        cooldowns, liquidations = book([("000001", "2026-01-06 23:59:59")]).engine_alerts("2026-01-05", "2026-01-09")
        assert liquidations == {"2026-01-07": frozenset({"000001"})}
        assert sorted(cooldowns) == ["2026-01-07", "2026-01-08", "2026-01-09"]
        assert all(isinstance(v, frozenset) for v in cooldowns.values())

    def test_range_starts_after_first_signal(self):
        cooldowns, liquidations = book([("000001", "2026-01-06 23:59:59")]).engine_alerts("2026-01-08", "2026-01-13")
        assert liquidations == {}
        assert sorted(cooldowns) == ["2026-01-09", "2026-01-12", "2026-01-13"]

    def test_alert_not_known_by_t_is_absent(self):
        assert book([("000001", "2026-01-09 23:59:59")]).engine_alerts("2026-01-05", "2026-01-09") == ({}, {})

    def test_two_tickers_same_day(self):
        _, liquidations = book([("000001", "2026-01-06 23:59:59"), ("000002", "2026-01-06 23:59:59")]).engine_alerts(
            "2026-01-05", "2026-01-07")
        assert liquidations == {"2026-01-07": frozenset({"000001", "000002"})}

    def test_works_with_engine_input_shape(self):
        from src.backtest.engine import ExecutionEngine
        import inspect

        parameters = inspect.signature(ExecutionEngine.__init__).parameters
        assert {"event_liquidations", "cooldowns"} <= set(parameters)


def test_write_alert_files_by_date(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "ROOT", tmp_path)
    rows = [("1", "2026-01-06", "C1", "000001", "상장폐지사유발생"), ("2", "2026-01-07", "C2", "000002", "상장폐지사유발생")]
    result = build_alerts(disclosures(rows), TICKERS, EVENTS, DAYS)
    assert write_alert_files(result.alerts) == ["2026-01-06", "2026-01-07"]
    back = io.read_table("alerts", date="2026-01-06")
    assert list(back["rcept_no"]) == ["1"]
