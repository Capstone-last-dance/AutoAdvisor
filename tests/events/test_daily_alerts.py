"""M9 일간 경보·날짜별 파일·generated_at 테스트 (DEC-32, DEC-33). 합성 자료만 쓴다."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.common import io
from src.events import alerts as m9
from src.events.alerts import (ALERT_COLUMNS, AlertBook, AlertsNotCollected, AlertsNotStarted, backfill_alert_files,
                               build_alerts, cooldown_at, daily_alerts, engine_alerts)
from src.pipeline import run_daily_alerts as rda
from src.portfolio import params
from tests.conftest import item, page
from tests.events.test_alerts import COOLDOWN, DAYS, EVENTS, KST, TICKERS, alerts_frame, disclosures

CONFIG = {"events": EVENTS, "calendar": {"signal_cutoff_time": "15:30"}}
NOW = pd.Timestamp("2026-04-01 08:00:00", tz=KST)
TITLE = "상장폐지사유발생"
EMPTY = alerts_frame([("000001", "2026-01-06 23:59:59")]).iloc[0:0]
SIGNAL = "2026-02-20"


def at(text):
    return pd.Timestamp(text, tz=KST)


def stamped(available, generated):
    frame = alerts_frame([("000001", available)])
    frame["generated_at"] = pd.Series([at(generated)], dtype=pd.DatetimeTZDtype("ns", KST))
    return frame


def put(day, rows=()):
    return io.write_table("alerts", alerts_frame(list(rows)) if rows else EMPTY, date=day)


def fill(first, last):
    for day in pd.date_range(first, last, freq="D"):
        put(day.strftime("%Y-%m-%d"))


def collected(monkeypatch, rows):
    calls = []

    def fake(day, now):
        calls.append(day)
        return disclosures(rows)

    monkeypatch.setattr(m9, "_collected_disclosures", fake)
    return calls


@pytest.fixture
def forward(tmp_path, monkeypatch):
    """임시 저장소. 달력·설정·종목표를 합성으로 바꾸고 현재 시각을 고정한다."""
    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(m9, "load_trading_days", lambda: DAYS)
    monkeypatch.setattr(params, "load_prereg", lambda *args, **kwargs: CONFIG)
    real = io.read_table
    monkeypatch.setattr(io, "read_table", lambda name, **kw: TICKERS if name == "tickers" else real(name, **kw))
    monkeypatch.setattr(m9, "_now_kst", lambda: NOW)
    return tmp_path


class TestGeneratedAt:
    """금요일 공시 → 월요일 01-12 체결 경보. 체결일 15:30 전에 만들어진 경보만 그날 적용한다."""

    AVAILABLE = "2026-01-09 23:59:59"

    def schedule(self, generated):
        return AlertBook(stamped(self.AVAILABLE, generated), DAYS, COOLDOWN).engine_alerts("2026-01-05", "2026-01-16")

    def test_on_time_alert_liquidates_and_cools_down(self):
        cooldowns, liquidations = self.schedule("2026-01-10 08:00:00")
        assert liquidations == {"2026-01-12": frozenset({"000001"})}
        assert "000001" in cooldowns["2026-01-12"]

    def test_one_second_before_cutoff_counts(self):
        assert self.schedule("2026-01-12 15:29:59")[1] == {"2026-01-12": frozenset({"000001"})}

    def test_exact_cutoff_is_too_late(self):
        cooldowns, liquidations = self.schedule("2026-01-12 15:30:00")
        assert liquidations == {}
        assert "2026-01-12" not in cooldowns
        assert "000001" in cooldowns["2026-01-13"]

    def test_late_alert_cools_down_later_days_only(self):
        cooldowns, liquidations = self.schedule("2026-01-14 09:00:00")
        assert liquidations == {}
        assert sorted(cooldowns) == ["2026-01-14", "2026-01-15", "2026-01-16"]

    def test_backfilled_after_settlement_does_not_change_past_replay(self):
        before = AlertBook(EMPTY, DAYS, COOLDOWN).engine_alerts("2026-01-09", "2026-01-12")
        after = AlertBook(stamped("2026-01-10 23:59:59", "2026-01-13 09:00:00"), DAYS, COOLDOWN)
        assert after.engine_alerts("2026-01-09", "2026-01-12") == before

    def test_backtest_rows_are_not_filtered(self):
        book = AlertBook(alerts_frame([("000001", self.AVAILABLE)]), DAYS, COOLDOWN)
        cooldowns, liquidations = book.engine_alerts("2026-01-05", "2026-01-16")
        assert liquidations == {"2026-01-12": frozenset({"000001"})}
        assert "2026-01-12" in cooldowns

    def test_cooldown_at_uses_the_same_rule(self):
        late = AlertBook(stamped(self.AVAILABLE, "2026-01-13 16:00:00"), DAYS, COOLDOWN)
        early = AlertBook(stamped(self.AVAILABLE, "2026-01-13 15:00:00"), DAYS, COOLDOWN)
        assert late.cooldown_at("2026-01-12") == set()
        assert early.cooldown_at("2026-01-12") == {"000001"}

    def test_build_defaults_to_available_at(self):
        built = build_alerts(disclosures([("1", "2026-01-06", "C1", "000001", TITLE)]), TICKERS, EVENTS, DAYS)
        assert built.alerts.loc[0, "generated_at"] == built.alerts.loc[0, "available_at"]

    def test_build_uses_given_time(self):
        built = build_alerts(disclosures([("1", "2026-01-06", "C1", "000001", TITLE)]), TICKERS, EVENTS, DAYS,
                             generated_at=at("2026-01-07 08:00:00"))
        assert built.alerts.loc[0, "generated_at"] == at("2026-01-07 08:00:00")
        io.validate("alerts", built.alerts)


class TestForwardFiles:
    def window(self):
        return m9._forward_dates(SIGNAL, SIGNAL, DAYS, COOLDOWN)

    def fill_window(self):
        dates = self.window()
        fill(dates[0], dates[-1])
        return dates

    def test_window_is_oldest_possibly_active_date_through_signal(self):
        execution = int(DAYS.searchsorted(pd.Timestamp(SIGNAL), side="right"))
        dates = self.window()
        assert dates[0] == DAYS[execution - COOLDOWN].strftime("%Y-%m-%d")
        assert dates[-1] == SIGNAL
        assert len(dates) == (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days + 1

    @pytest.mark.parametrize("t", ["2026-02-06", SIGNAL, "2026-03-06"])
    def test_same_answer_as_reading_every_file(self, forward, t):
        rows = []
        for i, day in enumerate(pd.date_range("2026-01-05", "2026-03-20", freq="D")):
            row = (f"{i % 40:06d}", f"{day:%Y-%m-%d} 23:59:59")
            rows.append(row)
            put(f"{day:%Y-%m-%d}", [row])
        book = AlertBook(alerts_frame(rows), DAYS, COOLDOWN)
        assert cooldown_at(t) == book.cooldown_at(t)
        assert engine_alerts("2026-01-30", t) == book.engine_alerts("2026-01-30", t)

    def test_broken_files_outside_window_do_not_matter(self, forward):
        dates = self.fill_window()
        before = (pd.Timestamp(dates[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        after = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for day in (before, after):
            (forward / "alerts" / f"{day}.parquet").write_bytes(b"broken")
        assert cooldown_at(SIGNAL) == set()
        assert engine_alerts(SIGNAL, SIGNAL) == ({}, {})

    def test_broken_file_inside_window_stops(self, forward):
        dates = self.fill_window()
        (forward / "alerts" / f"{dates[3]}.parquet").write_bytes(b"broken")
        with pytest.raises(Exception) as caught:
            cooldown_at(SIGNAL)
        assert not isinstance(caught.value, FileNotFoundError)

    @pytest.mark.parametrize("position", [0, 5, -1])
    def test_missing_needed_date_is_not_collected(self, forward, position):
        dates = self.fill_window()
        (forward / "alerts" / f"{dates[position]}.parquet").unlink()
        with pytest.raises(AlertsNotCollected, match=dates[position]) as caught:
            cooldown_at(SIGNAL)
        assert not isinstance(caught.value, AlertsNotStarted)

    def test_no_files_at_all_is_not_started(self, forward):
        with pytest.raises(AlertsNotStarted):
            cooldown_at(SIGNAL)
        with pytest.raises(AlertsNotStarted):
            engine_alerts(SIGNAL, SIGNAL)

    def test_exceptions_stay_file_not_found_errors(self):
        assert issubclass(AlertsNotCollected, FileNotFoundError)
        assert issubclass(AlertsNotStarted, AlertsNotCollected)

    def test_zero_row_files_mean_collected_with_no_alerts(self, forward):
        self.fill_window()
        assert cooldown_at(SIGNAL) == set()


class TestDailyAlerts:
    DAY = "2026-02-20"

    def history(self):
        days = m9._forward_dates(self.DAY, "2026-02-19", DAYS, COOLDOWN)
        fill(days[0], days[-1])
        return days

    def test_zero_alerts_is_an_empty_valid_frame(self, forward, monkeypatch):
        self.history()
        collected(monkeypatch, [])
        out = daily_alerts(self.DAY)
        assert out.empty and list(out.columns) == ALERT_COLUMNS
        io.validate("alerts", out)

    def test_alert_has_run_time(self, forward, monkeypatch):
        self.history()
        collected(monkeypatch, [("1", self.DAY, "C1", "000001", TITLE)])
        out = daily_alerts(self.DAY)
        assert list(out["rcept_no"]) == ["1"]
        assert out.loc[0, "generated_at"] == NOW
        io.write_table("alerts", out, date=self.DAY)

    def test_before_eight_next_day_is_not_collected(self, forward, monkeypatch):
        calls = collected(monkeypatch, [])
        monkeypatch.setattr(m9, "_now_kst", lambda: at("2026-02-21 07:59:59"))
        with pytest.raises(AlertsNotCollected, match="08:00"):
            daily_alerts(self.DAY)
        assert calls == []

    def test_ready_at_eight_next_day(self, forward, monkeypatch):
        self.history()
        collected(monkeypatch, [("1", self.DAY, "C1", "000001", TITLE)])
        monkeypatch.setattr(m9, "_now_kst", lambda: at("2026-02-21 08:00:00"))
        io.validate("alerts", daily_alerts(self.DAY))

    def test_missing_history_stops_before_calling_opendart(self, forward, monkeypatch):
        days = self.history()
        (forward / "alerts" / f"{days[2]}.parquet").unlink()
        calls = collected(monkeypatch, [])
        with pytest.raises(AlertsNotCollected, match=days[2]):
            daily_alerts(self.DAY)
        assert calls == []

    def test_same_event_inside_cooldown_comes_from_earlier_file(self, forward, monkeypatch):
        days = self.history()
        put(days[-1], [("000001", f"{days[-1]} 23:59:59")])
        collected(monkeypatch, [("1", self.DAY, "C1", "000001", TITLE)])
        assert daily_alerts(self.DAY).empty

    def test_new_event_after_cooldown_ended(self, forward, monkeypatch):
        days = self.history()
        expired = (pd.Timestamp(days[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        put(expired, [("000001", f"{expired} 23:59:59")])
        collected(monkeypatch, [("1", self.DAY, "C1", "000001", TITLE)])
        assert list(daily_alerts(self.DAY)["rcept_no"]) == ["1"]

    def test_day_by_day_equals_whole_history(self, forward, monkeypatch):
        calendar = pd.date_range("2026-01-05", "2026-03-20", freq="D")
        rows = [(str(i), f"{day:%Y-%m-%d}", f"C{i % 3}", f"{i % 3 + 1:06d}", TITLE)
                for i, day in enumerate(calendar) if i % 4 == 0]
        rows = [row for row in rows if row[3] != "000002"]
        whole = build_alerts(disclosures(rows), TICKERS, EVENTS, DAYS).alerts
        cooldown_seed = calendar[0] - pd.Timedelta(days=45)
        fill(cooldown_seed, calendar[0] - pd.Timedelta(days=1))
        monkeypatch.setattr(m9, "load_trading_days", lambda: pd.bdate_range("2025-11-03", periods=120))
        for day in calendar:
            key = day.strftime("%Y-%m-%d")
            collected(monkeypatch, [row for row in rows if row[1] == key])
            io.write_table("alerts", daily_alerts(key), date=key)
        got = m9._concat_alerts([io.read_table("alerts", date=f"{day:%Y-%m-%d}") for day in calendar])
        assert list(got["rcept_no"]) == list(whole["rcept_no"])

    def test_rerun_keeps_first_generated_time(self, forward, monkeypatch):
        self.history()
        collected(monkeypatch, [("1", self.DAY, "C1", "000001", TITLE)])
        io.write_table("alerts", daily_alerts(self.DAY), date=self.DAY)
        later = NOW + pd.Timedelta(days=2)
        monkeypatch.setattr(m9, "_now_kst", lambda: later)
        collected(monkeypatch, [("1", self.DAY, "C1", "000001", TITLE), ("2", self.DAY, "C3", "000003", TITLE)])
        result = daily_alerts(self.DAY)
        times = result.set_index("rcept_no")["generated_at"]
        assert (times["1"], times["2"]) == (NOW, later)
        io.write_table("alerts", result, date=self.DAY)


class TestDailyRunner:
    """D-06 실행기가 실제 `daily_alerts`를 부르는 경로."""

    DAY = "2026-02-20"

    def prepare(self):
        days = m9._forward_dates(self.DAY, "2026-02-19", DAYS, COOLDOWN)
        fill(days[0], days[-1])

    def test_zero_alerts_saves_zero_rows(self, forward, monkeypatch):
        self.prepare()
        collected(monkeypatch, [])
        assert rda.run(self.DAY) == 0
        assert io.read_table("alerts", date=self.DAY).empty

    def test_alerts_are_saved(self, forward, monkeypatch):
        self.prepare()
        collected(monkeypatch, [("1", self.DAY, "C1", "000001", TITLE)])
        assert rda.run(self.DAY) == 0
        assert list(io.read_table("alerts", date=self.DAY)["rcept_no"]) == ["1"]

    def test_not_collected_writes_nothing(self, forward, monkeypatch):
        self.prepare()

        def refuse(day, now):
            raise AlertsNotCollected("수집이 끝나지 않았다")

        monkeypatch.setattr(m9, "_collected_disclosures", refuse)
        assert rda.run(self.DAY) == 1
        assert not (forward / "alerts" / f"{self.DAY}.parquet").exists()
        assert "미수집" in (forward / "logs" / f"alerts_{self.DAY}.log").read_text(encoding="utf-8")

    def test_run_after_midnight_writes_nothing(self, forward, monkeypatch):
        self.prepare()
        calls = collected(monkeypatch, [])
        monkeypatch.setattr(m9, "_now_kst", lambda: at("2026-02-21 00:30:00"))
        assert rda.run(self.DAY) == 1
        assert calls == []
        assert not (forward / "alerts" / f"{self.DAY}.parquet").exists()


class TestCollectedDisclosures:
    """실제 수집·원본 보관·정제 경로를 가짜 OpenDART 응답으로 지난다."""

    NOW = pd.Timestamp("2026-10-28 21:00:00", tz=KST)

    def use(self, monkeypatch, tmp_path, client_factory, scenario):
        monkeypatch.setattr(io, "ROOT", tmp_path)
        client, session, _ = client_factory(scenario)
        monkeypatch.setattr("src.data.dart_client.DartClient", lambda *args, **kwargs: client)
        monkeypatch.setattr(m9, "_load_env", lambda: None)
        return session

    def test_returns_only_rows_available_on_that_day(self, monkeypatch, tmp_path, client_factory):
        def scenario(key, count):
            bgn, end, kind, _ = key
            if kind == "I" and bgn == "20261027":
                return page([item(1, day="20261027", report_nm=TITLE)])
            if kind == "I":
                return page([item(2, day="20261026", report_nm=TITLE)])
            return page(status="013")

        self.use(monkeypatch, tmp_path, client_factory, scenario)
        frame = m9._collected_disclosures(pd.Timestamp("2026-10-27"), self.NOW)
        assert list(frame["rcept_no"]) == ["20261027000001"]
        assert frame.loc[0, "stock_code"] == "005930"
        assert frame.loc[0, "available_at"] == at("2026-10-27 23:59:59")

    def test_no_disclosures_is_collected_empty(self, monkeypatch, tmp_path, client_factory):
        self.use(monkeypatch, tmp_path, client_factory, lambda key, count: page(status="013"))
        assert m9._collected_disclosures(pd.Timestamp("2026-10-27"), self.NOW).empty

    def test_empty_day_runs_through_to_a_zero_row_file(self, forward, monkeypatch, client_factory):
        client, _, _ = client_factory(lambda key, count: page(status="013"))
        monkeypatch.setattr("src.data.dart_client.DartClient", lambda *args, **kwargs: client)
        monkeypatch.setattr(m9, "_load_env", lambda: None)
        days = m9._forward_dates("2026-02-20", "2026-02-19", DAYS, COOLDOWN)
        fill(days[0], days[-1])
        assert rda.run("2026-02-20") == 0
        assert io.read_table("alerts", date="2026-02-20").empty

    def test_open_window_is_not_final(self, monkeypatch, tmp_path, client_factory):
        self.use(monkeypatch, tmp_path, client_factory, lambda key, count: page(status="013"))
        with pytest.raises(AlertsNotCollected, match="끝나지 않았다"):
            m9._collected_disclosures(pd.Timestamp("2026-10-28"), self.NOW)

    def test_interrupted_collection_is_not_collected(self, monkeypatch, tmp_path, client_factory):
        self.use(monkeypatch, tmp_path, client_factory, lambda key, count: page(status="020"))
        with pytest.raises(AlertsNotCollected, match="끝나지 않았다"):
            m9._collected_disclosures(pd.Timestamp("2026-10-27"), self.NOW)


class TestBackfill:
    FIRST, LAST = "2026-01-12", "2026-01-16"

    def built(self, monkeypatch, rows, windows):
        frame = disclosures(rows)
        observations = pd.DataFrame({"rcept_no": frame["rcept_no"], "stock_code": frame["stock_code"],
                                     "fetched_at": "2026-01-20T08:00:00+09:00"})
        result = SimpleNamespace(df=frame, observations=observations,
                                 windows=pd.DataFrame(windows, columns=["pblntf_ty", "bgn", "end", "state"]))
        monkeypatch.setattr("src.data.disclosures.build_disclosures", lambda *args, **kwargs: result)

    @staticmethod
    def complete(end="2026-01-31"):
        return [(kind, "2025-10-01", end, "complete") for kind in ("B", "I")]

    def files(self, root):
        return sorted(p.stem for p in (root / "alerts").glob("*.parquet")) if (root / "alerts").exists() else []

    def test_writes_every_calendar_day_with_zero_rows(self, forward, monkeypatch):
        self.built(monkeypatch, [("1", "2026-01-14", "C1", "000001", TITLE)], self.complete())
        assert backfill_alert_files(self.FIRST, self.LAST) == list(
            pd.date_range(self.FIRST, self.LAST, freq="D").strftime("%Y-%m-%d"))
        assert io.read_table("alerts", date="2026-01-13").empty
        saved = io.read_table("alerts", date="2026-01-14")
        assert list(saved["rcept_no"]) == ["1"] and saved.loc[0, "generated_at"] == NOW

    def test_earlier_history_decides_same_event(self, forward, monkeypatch):
        rows = [("1", "2026-01-06", "C1", "000001", TITLE), ("2", "2026-01-14", "C1", "000001", TITLE)]
        self.built(monkeypatch, rows, self.complete())
        backfill_alert_files(self.FIRST, self.LAST)
        assert io.read_table("alerts", date="2026-01-14").empty

    def test_uncollected_days_write_nothing(self, forward, monkeypatch):
        self.built(monkeypatch, [], self.complete(end="2026-01-14"))
        with pytest.raises(AlertsNotCollected, match="2일"):
            backfill_alert_files(self.FIRST, self.LAST)
        assert self.files(forward) == []

    def test_one_disclosure_type_missing_is_uncollected(self, forward, monkeypatch):
        self.built(monkeypatch, [], [("B", "2025-10-01", "2026-01-31", "complete")])
        with pytest.raises(AlertsNotCollected):
            backfill_alert_files(self.FIRST, self.LAST)

    def test_incomplete_window_does_not_count(self, forward, monkeypatch):
        self.built(monkeypatch, [], [(kind, "2025-10-01", "2026-01-31", "provisional") for kind in ("B", "I")])
        with pytest.raises(AlertsNotCollected):
            backfill_alert_files(self.FIRST, self.LAST)

    def test_no_collection_at_all(self, forward, monkeypatch):
        self.built(monkeypatch, [], [])
        with pytest.raises(AlertsNotCollected):
            backfill_alert_files(self.FIRST, self.LAST)

    def test_before_eight_after_last_day_refused(self, forward, monkeypatch):
        self.built(monkeypatch, [], self.complete())
        monkeypatch.setattr(m9, "_now_kst", lambda: at("2026-01-17 07:59:59"))
        with pytest.raises(AlertsNotCollected, match="08:00"):
            backfill_alert_files(self.FIRST, self.LAST)

    def test_rerun_keeps_first_generated_time(self, forward, monkeypatch):
        self.built(monkeypatch, [("1", "2026-01-14", "C1", "000001", TITLE)], self.complete())
        backfill_alert_files(self.FIRST, self.LAST)
        monkeypatch.setattr(m9, "_now_kst", lambda: NOW + pd.Timedelta(days=3))
        backfill_alert_files(self.FIRST, self.LAST)
        assert io.read_table("alerts", date="2026-01-14").loc[0, "generated_at"] == NOW
