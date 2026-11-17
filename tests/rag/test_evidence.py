"""합성 저장소에서 주간 M6 템플릿의 대상·공시 시점·출력을 검증한다."""
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from src.common import io
from src.data.disclosures import collect_daily
from src.events import alerts as m9
from src.pipeline.run_weekly import StepContext, resolve
from src.portfolio import params
from src.rag import evidence
from tests.conftest import FakeClock, item, page
from tests.events.test_alerts import alerts_frame, disclosures

T = "2026-02-20"
DAY = pd.Timestamp(T)
PREVIOUS = "2026-02-12"  # 합성 달력에서 전 주 금요일은 휴장
DAYS = pd.bdate_range("2026-01-05", "2026-02-25").difference(pd.DatetimeIndex(["2026-02-13"]))
ARCHIVE_READY = "2026-02-21T08:00:00+09:00"
RECORD_DEADLINE = "2026-02-23T09:00:00+09:00"


def ticker(n):
    return f"{n:06d}"


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    tables = {
        "signals": pd.DataFrame({"ticker": [ticker(1)], "rank": [1], "crash_flag": [1]}),
        "tickers": pd.DataFrame({
            "ticker": [ticker(1)], "listing_date": pd.to_datetime(["2020-01-01"]),
            "delisting_date": pd.to_datetime([None]),
        }),
    }
    alerts = {
        day: alerts_frame([])
        for day in pd.date_range(DAYS[0], DAY).strftime("%Y-%m-%d")
    }

    def read(name, *, date=None, stub=False):
        assert not stub
        if name == "alerts":
            return alerts[date]
        if name == "signals":
            assert date == T
        return tables[name]

    monkeypatch.setattr(io, "ROOT", tmp_path)
    monkeypatch.setattr(io, "read_table", read)
    monkeypatch.setattr(io, "list_dates", lambda name: sorted(alerts) if name == "alerts" else [])
    monkeypatch.setattr(m9, "load_trading_days", lambda: DAYS)
    monkeypatch.setattr(params, "load_prereg", lambda: {
        "events": {"cooldown_trading_days": 20}, "calendar": {"signal_cutoff_time": "15:30"},
    })
    return tables, alerts


def archive_filings(client_factory, filings=(), *, day=DAY, observed_at=ARCHIVE_READY,
                    lookback=35, types=("B", "I")):
    def scenario(key, attempt):
        start, end, kind, _ = key
        rows = [row for row in filings if kind == "B" and start <= row["rcept_dt"] <= end]
        return page(rows) if rows else page(status="013")

    clock = FakeClock(datetime.fromisoformat(observed_at))
    client, _, _ = client_factory(scenario, clock=clock)
    return collect_daily(day.date(), lookback_days=lookback, types=types, client=client,
                         data_raw=io.ROOT / "data" / "raw")


def test_archived_disclosures_calendar_boundaries_and_schema(inputs, client_factory):
    position = DAYS.get_loc(DAY)
    earliest = DAYS[position - 19]
    too_old = DAYS[position - 20]
    filings = [
        item(n, day=day.strftime("%Y%m%d"), stock_code=ticker(1), report_nm=title)
        for n, day, title in [
            (1, too_old, "검색 창보다 이른 공시"),
            (2, earliest, "검색 창 첫 거래일 공시"),
            (3, pd.Timestamp("2026-02-14"), "비거래일 공시"),
            (4, DAY - pd.Timedelta(days=1), "전일 공시"),
            (5, DAY, "신호일 공시"),
        ]
    ]

    archive_filings(client_factory, filings, lookback=(DAY - too_old).days)
    resolve("src.rag.evidence:m6_step")(StepContext(T))

    record, = io.read_evidence(T)
    assert set(record) == {"date", "ticker", "signal_type", "summary", "sources", "disclaimer"}
    assert (record["date"], record["ticker"], record["signal_type"]) == (T, ticker(1), "crash")
    assert record["disclaimer"] == io.DISCLAIMER
    assert record["summary"].splitlines() == [
        "최근 20거래일 공시: 주요사항보고 전일 공시 (2026-02-19) [1]",
        f"최근 20거래일 공시: 주요사항보고 검색 창 첫 거래일 공시 ({earliest:%Y-%m-%d}) [2]",
    ]
    assert record["sources"] == [
        {
            "n": n, "type": "disclosure", "id": row["rcept_no"], "title": row["report_nm"],
            "url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + row["rcept_no"],
            "timestamp": pd.Timestamp(row["rcept_dt"], tz=m9.KST).replace(
                hour=23, minute=59, second=59
            ).isoformat(),
        }
        for n, row in enumerate([filings[3], filings[1]], 1)
    ]


def test_availability_is_strict_and_summary_has_at_most_three_citations(inputs, monkeypatch):
    frame = disclosures([
        (str(i), "2026-02-19", "C1", ticker(1), title)
        for i, title in enumerate(["정각", "직전", "기타", "주식매수선택권", "더 이전"])
    ])
    frame["available_at"] = pd.DatetimeIndex([
        pd.Timestamp(f"{T} {time}", tz=m9.KST)
        for time in ("15:30:00", "15:29:59", "15:29:58", "15:29:57", "15:29:56")
    ]).as_unit("ns")
    observed = frame.assign(fetched_at=frame["available_at"], pblntf_ty="I", run_id="run", window_id="I")
    windows = pd.DataFrame([
        {"run_id": "run", "window_id": kind, "pblntf_ty": kind, "state": "complete",
         "bgn": "2026-01-05", "end": "2026-02-19", "accepted_attempts": [1],
         "attempts": [{"n": 1, "finished_at": ARCHIVE_READY}]}
        for kind in ("B", "I")
    ])
    monkeypatch.setattr(evidence, "build_disclosures", lambda *args, **kw: SimpleNamespace(
        df=frame, observations=observed, windows=windows,
    ))
    # 날짜 전용 원본에서 직접 만들 수 없는 시·분·초 경계는 관측 버전에 주입한다.
    monkeypatch.setattr(evidence, "_latest_versions", lambda observations, days: observations)

    evidence.m6_step(StepContext(T))

    record, = io.read_evidence(T)
    assert [source["id"] for source in record["sources"]] == ["1", "2", "3"]
    assert record["sources"][0]["timestamp"] == "2026-02-20T15:29:59+09:00"
    assert record["sources"][2]["title"] == "주식매수선택권"
    assert record["summary"].splitlines()[-1] == "최근 20거래일 공시: 거래소공시 주식…선택권 (2026-02-19) [3]"


def test_targets_cooldown_rank_limit_and_alert_window(inputs, client_factory):
    tables, alerts = inputs
    tables["signals"] = pd.DataFrame({
        "ticker": [ticker(n) for n in range(1, 25)],
        "rank": range(1, 25), "crash_flag": [1] + [0] * 23,
    }).iloc[::-1]
    alerts["2026-02-11"] = alerts_frame([(ticker(2), "2026-02-11 23:59:59")])
    previous_alerts = alerts_frame([
        (ticker(990003), f"{PREVIOUS} 15:29:59"),
        (ticker(1), f"{PREVIOUS} 15:30:00"),
        (ticker(990004), f"{PREVIOUS} 15:30:00"),
        (ticker(3), f"{PREVIOUS} 15:30:00"),
        (ticker(990005), f"{PREVIOUS} 15:30:00"),
    ])
    deadline = pd.Timestamp(RECORD_DEADLINE).tz_convert(m9.KST)
    for row, seconds in enumerate((-1, 0, 1), start=2):
        previous_alerts.loc[row, "generated_at"] = deadline + pd.Timedelta(seconds=seconds)
    alerts[PREVIOUS] = previous_alerts.iloc[:3]
    alerts[T] = alerts_frame([
        (ticker(990001), f"{T} 15:29:59"),
        (ticker(990002), f"{T} 15:30:00"),
    ])
    assert ticker(2) in m9.cooldown_at(T)
    archive_filings(client_factory)

    evidence.m6_step(StepContext(T))

    records = io.read_evidence(T)
    assert {row["ticker"]: row["signal_type"] for row in records} == {
        ticker(1): "crash",
        **{ticker(n): "rank" for n in range(3, 23)},
        ticker(990001): "alert",
        ticker(990004): "alert",
    }
    assert len(records) == 23
    assert all(row["summary"] == "최근 20거래일 공시: 조회된 근거가 없습니다." for row in records)
    assert all(row["sources"] == [] for row in records)

    # 같은 available_at의 경보가 기록 마감 정각과 직후에 추가돼도 과거 근거는 같다.
    alerts[PREVIOUS] = previous_alerts
    cooldown = m9.cooldown_at(T)
    assert ticker(990004) in cooldown
    assert {ticker(3), ticker(990005)}.isdisjoint(cooldown)
    evidence.m6_step(StepContext(T))
    assert io.read_evidence(T) == records


def test_mixed_iso_observation_times_keep_latest_before_deadline(inputs, client_factory):
    old = item(day="20260219", stock_code=ticker(1), report_nm="첫 관측")
    latest = item(day="20260219", stock_code=ticker(1), report_nm="마감 전 최신 관측")
    archive_filings(client_factory, [old], day=DAY - pd.Timedelta(days=1),
                    observed_at="2026-02-20T08:00:00.100003+09:00")
    archive_filings(client_factory, [latest])

    evidence.m6_step(StepContext(T))

    record, = io.read_evidence(T)
    assert [source["title"] for source in record["sources"]] == ["마감 전 최신 관측"]


@pytest.mark.parametrize("observed_at", [RECORD_DEADLINE, "2026-02-26T10:00:00+09:00"])
def test_late_archive_cannot_change_past_evidence(inputs, client_factory, observed_at):
    tables, _ = inputs
    tables["tickers"] = pd.DataFrame({
        "ticker": [ticker(1), ticker(2)], "listing_date": pd.to_datetime(["2020-01-01"] * 2),
        "delisting_date": pd.to_datetime([None, None]),
    })
    tables["signals"] = pd.DataFrame({"ticker": [ticker(1), ticker(2)], "rank": [1, 2], "crash_flag": [1, 0]})
    original = item(day="20260219", stock_code=ticker(1), report_nm="기록 마감 전 원본")
    archive_filings(client_factory, [original])
    evidence.m6_step(StepContext(T))
    before = io.read_evidence(T)
    assert before[0]["sources"][0]["title"] == "기록 마감 전 원본"

    revised = item(day="20260219", stock_code=ticker(2), report_nm="기록 마감 뒤 제목")
    late_only = item(2, day="20260219", stock_code=ticker(2), report_nm="뒤늦게 발견된 공시")
    archive_filings(client_factory, [revised, late_only], day=DAY + pd.Timedelta(days=1),
                    observed_at=observed_at)

    evidence.m6_step(StepContext(T))
    assert io.read_evidence(T) == before


@pytest.mark.parametrize("case", ["absent", "missing_day", "missing_type", "completed_after_deadline"])
def test_missing_disclosure_coverage_stops_without_writing(inputs, client_factory, case):
    if case == "missing_day":
        archive_filings(client_factory, lookback=5)
    elif case == "missing_type":
        archive_filings(client_factory, types=("B",))
    elif case == "completed_after_deadline":
        archive_filings(client_factory, observed_at=RECORD_DEADLINE)

    with pytest.raises(m9.AlertsNotCollected, match="공시 수집이 끝나지 않은 날짜"):
        evidence.m6_step(StepContext(T))
    assert io.list_dates("evidence") == []


@pytest.mark.parametrize("table", ["signals", "index", "tickers", "alerts"])
def test_stub_inputs_short_circuit_to_empty_stub(inputs, monkeypatch, table):
    def unexpected():
        pytest.fail("스텁 입력에서 계산하면 안 된다")

    monkeypatch.setattr(m9, "load_trading_days", unexpected)
    ctx = StepContext(T, stub_tables={table})
    evidence.m6_step(ctx)
    assert ctx.stub_tables == {table, "evidence"}
    assert io.read_evidence(T, stub=True) == []
    assert io.list_dates("evidence") == []


@pytest.mark.parametrize("missing_day", [PREVIOUS, T])
def test_uncollected_alerts_propagate_without_writing(inputs, missing_day):
    _, alerts = inputs
    del alerts[missing_day]

    with pytest.raises(m9.AlertsNotCollected, match=missing_day):
        evidence.m6_step(StepContext(T))
    assert io.list_dates("evidence") == []
