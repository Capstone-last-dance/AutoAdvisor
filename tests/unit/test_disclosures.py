"""완결성, 중단 복구, 시점 규약과 저장 거부 조건을 합성 데이터로 검증한다."""

import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from conftest import FakeClock, item, page
from src.data import disclosures as mod
from src.data.dart_client import KST
from src.data.disclosure_archive import RawArchive
from src.data.disclosures import (EXTENDED_COLUMNS, V03_COLUMNS, PublishRefused, Window,
                                  build_disclosures, collect_daily, collect_run, coverage_counts,
                                  make_windows, new_archive, normalize_report_nm, plan_run,
                                  publish_disclosures, run_coverage, write_build_report)

TODAY = date(2026, 10, 28)
START, END = date(2025, 1, 1), date(2025, 3, 31)
KEY = "B_20250101_20250331"


def archive_at(tmp_path, name="20261028_PM_disclosures_backfill"):
    return RawArchive.create(tmp_path / "data/raw", name, TODAY)


def run_window(tmp_path, client_factory, scenario, *, start=START, end=END, types=("B",),
               attempts=3, split=False, name="20261028_PM_disclosures_backfill"):
    client, session, clock = client_factory(scenario)
    archive = archive_at(tmp_path, name)
    plan_run(start, end, types=types, client=client, archive=archive, today=TODAY)
    if archive.load_manifest()["stop_reason"] is None:
        collect_run(client=client, archive=archive, today=TODAY, max_attempts=attempts, split_on_failure=split)
    return archive, client, session, clock


def built(tmp_path, client_factory, records=None, *, start=START, end=END, name="20261028_PM_disclosures_backfill", types=("B",)):
    records = records if records is not None else [item()]
    archive, _, _, _ = run_window(tmp_path, client_factory, [page(records) for _ in types], start=start, end=end,
                                  name=name, types=types)
    return build_disclosures([archive.root], today=TODAY), archive


def test_quarter_windows_aligned_and_clipped():
    windows = make_windows(date(2010, 2, 15), date(2010, 8, 10), ["B"])
    assert [(w.bgn.isoformat(), w.end.isoformat()) for w in windows] == [
        ("2010-02-15", "2010-03-31"), ("2010-04-01", "2010-06-30"), ("2010-07-01", "2010-08-10")]
    assert [w.pblntf_ty for w in make_windows(START, END, ["I", "B"])] == ["I", "B"]


def test_window_max_three_months():
    with pytest.raises(ValueError):
        make_windows(START, END, ["B"], months=4)


def test_open_window_flag():
    assert Window("B", TODAY, TODAY).is_open(TODAY)
    assert not Window("B", START, END).is_open(TODAY)


def test_default_range_has_136_windows():
    assert len(make_windows(date(2010, 1, 1), date(2026, 10, 27), ["B", "I"])) == 136


def test_plan_saves_page1_and_expected_totals(tmp_path, client_factory):
    client, session, _ = client_factory([page([item()], total=3, pages=3)])
    archive = archive_at(tmp_path)
    summary = plan_run(START, END, types=["B"], client=client, archive=archive, today=TODAY)
    entry = archive.load_manifest()["windows"][KEY]
    assert summary.expected_rows == summary.expected_calls == 3
    assert entry["state"] == "planned"
    assert archive.fetched_page_numbers(KEY, 1) == {1}
    assert len(session.requests) == 1


def test_multi_page_window_complete(tmp_path, client_factory):
    archive, _, session, _ = run_window(tmp_path, client_factory, [
        page([item(1)], total=3, pages=3), page([item(2)], total=3, pages=3, number=2),
        page([item(3)], total=3, pages=3, number=3)])
    assert archive.load_manifest()["windows"][KEY]["state"] == "complete"
    assert [r[-1] for r in session.requests] == [1, 2, 3]


def test_never_requests_beyond_total_page(tmp_path, client_factory):
    archive, _, session, _ = run_window(tmp_path, client_factory, [page([item()])])
    assert [r[-1] for r in session.requests] == [1]
    assert archive.load_manifest()["status"] == "complete"


def test_zero_result_window_complete(tmp_path, client_factory):
    archive, _, session, _ = run_window(tmp_path, client_factory, [page(status="013")])
    assert archive.load_manifest()["windows"][KEY]["state"] == "complete"
    assert len(session.requests) == 1


def test_rate_limit_then_resume_without_refetch(tmp_path, client_factory):
    archive, _, _, _ = run_window(tmp_path, client_factory, [
        page([item(1)], total=2, pages=2), page(status="020")])
    manifest = archive.load_manifest()
    assert manifest["stop_reason"] == "rate_limit"
    client, session, _ = client_factory([page([item(2)], total=2, pages=2, number=2)],
                                        clock=FakeClock(datetime(2026, 10, 29, 21, tzinfo=KST)))
    result = collect_run(client=client, archive=RawArchive.open(archive.root), today=TODAY)
    assert result.status == "complete"
    assert [r[-1] for r in session.requests] == [2]
    assert sum(archive.load_manifest()["calls_by_date"].values()) == 3


def test_rate_limit_cannot_resume_on_same_kst_day(tmp_path, client_factory):
    archive, _, _, _ = run_window(tmp_path, client_factory, [
        page([item(1)], total=2, pages=2), page(status="020")])
    client, session, _ = client_factory([])
    result = collect_run(client=client, archive=archive, today=TODAY)
    assert result.stop_reason == "rate_limit" and session.requests == []


def test_interrupt_then_resume(tmp_path, client_factory):
    archive, _, _, _ = run_window(tmp_path, client_factory, [
        page([item(1)], total=2, pages=2), KeyboardInterrupt()])
    assert archive.load_manifest()["stop_reason"] == "interrupted"
    client, session, _ = client_factory([page([item(2)], total=2, pages=2, number=2)])
    assert collect_run(client=client, archive=archive, today=TODAY).status == "complete"
    assert [r[-1] for r in session.requests] == [2]


def test_budget_resume_uses_persisted_calls(tmp_path, client_factory):
    client, _, _ = client_factory([page([item(1)], total=2, pages=2)], limit=1)
    archive = archive_at(tmp_path)
    plan_run(START, END, types=["B"], client=client, archive=archive, today=TODAY)
    assert collect_run(client=client, archive=archive, today=TODAY).stop_reason == "daily_budget"
    resumed, session, _ = client_factory([], limit=1)
    assert collect_run(client=resumed, archive=archive, today=TODAY).stop_reason == "daily_budget"
    assert not session.requests


def test_closed_window_count_change_new_attempt(tmp_path, client_factory):
    archive, _, session, _ = run_window(tmp_path, client_factory, [
        page([item(1)], total=2, pages=2), page([item(2)], total=3, pages=2, number=2),
        page([item(1), item(2), item(3)])])
    entry = archive.load_manifest()["windows"][KEY]
    assert entry["state"] == "complete"
    assert entry["attempts"][0]["result"] == "changed"
    assert entry["accepted_attempts"] == [2]
    assert [r[-1] for r in session.requests] == [1, 2, 1]


def test_union_of_attempts_completes(tmp_path, client_factory):
    archive, _, _, _ = run_window(tmp_path, client_factory, [
        page([item(1)], total=3, pages=2), page([item(2)], total=3, pages=2, number=2),
        page([item(2)], total=3, pages=2), page([item(3)], total=3, pages=2, number=2)])
    entry = archive.load_manifest()["windows"][KEY]
    assert entry["state"] == "complete"
    assert entry["accepted_attempts"] == [1, 2]
    assert entry["accepted_unique_rcept_no"] == 3


def test_split_after_max_attempts(tmp_path, client_factory):
    def scenario(key, count):
        bgn, end, _, number = key
        if end == "20250331" and bgn == "20250101":
            return page([item(1)], total=2)
        return page(status="013")
    archive, _, _, _ = run_window(tmp_path, client_factory, scenario, attempts=2, split=True)
    manifest = archive.load_manifest()
    assert manifest["windows"][KEY]["state"] == "split"
    assert len(manifest["windows"][KEY]["children"]) == 3
    assert manifest["status"] == "complete"


def test_failed_when_union_exceeds_total(tmp_path, client_factory):
    archive, _, _, _ = run_window(tmp_path, client_factory, [
        page([item(1)], total=2), page([item(2), item(3)], total=2)])
    assert archive.load_manifest()["windows"][KEY]["state"] == "failed"


def test_open_window_provisional_and_growing_total(tmp_path, client_factory):
    client, session, _ = client_factory([page([item(1)], total=2, pages=2),
                                        page([item(2)], total=3, pages=3, number=2),
                                        page([item(3)], total=3, pages=3, number=3)])
    archive = archive_at(tmp_path)
    plan_run(TODAY, TODAY, types=["B"], client=client, archive=archive, today=TODAY)
    assert collect_run(client=client, archive=archive, today=TODAY).status == "complete"
    entry = next(iter(archive.load_manifest()["windows"].values()))
    assert entry["state"] == "provisional" and entry["expected_total"] == 3
    assert len(session.requests) == 3


def test_provisional_recollected_and_can_close(tmp_path, client_factory):
    client, _, _ = client_factory([page([item(day="20261028")])])
    archive = archive_at(tmp_path)
    plan_run(TODAY, TODAY, types=["B"], client=client, archive=archive, today=TODAY)
    resumed, _, _ = client_factory([page([item(day="20261028")])])
    collect_run(client=resumed, archive=archive, today=TODAY + timedelta(days=1))
    entry = next(iter(archive.load_manifest()["windows"].values()))
    assert entry["state"] == "complete" and entry["accepted_attempts"] == [2]


def test_rcept_dt_outside_window_recorded_not_fatal(tmp_path, client_factory):
    result, archive = built(tmp_path, client_factory, [item(day="20250401")])
    assert next(iter(archive.load_manifest()["windows"].values()))["state"] == "complete"
    assert result.quality["rcept_dt_outside_window"] == 1
    assert "rcept_dt_outside_window" in (archive.root / "events.jsonl").read_text()


def test_malformed_item_kept_and_flagged(tmp_path, client_factory):
    archive, _, session, _ = run_window(tmp_path, client_factory, [page([item(rcept_no="bad")])], attempts=1)
    assert len(session.requests) == 1
    assert next(iter(archive.load_manifest()["windows"].values()))["state"] == "failed"
    assert b'bad' in next(archive.root.rglob("page_0001.json")).read_bytes()


def test_daily_windows_closed_and_open(tmp_path, client_factory):
    client, session, _ = client_factory(lambda key, count: page(status="013"))
    result = collect_daily(TODAY, client=client, data_raw=tmp_path)
    manifest = RawArchive(result.run_dir).load_manifest()
    assert manifest["run_id"] == "20261028_PM_disclosures_daily"
    assert len(session.requests) == 4
    assert sorted(w["state"] for w in manifest["windows"].values()) == ["complete", "complete", "provisional", "provisional"]
    again, _, _ = client_factory(lambda key, count: page(status="013"))
    collect_daily(TODAY, client=again, data_raw=tmp_path)
    assert len(list(tmp_path.glob("*/*/opendart_list"))) == 1


def test_corrupt_partial_pair_restarts_without_overwrite(tmp_path, client_factory):
    client, _, _ = client_factory([page([item(1)], total=2, pages=2)])
    archive = archive_at(tmp_path)
    plan_run(START, END, types=["B"], client=client, archive=archive, today=TODAY)
    path = archive._attempt_dir(KEY, 1) / "page_0002.json"
    path.write_bytes(b"orphan")
    resumed, _, _ = client_factory([page([item(1), item(2)])])
    assert collect_run(client=resumed, archive=archive, today=TODAY).status == "complete"
    assert path.read_bytes() == b"orphan"


def test_normalize_report_nm():
    assert normalize_report_nm(" [기재정정]\u3000유상 ㆍ 증자·결정・\n") == "[기재정정]유상증자결정・"


def test_ticker_leading_zero_and_null(tmp_path, client_factory):
    result, _ = built(tmp_path, client_factory, [item(1), item(2, stock_code=""), item(3, stock_code="00AB12"), item(4, stock_code="bad")])
    assert result.df.ticker.tolist() == [None, None, None, None]
    assert result.observations.stock_code.tolist() == ["005930", "", "00AB12", "bad"]
    assert result.quality["invalid_stock_code"] == 1
    assert result.quality["ticker_unverified_asof"] == 2


def test_available_at_uses_later_date(tmp_path, client_factory):
    result, _ = built(tmp_path, client_factory, [item(day="20250108", rcept_dt="20250109")])
    assert result.df.available_at.iloc[0].isoformat() == "2025-01-09T23:59:59+09:00"
    assert result.df.rcept_dt.isna().all()
    assert result.quality["rcept_no_date_differs"] == 1


def test_v03_and_extended_columns(tmp_path, client_factory):
    result, archive = built(tmp_path, client_factory)
    legacy = build_disclosures([archive.root], schema_mode="v03", today=TODAY)
    assert list(result.df.columns) == EXTENDED_COLUMNS
    assert list(legacy.df.columns) == V03_COLUMNS
    assert str(result.df.rcept_date.dtype) == "datetime64[ns]"
    assert result.df.rcept_dt.isna().all()
    assert result.df.available_at.iloc[0] == legacy.df.rcept_dt.iloc[0]
    assert result.df.rcept_dt.dtype.unit == "ns"


def test_mutable_fields_take_latest(tmp_path, client_factory):
    _, first = built(tmp_path / "first", client_factory, [item(corp_name="옛이름")])
    _, second = built(tmp_path / "second", client_factory, [item(corp_name="새이름")], name="20261028_PM_disclosures_backfill_2")
    for meta_path in second.root.rglob("*.meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["fetched_at"] = "2026-10-28T22:00:00+09:00"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    result = build_disclosures([first.root, second.root], today=TODAY)
    assert result.df.corp_name.tolist() == ["새이름"]
    assert result.quality["mutable_changed_corp_name"] == 1
    assert result.conflicts.empty


def test_immutable_conflict_recorded(tmp_path, client_factory):
    _, first = built(tmp_path / "first", client_factory, [item(report_nm="이름1")])
    _, second = built(tmp_path / "second", client_factory, [item(report_nm="이름2")], name="20261028_PM_disclosures_backfill_2")
    result = build_disclosures([first.root, second.root], today=TODAY)
    assert result.conflicts.field.tolist() == ["report_nm"]
    assert set(result.conflicts.iloc[0]["values"]) == {"이름1", "이름2"}


def test_rm_and_corp_cls_absent(tmp_path, client_factory):
    result, archive = built(tmp_path, client_factory, [item(rm="철", corp_cls="E")])
    extended = build_disclosures([archive.root], schema_mode="extended", today=TODAY)
    assert len(result.df) == 1
    for frame in (result.df, extended.df):
        assert "rm" not in frame and "corp_cls" not in frame


def test_build_idempotent(tmp_path, client_factory):
    result, archive = built(tmp_path, client_factory)
    again = build_disclosures([archive.root], today=TODAY)
    pd.testing.assert_frame_equal(result.df, again.df)
    assert result.quality == again.quality


def test_invalid_calendar_dates_removed(tmp_path, client_factory):
    result, _ = built(tmp_path, client_factory, [item(rcept_dt="20250230")])
    assert result.df.empty and result.quality["invalid_rcept_dt"] == 1


def test_refuse_on_gap(tmp_path, client_factory, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, _ = built(tmp_path, client_factory)
    with pytest.raises(PublishRefused, match="P1"):
        publish_disclosures(result)
    assert not Path("data/processed/disclosures.parquet").exists()


def continuous_result(tmp_path, client_factory, records=None):
    return built(tmp_path, client_factory, records, start=date(2010, 1, 1), end=date(2010, 3, 31))


class PreparedIO:
    """공용 I/O가 준비돼도 미합의 데이터가 쓰이지 않는지 검증하는 대역."""
    def __init__(self, old=None):
        self.old = old
        self.calls = []

    def read_table(self, name):
        self.calls.append("read")
        return self.old

    def validate(self, name, df):
        self.calls.append("validate")

    def write_table(self, name, df):
        pytest.fail("미합의 진단 결과를 공용 경로에 쓰려고 했습니다")


def test_refuse_on_conflict(tmp_path, client_factory, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, _ = continuous_result(tmp_path, client_factory)
    result.conflicts = pd.DataFrame([{"rcept_no": "bad"}])
    with pytest.raises(PublishRefused, match="P2"):
        publish_disclosures(result)


def test_refuse_on_shrink_without_reason(tmp_path, client_factory, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, _ = continuous_result(tmp_path, client_factory, [item(1, day="20100101"), item(2, day="20100102")])
    io = PreparedIO(result.df)
    monkeypatch.setattr(mod, "_common_io", lambda: io)
    target = Path("data/processed/disclosures.parquet")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing file")
    smaller = replace(result, df=result.df.iloc[:1].copy())
    with pytest.raises(PublishRefused, match="P3"):
        publish_disclosures(smaller)
    with pytest.raises(PublishRefused, match="DEC-09"):
        publish_disclosures(smaller, allow_shrink_reason="승인된 오류 행 제거")
    assert target.read_bytes() == b"existing file"


class MissingTableIO(PreparedIO):
    """공용 표가 아직 없을 때 FileNotFoundError를 내는 대역 (D-02 합의)."""
    def read_table(self, name):
        self.calls.append("read")
        raise FileNotFoundError(name)


def test_shrink_check_uses_common_io_not_cwd(tmp_path, client_factory, monkeypatch):
    # 현재 폴더에 파일이 없어도 공용 I/O가 기존 표를 돌려주면 행 수 감소를 막는다.
    monkeypatch.chdir(tmp_path)
    result, _ = continuous_result(tmp_path, client_factory, [item(1, day="20100101"), item(2, day="20100102")])
    monkeypatch.setattr(mod, "_common_io", lambda: PreparedIO(result.df))
    assert not Path("data/processed/disclosures.parquet").exists()
    smaller = replace(result, df=result.df.iloc[:1].copy())
    with pytest.raises(PublishRefused, match="P3"):
        publish_disclosures(smaller)


def test_missing_public_table_skips_shrink_check(tmp_path, client_factory, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, _ = continuous_result(tmp_path, client_factory)
    io = MissingTableIO()
    monkeypatch.setattr(mod, "_common_io", lambda: io)
    with pytest.raises(PublishRefused, match="DEC-09"):
        publish_disclosures(result)
    assert io.calls == ["read", "validate"]


@pytest.mark.parametrize("dry_run", [False, True])
def test_missing_common_io_never_reads_or_writes_public_file(tmp_path, client_factory, monkeypatch, dry_run):
    monkeypatch.chdir(tmp_path)
    result, _ = continuous_result(tmp_path, client_factory)
    target = Path("data/processed/disclosures.parquet")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing file")
    monkeypatch.setattr(mod, "_common_io", lambda: None)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("공용 I/O 우회 읽기"))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda *a, **k: pytest.fail("공용 I/O 우회 쓰기"))
    with pytest.raises(PublishRefused, match="R-18"):
        publish_disclosures(result, dry_run=dry_run)
    assert target.read_bytes() == b"existing file"
    assert not target.with_name(target.name + ".tmp").exists()


def test_p1_accepts_provisional_tail_and_refuses_gap(tmp_path, client_factory):
    result, _ = continuous_result(tmp_path, client_factory)
    # 기간을 크게 묶은 이력 + 일간 두 실행의 기간 합집합만 검사한다.
    result.windows = pd.DataFrame([
        {"pblntf_ty": "B", "bgn": "2010-01-01", "end": "2026-10-26", "state": "complete"},
        {"pblntf_ty": "B", "bgn": "2026-10-24", "end": "2026-10-26", "state": "complete"},
        {"pblntf_ty": "B", "bgn": "2026-10-27", "end": "2026-10-27", "state": "provisional"},
        {"pblntf_ty": "B", "bgn": "2026-10-25", "end": "2026-10-27", "state": "complete"},
        {"pblntf_ty": "B", "bgn": "2026-10-28", "end": "2026-10-28", "state": "provisional"}])
    mod._coverage_check(result)
    result.windows.loc[1:, ["bgn", "end"]] = [["2026-10-28", "2026-10-30"], ["2026-10-31", "2026-10-31"],
                                                           ["2026-10-29", "2026-10-31"], ["2026-11-01", "2026-11-01"]]
    with pytest.raises(PublishRefused, match="P1"):
        mod._coverage_check(result)


@pytest.mark.parametrize("schema_mode", ["v03", "extended"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_common_io_cannot_approve_unagreed_time_contract(tmp_path, client_factory, monkeypatch, schema_mode, dry_run):
    monkeypatch.chdir(tmp_path)
    _, archive = continuous_result(tmp_path, client_factory)
    result = build_disclosures([archive.root], schema_mode=schema_mode, today=TODAY)
    io = PreparedIO()
    monkeypatch.setattr(mod, "_common_io", lambda: io)
    with pytest.raises(PublishRefused, match="DEC-09"):
        publish_disclosures(result, dry_run=dry_run)
    assert io.calls == ["read", "validate"]
    assert not Path("data/processed/disclosures.parquet").exists()


def test_report_outputs_and_unique_name(tmp_path, client_factory, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, _ = continuous_result(tmp_path, client_factory)
    first = write_build_report(result)
    second = write_build_report(result)
    assert second.name == first.name + "_2"
    assert set(p.name for p in first.iterdir()) >= {"config.yaml", "git_commit.txt", "metrics.json", "tables", "summary.md"}
    assert pd.read_csv(first / "tables/gaps.csv").empty
    assert "사후 정보" in (first / "tables/rm_values.csv").read_text(encoding="utf-8")
    metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["rows_total"] == 1


def test_keyword_counts_by_type():
    frame = pd.DataFrame([
        {"rcept_no": "1", "pblntf_ty": "B", "report_nm": "유상ㆍ증자결정"},
        {"rcept_no": "2", "pblntf_ty": "E", "report_nm": "[기재정정] 유상증자결정"},
        {"rcept_no": "3", "pblntf_ty": "E", "report_nm": "다른 문서"}])
    config = {"features": {"event_keywords": {"ev_rights": ["유상증자결정"]}}, "events": {"triggers": {}}}
    counts = coverage_counts(frame, config).set_index("pblntf_ty")
    assert counts.loc["B", "count"] == counts.loc["E", "count"] == 1
    assert counts.loc["E", "ratio_to_BI"] == 1
    assert json.loads(counts.loc["E", "examples"])[0]["rcept_no"] == "2"


def test_coverage_workflow_reuses_backfill(tmp_path, client_factory, monkeypatch):
    import yaml
    monkeypatch.chdir(tmp_path)
    _, reference = built(tmp_path, client_factory, [item(day="20200101")],
                          start=date(2020, 1, 1), end=date(2020, 3, 31), types=("B", "I"))
    config = {"features": {"event_keywords": {"ev": ["유상증자결정"]}}, "events": {"triggers": {}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    client, session, _ = client_factory(lambda key, count: page([item(day="20200101")]))
    archive = new_archive("coverage", data_raw=tmp_path / "data/raw", today=TODAY)
    result, report = run_coverage(client=client, archive=archive, today=TODAY, quarters=["2020Q1"],
                                 types=["E"], config_path=path, data_raw=tmp_path / "data/raw")
    assert result.status == "complete"
    assert [r[2] for r in session.requests] == ["E"]
    assert not pd.read_csv(report / "coverage.csv").empty
    assert archive.load_manifest()["mode"] == "coverage"
    assert build_disclosures([archive.root], today=TODAY).df.empty


@pytest.mark.parametrize("missing", ["read_table", "validate", "write_table"])
def test_incomplete_common_io_contract_refused(tmp_path, client_factory, monkeypatch, missing):
    monkeypatch.chdir(tmp_path)
    result, _ = continuous_result(tmp_path, client_factory)
    io = PreparedIO()
    setattr(io, missing, None)
    monkeypatch.setattr(mod, "_common_io", lambda: io)
    with pytest.raises(PublishRefused, match="R-18"):
        publish_disclosures(result)
    assert not io.calls
    assert not Path("data/processed/disclosures.parquet").exists()


def test_common_io_import_failure_is_refusal(tmp_path, client_factory, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, _ = continuous_result(tmp_path, client_factory)
    def unavailable():
        raise ImportError("비공개 외부 라이브러리 예외 내용")
    monkeypatch.setattr(mod, "_common_io", unavailable)
    with pytest.raises(PublishRefused, match="R-18") as exc:
        publish_disclosures(result)
    assert "비공개" not in str(exc.value)


def test_common_schema_validation_failure_is_not_bypassed(tmp_path, client_factory, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result, _ = continuous_result(tmp_path, client_factory)
    io = PreparedIO()
    def reject(name, df):
        raise ValueError("민감한 상세 예외")
    io.validate = reject
    monkeypatch.setattr(mod, "_common_io", lambda: io)
    with pytest.raises(PublishRefused, match="P4") as exc:
        publish_disclosures(result)
    assert "민감한" not in str(exc.value)


@pytest.mark.parametrize("first_code,later_code", [("", "005930"), ("005930", "000660"), ("005930", "005930")])
def test_later_observed_mapping_never_backfills_past_ticker(tmp_path, client_factory, first_code, later_code):
    _, first = built(tmp_path / "first", client_factory, [item(stock_code=first_code)])
    _, second = built(tmp_path / "second", client_factory, [item(stock_code=later_code)], name="20261028_PM_disclosures_backfill_2")
    for path in second.root.rglob("*.meta.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["fetched_at"] = "2026-10-28T22:00:00+09:00"
        path.write_text(json.dumps(metadata), encoding="utf-8")
    result = build_disclosures([first.root, second.root], today=TODAY)
    assert result.df.ticker.tolist() == [None]
    assert result.observations.stock_code.tolist() == [first_code, later_code]
    assert result.observations.fetched_at.nunique() == 2
    assert result.quality["ticker_unverified_asof"] == 1
    assert result.quality.get("mutable_changed_stock_code", 0) == int(first_code != later_code)


@pytest.mark.parametrize("cutoff,estimate_visible", [
    ("2010-01-04T15:29:59+09:00", False),
    ("2010-01-04T15:30:00+09:00", False),
    ("2010-01-04T15:30:01+09:00", False),
    ("2010-01-04T23:59:59+09:00", True),
    ("2010-01-05T15:30:00+09:00", True),
])
def test_cutoff_diagnostics_never_reach_public_contract(tmp_path, client_factory, monkeypatch, cutoff, estimate_visible):
    monkeypatch.chdir(tmp_path)
    _, archive = continuous_result(tmp_path, client_factory, [item(day="20100104")])
    result = build_disclosures([archive.root], schema_mode="v03", today=TODAY)
    signal = pd.Timestamp(cutoff)
    estimated = result.df.rcept_dt.iloc[0]
    assert bool(estimated <= signal) == estimate_visible
    # 날짜만 비교하면 C가 지적한 같은 날 조기 포함 반례가 재현된다.
    assert estimated.date() <= signal.date()
    assert result.df.attrs["diagnostic_only"] is True
    assert result.df.attrs["actual_reception_times_available"] is False
    extended = build_disclosures([archive.root], schema_mode="extended", today=TODAY)
    assert extended.df.rcept_dt.isna().all()
    assert extended.df.available_at.iloc[0] == estimated
    monkeypatch.setattr(mod, "_common_io", lambda: PreparedIO())
    with pytest.raises(PublishRefused, match="DEC-09"):
        publish_disclosures(result)
    assert not Path("data/processed/disclosures.parquet").exists()
