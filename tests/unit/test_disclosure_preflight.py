"""공시 사전점검과 신호일 피처의 완료 판정을 같은 보관본으로 비교한다."""

import json

import pandas as pd
import pytest

from src.data import disclosure_preflight as preflight
from src.data.disclosure_archive import RawArchive
from src.events import alerts
from src.rag.evidence import signal_day_disclosures
from tests.rag.test_evidence import AFTER_CUTOFF, DAY, DAYS, T, archive_filings, inputs  # noqa: F401


@pytest.mark.parametrize(
    ("clock", "ready"),
    [("15:29:59", True), ("15:30:00", False), ("15:30:01", False)],
)
def test_cutoff_agrees_with_event_features(inputs, client_factory, capsys, clock, ready):
    timestamp = f"{T}T{clock}+09:00"
    run = archive_filings(client_factory, observed_at=f"{T}T12:00:00+09:00")
    archive = RawArchive(run.run_dir)
    manifest = archive.load_manifest()
    # 호출 간 대기와 무관하게 manifest의 실제 완료 시각 자체를 경계에 둔다.
    for window in manifest["windows"].values():
        if window["state"] == "complete":
            for attempt in window["attempts"]:
                attempt["finished_at"] = timestamp
    archive.save_manifest(manifest)
    code = preflight.main(["--date", T, "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == (0 if ready else 2)
    assert report["ready"] is ready
    assert report["required_slots"] == 38
    position = DAYS.get_loc(DAY)
    assert report["required_dates"] == DAYS[position - 19:position].strftime("%Y-%m-%d").tolist()

    tables, _ = inputs
    if ready:
        assert report["gaps"] == []
        assert signal_day_disclosures(T, tables["tickers"]).empty   # 확정된 공시 0건
    else:
        assert len(report["gaps"]) == 38
        assert {gap["status"] for gap in report["gaps"]} == {"late"}
        assert {gap["completed_at"] for gap in report["gaps"]} == {timestamp}
        with pytest.raises(alerts.AlertsNotCollected):
            signal_day_disclosures(T, tables["tickers"])


def test_report_distinguishes_incomplete_type_from_missing_type(inputs, client_factory):
    run = archive_filings(client_factory, observed_at=f"{T}T12:00:00+09:00", types=("B",))
    archive = RawArchive(run.run_dir)
    manifest = archive.load_manifest()
    for window in manifest["windows"].values():
        window["state"] = "provisional"
    archive.save_manifest(manifest)
    report = preflight.check_disclosure_inputs(DAY.date())
    assert len(report["gaps"]) == 38
    assert {gap["status"] for gap in report["gaps"] if gap["type"] == "B"} == {"incomplete"}
    assert {gap["status"] for gap in report["gaps"] if gap["type"] == "I"} == {"missing"}


def test_late_overlap_does_not_replace_timely_archive(inputs, client_factory):
    archive_filings(client_factory, observed_at=f"{T}T12:00:00+09:00")
    archive_filings(client_factory, day=DAY + pd.Timedelta(days=1), observed_at=AFTER_CUTOFF)
    report = preflight.check_disclosure_inputs(DAY.date())
    assert report["ready"]


def test_preflight_does_not_touch_archive_even_with_a_corrupt_page(inputs, client_factory, capsys):
    run = archive_filings(client_factory, observed_at=f"{T}T12:00:00+09:00")
    temporary_file = run.run_dir / "collecting.tmp"
    temporary_file.write_bytes(b"collection in progress")
    page = next(run.run_dir.rglob("page_0001.json"))
    page.write_bytes(b"corrupted")
    before = {path: path.read_bytes() for path in run.run_dir.rglob("*") if path.is_file()}

    assert preflight.main(["--date", T]) == 1
    assert "원본 페이지 손상 또는 누락" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in run.run_dir.rglob("*") if path.is_file()} == before
