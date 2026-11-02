"""원본 보관의 불변성, 체크섬과 중단 복구 검증."""

import json
from datetime import date

import pytest

from tests.conftest import item, page
from src.data.dart_client import ListRequest
from src.data.disclosure_archive import ArchiveOverwriteError, RawArchive


@pytest.fixture
def archive(tmp_path):
    return RawArchive.create(tmp_path, "20261028_PM_disclosures_backfill", date(2026, 10, 28))


def saved_page(client_factory):
    client, _, _ = client_factory([page([item()])])
    return client.fetch_list_page(ListRequest("20250101", "20250331", "B", 1))


def test_atomic_write_no_overwrite(archive, client_factory):
    response = saved_page(client_factory)
    path = archive.write_page("B_20250101_20250331", 1, response)
    with pytest.raises(ArchiveOverwriteError):
        archive.write_page("B_20250101_20250331", 1, response)
    assert path.read_bytes() == response.body
    assert not list(archive.root.rglob("*.tmp"))


def test_meta_has_sha256_and_type_without_key(archive, client_factory):
    path = archive.write_page("B_20250101_20250331", 1, saved_page(client_factory))
    metadata = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert len(metadata["sha256"]) == 64
    assert metadata["pblntf_ty"] == "B"
    assert "crtfc_key" not in json.dumps(metadata)


def test_leftover_tmp_ignored_on_resume(archive):
    archive.save_manifest({"status": "incomplete"})
    path = archive.root / "windows/B_20250101_20250331/attempt_01/page_0001.json.tmp"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partial")
    opened = RawArchive.open(archive.root)
    assert opened.fetched_page_numbers("B_20250101_20250331", 1) == set()
    assert not path.exists()
    assert "leftover_tmp_removed" in (archive.root / "events.jsonl").read_text()


def test_manifest_roundtrip(archive):
    manifest = {"status": "planned", "한글": "보존", "windows": {"I": {}, "B": {}}}
    archive.save_manifest(manifest)
    assert archive.load_manifest() == manifest
    manifest["status"] = "complete"
    archive.save_manifest(manifest)
    assert archive.load_manifest() == manifest
    assert list(archive.load_manifest()["windows"]) == ["I", "B"]


def test_checksum_mismatch_is_not_fetched(archive, client_factory):
    path = archive.write_page("B_20250101_20250331", 1, saved_page(client_factory))
    path.write_bytes(b"tampered")
    assert not archive.fetched_page_numbers("B_20250101_20250331", 1)
    assert "invalid_page_pair" in (archive.root / "events.jsonl").read_text()


def test_path_traversal_refused(tmp_path):
    with pytest.raises(ValueError):
        RawArchive.create(tmp_path, "../outside", date(2026, 10, 28))
