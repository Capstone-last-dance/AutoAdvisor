"""응답 바이트의 불변 보관과 재개용 manifest 관리."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))


class ArchiveOverwriteError(FileExistsError):
    """이미 확정된 원본은 바꿀 수 없다."""


class ArchiveIntegrityError(ValueError):
    """원본 또는 메타 파일이 손상되었다."""


def _component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("경로 구성 요소는 ASCII 영숫자, 밑줄, 하이픈만 허용합니다")
    return value


def atomic_write(path: Path, body: bytes, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ArchiveOverwriteError(f"이미 보관된 파일: {path.name}")
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() and not overwrite:
        raise ArchiveOverwriteError(f"이미 보관된 파일: {path.name}")
    atomic_replace(temp, path)


def atomic_replace(source: Path, target: Path) -> None:
    """Windows 백신의 일시적인 파일 점유만 짧게 재시도한다."""
    for delay in (0.05, 0.1, 0.2, 0.4, None):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


class RawArchive:
    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def create(cls, data_raw: Path, run_id: str, run_date: date) -> "RawArchive":
        _component(run_id)
        run = Path(data_raw) / run_date.isoformat() / run_id
        run.mkdir(parents=True, exist_ok=False)
        root = run / "opendart_list"
        root.mkdir()
        return cls(root)

    @classmethod
    def open(cls, run_dir: Path) -> "RawArchive":
        root = Path(run_dir)
        if root.name != "opendart_list":
            root /= "opendart_list"
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError("수집 manifest를 찾을 수 없습니다")
        archive = cls(root)
        archive.cleanup_tmp()
        return archive

    def cleanup_tmp(self):
        for path in self.root.rglob("*.tmp"):
            path.unlink()
            self.append_event({"event": "leftover_tmp_removed", "file": str(path.relative_to(self.root))})

    def _attempt_dir(self, window_id: str, attempt: int) -> Path:
        if attempt < 1:
            raise ValueError("시도 번호는 1 이상이어야 합니다")
        return self.root / "windows" / _component(window_id) / f"attempt_{attempt:02d}"

    def next_attempt(self, window_id: str) -> int:
        parent = self.root / "windows" / _component(window_id)
        numbers = [int(p.name.split("_")[1]) for p in parent.glob("attempt_*")
                   if re.fullmatch(r"attempt_\d+", p.name)]
        return max(numbers, default=0) + 1

    def write_page(self, window_id: str, attempt: int, page) -> Path:
        path = self._attempt_dir(window_id, attempt) / f"page_{page.request.page_no:04d}.json"
        meta_path = path.with_suffix(".meta.json")
        if path.exists() or meta_path.exists():
            raise ArchiveOverwriteError(f"이미 보관된 페이지: {path.name}")
        metadata = {"request": page.request.params(), "pblntf_ty": page.request.pblntf_ty,
                    "fetched_at": page.fetched_at.isoformat(), "http_status": page.http_status,
                    "elapsed_ms": page.elapsed_ms, "sha256": hashlib.sha256(page.body).hexdigest(),
                    "total_count": page.total_count, "total_page": page.total_page,
                    "n_items": len(page.items)}
        atomic_write(path, page.body)
        atomic_write(meta_path, _json_bytes(metadata))
        return path

    def iter_pages(self, window_id: str, attempt: int):
        folder = self._attempt_dir(window_id, attempt)
        for path in sorted(folder.glob("page_*.json")):
            if not re.fullmatch(r"page_\d+\.json", path.name):
                continue
            meta_path = path.with_suffix(".meta.json")
            try:
                body = path.read_bytes()
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if hashlib.sha256(body).hexdigest() != meta["sha256"]:
                    raise ValueError("해시 불일치")
                payload = json.loads(body.decode("utf-8"))
                number = int(path.stem.split("_")[1])
                if int(meta["request"]["page_no"]) != number:
                    raise ValueError("메타 페이지 불일치")
                if payload.get("status") == "000" and int(payload["page_no"]) != number:
                    raise ValueError("본문 페이지 불일치")
            except (OSError, ValueError, KeyError, TypeError):
                self.append_event({"event": "invalid_page_pair", "window_id": window_id,
                                   "attempt": attempt, "file": path.name})
                continue
            yield payload, meta

    def fetched_page_numbers(self, window_id: str, attempt: int) -> set[int]:
        return {int(meta["request"]["page_no"]) for _, meta in self.iter_pages(window_id, attempt)}

    def load_manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    def save_manifest(self, manifest: dict) -> None:
        atomic_write(self.root / "manifest.json", _json_bytes(manifest), overwrite=True)

    def append_event(self, event: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        entry = {"at": datetime.now(KST).isoformat(), **event}
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
