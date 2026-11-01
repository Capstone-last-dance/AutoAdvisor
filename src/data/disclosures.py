"""PM-04 공시 수집, 재개, 정제, 저장과 품질 리포트."""

from __future__ import annotations

import calendar
import importlib
import json
import logging
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd

from .dart_client import (KST, PAGE_COUNT, CallBudget, DailyBudgetExceeded, DartAuthError,
                          DartClient, DartRateLimitError, DartRequestError, DartUnavailableError,
                          ListRequest)
from .disclosure_archive import RawArchive, atomic_replace, atomic_write

LOG = logging.getLogger(__name__)
HISTORY_START = date(2010, 1, 1)
DATE_ONLY_AVAILABLE_TIME = time(23, 59, 59)
V03_COLUMNS = ["rcept_no", "rcept_dt", "ticker", "corp_name", "report_nm", "report_nm_norm"]
EXTENDED_COLUMNS = ["rcept_no", "rcept_date", "rcept_dt", "available_at", "ticker", "corp_code",
                    "corp_name", "flr_nm", "report_nm", "report_nm_norm"]
IMMUTABLE = ("rcept_dt", "report_nm", "corp_code", "flr_nm")
MUTABLE = ("corp_name", "stock_code", "corp_cls", "rm")


@dataclass(frozen=True)
class Window:
    pblntf_ty: str
    bgn: date
    end: date

    @property
    def window_id(self):
        return f"{self.pblntf_ty}_{self.bgn:%Y%m%d}_{self.end:%Y%m%d}"

    def is_open(self, today: date):
        return self.end >= today


@dataclass
class PlanSummary:
    windows: int
    expected_rows: int
    expected_calls: int
    estimated_hours: float


@dataclass
class CollectionResult:
    run_dir: Path
    status: str
    stop_reason: str | None
    windows_by_state: dict[str, int]
    calls: int


@dataclass
class BuildResult:
    df: pd.DataFrame
    conflicts: pd.DataFrame
    quality: dict[str, int]
    windows: pd.DataFrame
    runs: list[str]
    # 공개 필드 외의 원본 맥락은 리포트와 저장 조건 검사에만 사용한다.
    observations: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    schema_mode: str = "v03"
    types: tuple[str, ...] = ("B", "I")
    today: date = field(default_factory=lambda: datetime.now(KST).date())
    calls_total: int = 0
    report_dir: Path | None = None
    allow_shrink_reason: str | None = None


class PublishRefused(ValueError):
    """완결성, 충돌, 축소 또는 스키마 조건 때문에 저장이 거부됨."""


def make_windows(start: date, end: date, types: Sequence[str], *, months: int = 3) -> list[Window]:
    if months not in (1, 3) or start > end or not types or len(set(types)) != len(types):
        raise ValueError("기간, 유형 목록 또는 창 크기가 잘못되었습니다 (1/3개월만 허용)")
    if any(t not in tuple("ABCDEFGHIJ") for t in types):
        raise ValueError("공시유형은 A~J만 허용합니다")
    windows = []
    for ty in types:
        current = start
        while current <= end:
            last_month = ((current.month - 1) // months + 1) * months
            bound = date(current.year, last_month, calendar.monthrange(current.year, last_month)[1])
            stop = min(end, bound)
            windows.append(Window(ty, current, stop))
            current = stop + timedelta(days=1)
    return windows


def code_version() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True,
                                    capture_output=True, text=True).stdout.strip())
        return {"code_commit": commit, "code_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"code_commit": "unknown", "code_dirty": "unknown"}


def _entry(window: Window, today: date) -> dict:
    return {"pblntf_ty": window.pblntf_ty, "bgn": window.bgn.isoformat(), "end": window.end.isoformat(),
            "open": window.is_open(today), "state": "pending", "expected_total": None,
            "expected_pages": None, "attempts": [], "accepted_attempts": [],
            "accepted_unique_rcept_no": 0, "children": []}


def _init_manifest(archive, windows, client, today, mode, months=3):
    if (archive.root / "manifest.json").exists():
        manifest = archive.load_manifest()
        existing = manifest.get("calls_by_date", {})
        # 같은 인스턴스로 plan→collect를 실행해도 이중 차감하지 않는다.
        used = client.budget.used_by_date
        client.budget = CallBudget(client.budget.limit_per_day,
                                  {key: max(existing.get(key, 0), used.get(key, 0))
                                   for key in set(existing) | set(used)})
        if manifest.get("stop_reason") == "rate_limit":
            # 프로세스를 다시 열어도 서버가 제한한 당일에는 호출하지 않는다.
            blocked = datetime.fromisoformat(manifest["updated_at"]).astimezone(KST).date()
            client._rate_limited_date = blocked
        return manifest
    stamp = client.now().isoformat()
    manifest = {"schema_version": 1, "run_id": archive.root.parent.name, "mode": mode,
                "created_at": stamp, "updated_at": stamp, **code_version(),
                "params": {"start": min(w.bgn for w in windows).isoformat(),
                           "end": max(w.end for w in windows).isoformat(),
                           "types": list(dict.fromkeys(w.pblntf_ty for w in windows)),
                           "window_months": months, "page_count": PAGE_COUNT, "last_reprt_at": "N",
                           "min_interval_sec": client.min_interval,
                           "daily_call_budget": client.budget.limit_per_day, "max_attempts": 3},
                "calls_by_date": client.budget.used_by_date, "status": "planned", "stop_reason": None,
                "windows": {w.window_id: _entry(w, today) for w in windows}}
    archive.save_manifest(manifest)
    return manifest


def _save(archive, manifest, client):
    manifest["updated_at"] = client.now().isoformat()
    manifest["calls_by_date"] = client.budget.used_by_date
    archive.save_manifest(manifest)


def _stop(archive, manifest, client, exc):
    mapping = {DartRateLimitError: "rate_limit", DailyBudgetExceeded: "daily_budget",
               DartUnavailableError: "unavailable", KeyboardInterrupt: "interrupted",
               DartAuthError: "auth", DartRequestError: "request_error"}
    manifest["stop_reason"] = next(v for k, v in mapping.items() if isinstance(exc, k))
    manifest["status"] = "failed" if isinstance(exc, (DartAuthError, DartRequestError)) else "incomplete"
    archive.append_event({"event": "stopped", "reason": manifest["stop_reason"]})
    _save(archive, manifest, client)


STOP_ERRORS = (DartRateLimitError, DailyBudgetExceeded, DartUnavailableError,
               KeyboardInterrupt, DartAuthError, DartRequestError)


def _result(archive, manifest, client):
    return CollectionResult(archive.root, manifest["status"], manifest["stop_reason"],
                            dict(Counter(w["state"] for w in manifest["windows"].values())), client.calls)


def new_archive(mode: str, *, data_raw=Path("data/raw"), today=None, run_id=None):
    today = today or datetime.now(KST).date()
    stem = run_id or f"{today:%Y%m%d}_PM_disclosures_{mode}"
    number = 1
    while True:
        candidate = stem if number == 1 else f"{stem}_{number}"
        try:
            return RawArchive.create(Path(data_raw), candidate, today)
        except FileExistsError:
            if run_id:
                raise
            number += 1


def _new_attempt(archive, key, entry, client):
    n = archive.next_attempt(key)
    # 폴더를 먼저 만들어 중단되어도 번호가 재사용되지 않게 한다.
    archive._attempt_dir(key, n).mkdir(parents=True, exist_ok=False)
    entry["attempts"].append({"n": n, "started_at": client.now().isoformat(), "finished_at": None,
                              "pages": 0, "unique_rcept_no": 0, "total_counts_seen": [], "result": None})
    return entry["attempts"][-1]


def _fetch(archive, key, attempt, entry, page_no, client):
    req = ListRequest(entry["bgn"].replace("-", ""), entry["end"].replace("-", ""),
                      entry["pblntf_ty"], page_no)
    page = client.fetch_list_page(req)
    archive.write_page(key, attempt["n"], page)
    for item in page.items:
        dt = str(item.get("rcept_dt", ""))
        if re.fullmatch(r"\d{8}", dt) and not req.bgn_de <= dt <= req.end_de:
            archive.append_event({"event": "rcept_dt_outside_window", "window_id": key,
                                  "attempt": attempt["n"], "page_no": page_no,
                                  "rcept_no": str(item.get("rcept_no", ""))})
    return page


def _inspect(archive, key, n):
    return list(archive.iter_pages(key, n))


def _valid_ids(records):
    return {str(item.get("rcept_no", "")).strip() for body, _ in records for item in body.get("list", [])
            if re.fullmatch(r"\d{14}", str(item.get("rcept_no", "")).strip())}


def _finish_attempt(archive, key, entry, attempt, today, client):
    records = _inspect(archive, key, attempt["n"])
    totals = [meta["total_count"] for _, meta in records]
    attempt.update(pages=len(records), unique_rcept_no=len(_valid_ids(records)),
                   total_counts_seen=list(dict.fromkeys(totals)), finished_at=client.now().isoformat())
    if not totals:
        attempt["result"] = "corrupt"
        entry["state"] = "mismatch"
        return
    if not entry["open"] and len(set(totals)) > 1:
        attempt["result"] = "changed"
        entry["state"] = "mismatch"
        return
    target = totals[-1] if entry["open"] else totals[0]
    # 닫힌 창의 기준 건수가 바뀌면 옛 기준의 시도는 합집합에 넣지 않는다.
    if not entry["open"]:
        entry["accepted_attempts"] = [n for n in entry["accepted_attempts"]
                                      if all(m["total_count"] == target for _, m in _inspect(archive, key, n))]
    if attempt["n"] not in entry["accepted_attempts"]:
        entry["accepted_attempts"].append(attempt["n"])
    union = set()
    for n in entry["accepted_attempts"]:
        union.update(_valid_ids(_inspect(archive, key, n)))
    entry["expected_total"] = target
    entry["expected_pages"] = records[-1][1]["total_page"] if entry["open"] else records[0][1]["total_page"]
    entry["accepted_unique_rcept_no"] = len(union)
    attempt["result"] = "ok" if len(union) == target else "mismatch"
    if len(union) > target:
        entry["state"] = "failed"
    elif len(union) == target:
        entry["state"] = "provisional" if entry["open"] else "complete"
    else:
        entry["state"] = "mismatch"


def plan_run(start: date, end: date, *, types=("B", "I"), months=3,
             client: DartClient, archive: RawArchive, today: date) -> PlanSummary:
    windows = make_windows(start, end, types, months=months)
    mode = "coverage" if "_coverage" in archive.root.parent.name else "backfill"
    manifest = _init_manifest(archive, windows, client, today, mode, months)
    if (manifest["params"]["start"], manifest["params"]["end"], manifest["params"]["types"]) != (
            start.isoformat(), end.isoformat(), list(types)):
        raise ValueError("기존 실행과 계획 인자가 다릅니다")
    try:
        for key, entry in manifest["windows"].items():
            if entry["state"] != "pending":
                continue
            attempt = entry["attempts"][-1] if entry["attempts"] else _new_attempt(archive, key, entry, client)
            records = _inspect(archive, key, attempt["n"])
            folder = archive._attempt_dir(key, attempt["n"])
            if not records and list(folder.glob("page_*.json")):
                attempt["result"] = "corrupt"
                attempt = _new_attempt(archive, key, entry, client)
            if not records:
                _fetch(archive, key, attempt, entry, 1, client)
                records = _inspect(archive, key, attempt["n"])
            meta = records[0][1]
            entry.update(expected_total=meta["total_count"], expected_pages=meta["total_page"], state="planned")
            if meta["total_page"] <= 1:
                _finish_attempt(archive, key, entry, attempt, today, client)
            _save(archive, manifest, client)
            LOG.info("%s pages=%s rows=%s state=%s", key, entry["expected_pages"],
                     entry["expected_total"], entry["state"])
    except STOP_ERRORS as exc:
        _stop(archive, manifest, client, exc)
    else:
        manifest["status"], manifest["stop_reason"] = "planned", None
        _save(archive, manifest, client)
    entries = list(manifest["windows"].values())
    elapsed = [m["elapsed_ms"] / 1000 for key, w in manifest["windows"].items()
               for a in w["attempts"] for _, m in _inspect(archive, key, a["n"])]
    calls = sum(max(1, w["expected_pages"] or 0) for w in entries)
    return PlanSummary(len(entries), sum(w["expected_total"] or 0 for w in entries), calls,
                       calls * max(client.min_interval, sum(elapsed) / len(elapsed) if elapsed else 0) / 3600)


def _collect_window(key, manifest, archive, client, today, max_attempts, split_on_failure):
    entry = manifest["windows"][key]
    if entry["state"] in {"complete", "split", "failed"}:
        return
    was_provisional = entry["state"] == "provisional"
    entry["open"] = date.fromisoformat(entry["end"]) >= today
    if was_provisional:
        # 같은 실행의 다음 일간 호출도 새 스냅샷이며 옛 원본을 보존한다.
        attempt = _new_attempt(archive, key, entry, client)
        entry["accepted_attempts"] = []
    elif entry["attempts"] and entry["attempts"][-1]["result"] is None:
        attempt = entry["attempts"][-1]
    else:
        attempt = None
    while True:
        if attempt is None:
            if len(entry["attempts"]) >= max_attempts:
                break
            attempt = _new_attempt(archive, key, entry, client)
        entry["state"] = "collecting"
        _save(archive, manifest, client)
        records = _inspect(archive, key, attempt["n"])
        found = {int(m["request"]["page_no"]) for _, m in records}
        folder = archive._attempt_dir(key, attempt["n"])
        disk_numbers = {int(p.name.split("_")[1].split(".")[0]) for p in folder.glob("page_*.json")}
        if disk_numbers - found:
            attempt.update(result="corrupt", finished_at=client.now().isoformat())
            archive.append_event({"event": "new_attempt_after_corruption", "window_id": key})
            attempt = None
            continue
        if not records:
            _fetch(archive, key, attempt, entry, 1, client)
            records = _inspect(archive, key, attempt["n"])
            found = {1}
        first = records[0][1]
        target, pages = first["total_count"], first["total_page"]
        if entry["open"]:
            pages = records[-1][1]["total_page"]
        entry.update(expected_total=target, expected_pages=pages)
        changed = not entry["open"] and any(m["total_count"] != target for _, m in records)
        page_no = 1
        while page_no <= pages and not changed:
            if page_no not in found:
                page = _fetch(archive, key, attempt, entry, page_no, client)
                if entry["open"]:
                    pages = page.total_page
                elif page.total_count != target:
                    changed = True
                if page_no % 20 == 0:
                    _save(archive, manifest, client)
            page_no += 1
        _finish_attempt(archive, key, entry, attempt, today, client)
        archive.append_event({"event": "window_state", "window_id": key,
                              "attempt": attempt["n"], "state": entry["state"]})
        _save(archive, manifest, client)
        LOG.info("%s pages=%s rows=%s state=%s", key, attempt["pages"],
                 entry["accepted_unique_rcept_no"], entry["state"])
        if entry["state"] in {"complete", "provisional", "failed"}:
            return
        attempt = None
    children = make_windows(date.fromisoformat(entry["bgn"]), date.fromisoformat(entry["end"]),
                            [entry["pblntf_ty"]], months=1)
    if split_on_failure and len(children) > 1:
        entry["children"] = [w.window_id for w in children]
        entry["state"] = "split"
        for w in children:
            manifest["windows"].setdefault(w.window_id, _entry(w, today))
    else:
        entry["state"] = "failed"
    _save(archive, manifest, client)


def collect_run(*, client: DartClient, archive: RawArchive, today: date,
                max_attempts: int = 3, split_on_failure: bool = True) -> CollectionResult:
    if max_attempts < 1:
        raise ValueError("최대 시도 수는 1 이상이어야 합니다")
    archive.cleanup_tmp()
    manifest = archive.load_manifest()
    _init_manifest(archive, [], client, today, manifest["mode"])
    manifest.update(status="collecting", stop_reason=None)
    manifest["params"]["max_attempts"] = max_attempts
    _save(archive, manifest, client)
    try:
        offset = 0
        while offset < len(manifest["windows"]):
            key = list(manifest["windows"])[offset]
            _collect_window(key, manifest, archive, client, today, max_attempts, split_on_failure)
            offset += 1
    except STOP_ERRORS as exc:
        _stop(archive, manifest, client, exc)
    else:
        states = {w["state"] for w in manifest["windows"].values()}
        manifest["status"] = "failed" if "failed" in states else (
            "complete" if states <= {"complete", "provisional", "split"} else "incomplete")
        _save(archive, manifest, client)
    return _result(archive, manifest, client)


def collect_daily(day: date, *, lookback_days: int = 3, types=("B", "I"),
                  client: DartClient | None = None, data_raw: Path = Path("data/raw")) -> CollectionResult:
    if lookback_days < 1:
        raise ValueError("lookback_days는 1 이상이어야 합니다")
    client = client if client is not None else DartClient()
    today = client.now().astimezone(KST).date()
    if day > today:
        raise ValueError("미래 일자의 일간 수집은 허용하지 않습니다")
    run_id = f"{day:%Y%m%d}_PM_disclosures_daily"
    matches = list(Path(data_raw).glob(f"*/{run_id}/opendart_list/manifest.json"))
    if len(matches) > 1:
        raise ValueError("같은 일간 run_id가 여러 개입니다")
    if matches:
        archive = RawArchive.open(matches[0].parent)
        manifest = archive.load_manifest()
        if manifest["params"].get("lookback_days") != lookback_days or manifest["params"]["types"] != list(types):
            raise ValueError("일간 실행의 유형 또는 lookback을 바꿀 수 없습니다")
    else:
        archive = new_archive("daily", data_raw=data_raw, today=today, run_id=run_id)
        windows = []
        for ty in types:
            windows.extend(make_windows(day - timedelta(days=lookback_days), day - timedelta(days=1), [ty]))
            windows.append(Window(ty, day, day))
        manifest = _init_manifest(archive, windows, client, today, "daily")
        manifest["params"]["lookback_days"] = lookback_days
        archive.save_manifest(manifest)
    return collect_run(client=client, archive=archive, today=today)


def normalize_report_nm(name: str) -> str:
    return re.sub(r"\s+", "", name).replace("\u318d", "").replace("\u00b7", "")


def _string(value):
    return str(value).strip() if value is not None else ""


def _read_observations(run_dirs, *, include_coverage=False):
    observations, windows, runs = [], [], []
    types, calls = [], 0
    for directory in sorted(set(Path(p) for p in run_dirs), key=str):
        archive = RawArchive.open(directory)
        manifest = archive.load_manifest()
        if manifest["mode"] == "coverage" and not include_coverage:
            continue
        run_id = manifest["run_id"]
        runs.append(run_id)
        types.extend(manifest["params"]["types"])
        calls += sum(manifest["calls_by_date"].values())
        for key, window in manifest["windows"].items():
            windows.append({"run_id": run_id, "window_id": key, **window})
            if window["state"] not in {"complete", "provisional"}:
                continue
            for n in window["accepted_attempts"]:
                records = _inspect(archive, key, n)
                attempt = next((a for a in window["attempts"] if a["n"] == n), None)
                if not records or (attempt and len(records) != attempt["pages"]):
                    raise PublishRefused(f"원본 페이지 손상 또는 누락: {run_id}/{key}")
                for body, meta in records:
                    for item in body.get("list", []):
                        row = {field: _string(item.get(field))
                               for field in ("rcept_no", *IMMUTABLE, *MUTABLE)}
                        row.update(pblntf_ty=meta["pblntf_ty"], run_id=run_id, window_id=key,
                                   fetched_at=meta["fetched_at"], window_bgn=window["bgn"], window_end=window["end"])
                        observations.append(row)
    return observations, windows, runs, tuple(dict.fromkeys(types)), calls


def build_disclosures(run_dirs: Sequence[Path] | None = None, *, schema_mode="v03", today: date) -> BuildResult:
    if schema_mode not in ("v03", "extended"):
        raise ValueError("스키마 모드는 v03 또는 extended여야 합니다")
    if run_dirs is None:
        run_dirs = [p.parent for p in Path("data/raw").glob("*/*_PM_disclosures_*/opendart_list/manifest.json")]
    raw, windows, runs, types, calls = _read_observations(run_dirs)
    quality = Counter({"invalid_rcept_no": 0, "invalid_rcept_dt": 0, "invalid_stock_code": 0,
                       "rcept_dt_outside_window": 0, "rcept_no_date_differs": 0, "multi_type": 0,
                       "similar_middle_dots": 0})
    groups = {}
    for row in raw:
        if not re.fullmatch(r"\d{14}", row["rcept_no"]):
            quality["invalid_rcept_no"] += 1
            continue
        if not re.fullmatch(r"\d{8}", row["rcept_dt"]):
            quality["invalid_rcept_dt"] += 1
            continue
        try:
            datetime.strptime(row["rcept_dt"], "%Y%m%d")
            datetime.strptime(row["rcept_no"][:8], "%Y%m%d")
        except ValueError:
            quality["invalid_rcept_dt"] += 1
            continue
        if not row["window_bgn"].replace("-", "") <= row["rcept_dt"] <= row["window_end"].replace("-", ""):
            quality["rcept_dt_outside_window"] += 1
        groups.setdefault(row["rcept_no"], []).append(row)
    rows, conflicts = [], []
    for number, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: (datetime.fromisoformat(r["fetched_at"]),
                                                r["run_id"], r["window_id"]))
        latest = ordered[-1]
        for field_name in IMMUTABLE:
            values = sorted({r[field_name] for r in group})
            if len(values) > 1:
                conflicts.append({"rcept_no": number, "field": field_name,
                                  "values": values, "run_ids": sorted({r["run_id"] for r in group})})
        for field_name in MUTABLE:
            if len({r[field_name] for r in group}) > 1:
                quality[f"mutable_changed_{field_name}"] += 1
        if len({r["pblntf_ty"] for r in group}) > 1:
            quality["multi_type"] += 1
        ticker = latest["stock_code"] or None
        if ticker is not None and not re.fullmatch(r"[0-9A-Z]{6}", ticker):
            ticker = None
            quality["invalid_stock_code"] += 1
        rcept_date = datetime.strptime(latest["rcept_dt"], "%Y%m%d").date()
        number_date = datetime.strptime(number[:8], "%Y%m%d").date()
        available = datetime.combine(max(rcept_date, number_date), DATE_ONLY_AVAILABLE_TIME, tzinfo=KST)
        quality["rcept_no_date_differs"] += rcept_date != number_date
        prefix = re.match(r"^\[[^\]]+\]", latest["report_nm"])
        if prefix:
            quality[f"prefix:{prefix.group()}"] += 1
        quality["similar_middle_dots"] += any(c in latest["report_nm"] for c in "・∙•･")
        rows.append({"rcept_no": number, "rcept_date": pd.Timestamp(rcept_date),
                     "rcept_dt": available if schema_mode == "v03" else pd.NaT,
                     "available_at": available, "ticker": ticker, "corp_code": latest["corp_code"],
                     "corp_name": latest["corp_name"], "flr_nm": latest["flr_nm"],
                     "report_nm": latest["report_nm"], "report_nm_norm": normalize_report_nm(latest["report_nm"])})
    all_columns = EXTENDED_COLUMNS
    frame = pd.DataFrame(rows, columns=all_columns)
    for column in ("rcept_dt", "available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert(KST).astype(pd.DatetimeTZDtype("ns", KST))
    frame["rcept_date"] = pd.to_datetime(frame["rcept_date"]).astype("datetime64[ns]")
    frame = frame.sort_values(["available_at", "rcept_no"]).reset_index(drop=True)
    frame = frame[V03_COLUMNS if schema_mode == "v03" else EXTENDED_COLUMNS]
    for column in frame.columns:
        if column not in {"rcept_date", "rcept_dt", "available_at"}:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    result = BuildResult(frame, pd.DataFrame(conflicts, columns=["rcept_no", "field", "values", "run_ids"]),
                         dict(quality), pd.DataFrame(windows), runs, pd.DataFrame(raw), schema_mode,
                         types or ("B", "I"), today, calls)
    LOG.info("정제 지표: %s", json.dumps(build_metrics(result), ensure_ascii=False))
    return result


def _interval_end(intervals, start):
    covered = start - timedelta(days=1)
    for bgn, end in sorted(intervals):
        if bgn > covered + timedelta(days=1):
            break
        covered = max(covered, end)
    return covered


def _coverage_check(result):
    if result.windows.empty:
        raise PublishRefused("P1: 수집 창이 없습니다")
    for ty in result.types:
        subset = result.windows[result.windows.pblntf_ty == ty]
        complete, provisional = [], []
        for window in subset.to_dict("records"):
            pair = date.fromisoformat(window["bgn"]), date.fromisoformat(window["end"])
            if window["state"] == "complete":
                complete.append(pair)
            elif window["state"] == "provisional":
                provisional.append(pair)
        last_complete = _interval_end(complete, HISTORY_START)
        # complete에 덮인 열린 스냅샷은 검사에서 대체된 것으로 본다.
        for bgn, end in complete:
            if end > last_complete:
                raise PublishRefused(f"P1: {ty} 닫힌 창 사이에 누락 기간이 있습니다")
        tail = [(max(bgn, last_complete + timedelta(days=1)), end)
                for bgn, end in provisional if end > last_complete]
        final = _interval_end(tail, last_complete + timedelta(days=1))
        requested_end = max(date.fromisoformat(value) for value in subset.end)
        if final < requested_end:
            raise PublishRefused(f"P1: {ty} 기간이 {final.isoformat()} 이후 이어지지 않습니다")


def _validate_internal(df, mode):
    columns = V03_COLUMNS if mode == "v03" else EXTENDED_COLUMNS
    if list(df.columns) != columns or df.rcept_no.duplicated().any():
        raise PublishRefused("P4: 열 목록 또는 고유 접수번호 규약 위반")
    if not df.rcept_no.map(lambda x: isinstance(x, str) and re.fullmatch(r"\d{14}", x) is not None).all():
        raise PublishRefused("P4: 접수번호 형식 오류")
    for column in columns:
        if column in {"rcept_dt", "available_at"}:
            if str(df[column].dtype) != str(pd.DatetimeTZDtype("ns", KST)):
                raise PublishRefused("P4: KST datetime 자료형 오류")
            if column != "rcept_dt" or mode == "v03":
                if df[column].isna().any():
                    raise PublishRefused("P4: 이용 가능 시각 누락")
        elif column == "rcept_date":
            if str(df[column].dtype) != "datetime64[ns]":
                raise PublishRefused("P4: 날짜 자료형 오류")
        elif not df[column].map(lambda x: isinstance(x, str) or (column == "ticker" and x is None)).all():
            raise PublishRefused(f"P4: 문자열 자료형 오류: {column}")
    if not df.ticker.map(lambda x: x is None or re.fullmatch(r"[0-9A-Z]{6}", x) is not None).all():
        raise PublishRefused("P4: 종목코드 형식 오류")


def _common_io():
    try:
        return importlib.import_module("src.common.io")
    except ModuleNotFoundError as exc:
        if exc.name != "src.common.io":
            raise
        return None


def _write_table_temp(name, df):
    # TODO(D-02): io.write_table로 교체
    if name != "disclosures":
        raise ValueError("임시 저장기는 disclosures만 지원합니다")
    path = Path("data/processed/disclosures.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        df.to_parquet(temp, index=False)
        with temp.open("rb+") as stream:
            import os
            os.fsync(stream.fileno())
        import os
        atomic_replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def publish_disclosures(result: BuildResult, *, allow_shrink_reason=None, dry_run=False):
    _coverage_check(result)
    if not result.conflicts.empty:
        raise PublishRefused("P2: 불변 필드 충돌이 있습니다")
    target = Path("data/processed/disclosures.parquet")
    io = _common_io()
    if target.exists():
        old = io.read_table("disclosures") if io is not None else pd.read_parquet(target)
        if len(result.df) < len(old) and not (allow_shrink_reason and allow_shrink_reason.strip()):
            raise PublishRefused("P3: 기존 파일보다 행 수가 줄어듭니다")
    _validate_internal(result.df, result.schema_mode)
    if io is not None:
        try:
            io.validate("disclosures", result.df)
        except (ValueError, TypeError, KeyError) as exc:
            raise PublishRefused(f"P4: 공용 스키마 검증 실패 ({type(exc).__name__})") from None
    result.allow_shrink_reason = allow_shrink_reason
    if result.report_dir is not None:
        _write_build_report_files(result, result.report_dir)
    if dry_run:
        return None
    return io.write_table("disclosures", result.df) if io is not None else _write_table_temp("disclosures", result.df)


def build_metrics(result):
    frame, observations = result.df, result.observations
    dt = "rcept_dt" if result.schema_mode == "v03" else "available_at"
    return {"rows_total": len(frame), "rows_with_ticker": int(frame.ticker.notna().sum()),
            "unique_tickers": int(frame.ticker.nunique()),
            "rows_by_type": (observations.groupby("pblntf_ty").rcept_no.nunique().to_dict()
                             if not observations.empty else {}),
            "date_min": frame[dt].min().isoformat() if len(frame) else None,
            "date_max": frame[dt].max().isoformat() if len(frame) else None,
            "conflicts": len(result.conflicts), "quality": result.quality,
            "windows_by_state": dict(Counter(result.windows.state)) if not result.windows.empty else {},
            "calls_total": result.calls_total, "runs": result.runs,
            "allow_shrink_reason": result.allow_shrink_reason}


def _report_directory(kind, today, root=Path("reports")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{today:%Y%m%d}_PM_disclosures_{kind}"
    index = 1
    while True:
        folder = root / (stem if index == 1 else f"{stem}_{index}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            index += 1


def _csv(frame, path, comment=None):
    prefix = f"# {comment}\n" if comment else ""
    atomic_write(path, (prefix + frame.to_csv(index=False)).encode("utf-8"), overwrite=True)


def _write_build_report_files(result, folder):
    import yaml
    tables = folder / "tables"
    tables.mkdir(exist_ok=True)
    config = {"runs": result.runs, "schema_mode": result.schema_mode, "types": list(result.types),
              "date_only_available_time": DATE_ONLY_AVAILABLE_TIME.isoformat(),
              "available_at_policy": "max(rcept_dt_date,rcept_no_date) 23:59:59+09:00",
              "allow_shrink_reason": result.allow_shrink_reason}
    atomic_write(folder / "config.yaml", yaml.safe_dump(config, allow_unicode=True).encode("utf-8"), overwrite=True)
    version = code_version()
    atomic_write(folder / "git_commit.txt", (json.dumps(version) + "\n").encode(), overwrite=True)
    atomic_write(folder / "metrics.json", (json.dumps(build_metrics(result), ensure_ascii=False, indent=2) + "\n").encode("utf-8"), overwrite=True)
    _csv(result.windows, tables / "windows.csv")
    gaps = result.windows[~result.windows.state.isin(["complete", "split"])] if not result.windows.empty else pd.DataFrame(columns=["run_id", "window_id", "state"])
    # complete 기간에 대체된 provisional은 누락이 아니다.
    if not gaps.empty:
        keep = []
        for row in gaps.to_dict("records"):
            covered = result.windows[(result.windows.pblntf_ty == row["pblntf_ty"]) & (result.windows.state == "complete")]
            end = _interval_end([(date.fromisoformat(w.bgn), date.fromisoformat(w.end)) for w in covered.itertuples()], date.fromisoformat(row["bgn"]))
            keep.append(end < date.fromisoformat(row["end"]))
        gaps = gaps[keep]
    _csv(gaps, tables / "gaps.csv")
    _csv(result.conflicts, tables / "conflicts.csv")
    _csv(pd.DataFrame(sorted(result.quality.items()), columns=["metric", "count"]), tables / "quality.csv")
    prefixes = [(key[len("prefix:"):], count) for key, count in result.quality.items() if key.startswith("prefix:")]
    _csv(pd.DataFrame(sorted(prefixes, key=lambda p: (-p[1], p[0]))[:30], columns=["prefix", "count"]), tables / "prefixes.csv")
    raw = result.observations.copy()
    if not raw.empty:
        raw["year"] = raw.rcept_dt.str[:4]
        yearly = raw.drop_duplicates(["rcept_no", "pblntf_ty"]).groupby(["year", "pblntf_ty"]).size().reset_index(name="count")
        current = raw.sort_values("fetched_at").drop_duplicates("rcept_no", keep="last")
        corps = current.groupby(["year", "corp_cls"]).size().reset_index(name="count")
        rm = current.groupby("rm").size().reset_index(name="count")
    else:
        yearly = pd.DataFrame(columns=["year", "pblntf_ty", "count"])
        corps = pd.DataFrame(columns=["year", "corp_cls", "count"])
        rm = pd.DataFrame(columns=["rm", "count"])
    _csv(yearly, tables / "counts_by_year_type.csv")
    _csv(corps, tables / "counts_by_year_corpcls_current.csv", "현재 기업 분류. 생존 편향 방지를 위해 필터에 쓰지 않는다")
    _csv(rm, tables / "rm_values.csv", "사후 정보. 필터에 쓰지 않는다")
    try:
        _coverage_check(result)
        coverage = "통과"
    except PublishRefused as exc:
        coverage = str(exc)
    summary = (f"# PM-04 공시 정제 리포트\n\n행 수: {len(result.df)} / 충돌: {len(result.conflicts)}\n\n"
               f"기간 연속성: {coverage}\n\n- [x] 보고서명 정규화 적용\n"
               f"- [{'x' if not len(gaps) else ' '}] 완결되지 않은 기간 없음 (tables/gaps.csv)\n"
               "- [ ] 실제 접수 시각 확보: API는 날짜만 제공하므로 미충족 (DEC-09)\n\n"
               "v03의 rcept_dt는 실제 접수 시각이 아니라 두 날짜 중 늦은 날 23:59:59 KST의 이용 가능 시점이다.\n"
               f"\n행 수 축소 사유: {result.allow_shrink_reason or '없음'}\n")
    atomic_write(folder / "summary.md", summary.encode("utf-8"), overwrite=True)


def write_build_report(result, *, reports=Path("reports")):
    result.report_dir = _report_directory("build", result.today, reports)
    _write_build_report_files(result, result.report_dir)
    return result.report_dir


def coverage_counts(observations, config):
    rows = []
    keywords = {word for words in config["features"]["event_keywords"].values() for word in words}
    keywords.update(word for trigger in config["events"]["triggers"].values() for word in trigger["keywords"])
    frame = observations.copy()
    frame["report_nm_norm"] = frame.report_nm.map(normalize_report_nm)
    for keyword in sorted(keywords):
        matches = frame[frame.report_nm_norm.str.contains(keyword, regex=False)]
        base = matches[matches.pblntf_ty.isin(["B", "I"])].rcept_no.nunique()
        for ty in sorted(frame.pblntf_ty.unique()):
            subset = matches[matches.pblntf_ty == ty].drop_duplicates("rcept_no")
            examples = [] if ty in ("B", "I") else subset[["rcept_no", "report_nm"]].head(5).to_dict("records")
            rows.append({"keyword": keyword, "pblntf_ty": ty, "count": len(subset),
                         "ratio_to_BI": len(subset) / base if base else None,
                         "examples": json.dumps(examples, ensure_ascii=False)})
    return pd.DataFrame(rows, columns=["keyword", "pblntf_ty", "count", "ratio_to_BI", "examples"])


def run_coverage(*, client, archive, today, quarters=("2020Q1", "2023Q2"),
                 types=("A", "C", "D", "E", "F", "J"), config_path=Path("configs/prereg.yaml"),
                 data_raw=Path("data/raw"), reports=Path("reports")):
    import yaml
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not config or "event_keywords" not in config.get("features", {}):
        raise ValueError("부록 1 설정의 이벤트 키워드가 필요합니다")
    if (archive.root / "manifest.json").exists():
        previous = archive.load_manifest()
        if (previous["mode"] != "coverage" or previous["params"].get("quarters") != list(quarters)
                or previous["params"].get("coverage_types") != list(types)):
            raise ValueError("기존 coverage 실행의 분기 또는 유형을 바꿀 수 없습니다")
    windows = []
    for quarter in quarters:
        match = re.fullmatch(r"(\d{4})Q([1-4])", quarter)
        if not match:
            raise ValueError("분기는 YYYYQ1~YYYYQ4 형식이어야 합니다")
        year, q = map(int, match.groups())
        first = date(year, (q - 1) * 3 + 1, 1)
        last_month = q * 3
        last = date(year, last_month, calendar.monthrange(year, last_month)[1])
        windows.extend(make_windows(first, last, types))
    # 같은 분기의 B/I 완결 백필이 있으면 재사용한다.
    references = []
    for path in sorted(Path(data_raw).glob("*/*_PM_disclosures_backfill*/opendart_list/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if any(w["state"] == "complete" for w in manifest["windows"].values()):
            references.append(path.parent)
    for quarter in quarters:
        year, q = int(quarter[:4]), int(quarter[-1])
        bgn = date(year, (q - 1) * 3 + 1, 1)
        end_month = q * 3
        end = date(year, end_month, calendar.monthrange(year, end_month)[1])
        for ty in ("B", "I"):
            intervals = []
            for directory in references:
                manifest = RawArchive(directory).load_manifest()
                intervals.extend((date.fromisoformat(w["bgn"]), date.fromisoformat(w["end"]))
                                 for w in manifest["windows"].values()
                                 if w["pblntf_ty"] == ty and w["state"] == "complete")
            if _interval_end(intervals, bgn) < end:
                windows.append(Window(ty, bgn, end))
    manifest = _init_manifest(archive, windows, client, today, "coverage")
    manifest["params"]["quarters"] = list(quarters)
    manifest["params"]["coverage_types"] = list(types)
    archive.save_manifest(manifest)
    collected = collect_run(client=client, archive=archive, today=today)
    if collected.status != "complete":
        return collected, None
    raw, _, _, _, _ = _read_observations([archive.root, *references], include_coverage=True)
    # 참조 백필의 다른 분기는 표본에 섞지 않는다.
    periods = [(w.bgn, w.end) for w in windows]
    raw = [row for row in raw if any(date.fromisoformat(row["window_bgn"]) <= end
                                    and date.fromisoformat(row["window_end"]) >= bgn
                                    for bgn, end in periods)]
    frame = pd.DataFrame(raw, columns=["rcept_no", "pblntf_ty", "report_nm"] if not raw else None)
    counts = coverage_counts(frame, config)
    folder = _report_directory("coverage", today, reports)
    _csv(counts, folder / "coverage.csv")
    atomic_write(folder / "config.yaml", yaml.safe_dump({"quarters": list(quarters), "types": list(types),
                 "runs": [archive.root.parent.name, *[p.parent.name for p in references]],
                 "keywords": {"features": config["features"]["event_keywords"],
                              "events": config["events"]["triggers"]}}, allow_unicode=True).encode("utf-8"))
    atomic_write(folder / "git_commit.txt", (json.dumps(code_version()) + "\n").encode())
    atomic_write(folder / "metrics.json", (json.dumps({"matching_rows": int(counts["count"].sum()),
                 "windows_by_state": collected.windows_by_state}, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    outside = counts[(~counts.pblntf_ty.isin(["B", "I"])) & (counts["count"] > 0)]
    message = "B·I 밖 매칭 0건" if outside.empty else outside[["keyword", "pblntf_ty", "count"]].to_string(index=False)
    atomic_write(folder / "summary.md", ("# 공시유형 범위 확인\n\n" + message + "\n\n유형 추가 여부는 PM이 결정한다.\n").encode("utf-8"))
    return collected, folder
