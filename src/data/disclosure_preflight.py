"""신호일 공시 피처의 원본 준비 상태를 확인한다."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.common import io
from src.data.disclosure_archive import RawArchive
from src.data.disclosures import PublishRefused
from src.events import alerts
from src.rag.evidence import _CLOSE, _windows_completed_before


class _ReadOnlyArchive(RawArchive):
    def append_event(self, event):
        """점검 중 손상 페이지를 발견해도 원본 폴더의 로그는 건드리지 않는다."""


def _load_windows(raw: Path) -> pd.DataFrame:
    windows = []
    for path in sorted(raw.glob("*/*_PM_disclosures_*/opendart_list/manifest.json")):
        # RawArchive.open은 임시 파일을 지운다. 수집 중에도 점검할 수 있게 직접 읽는다.
        archive = _ReadOnlyArchive(path.parent)
        manifest = archive.load_manifest()
        if manifest["mode"] == "coverage":
            continue
        for window_id, window in manifest["windows"].items():
            windows.append(window)
            if window["state"] not in {"complete", "provisional"}:
                continue
            expected = {attempt["n"]: attempt["pages"] for attempt in window["attempts"]}
            for number in window["accepted_attempts"]:
                pages = sum(1 for _ in archive.iter_pages(window_id, number))
                if not pages or (number in expected and pages != expected[number]):
                    raise PublishRefused(f"원본 페이지 손상 또는 누락: {manifest['run_id']}/{window_id}")
    return pd.DataFrame(windows, columns=["bgn", "end", "pblntf_ty", "state", "accepted_attempts", "attempts"])


def _finished_at(window) -> pd.Timestamp | None:
    """아직 채택된 시도가 전부 끝나지 않았다면 완료 시각이 없다."""
    accepted = set(window.accepted_attempts)
    finished = {a["n"]: pd.Timestamp(a["finished_at"]) for a in window.attempts
                if a["n"] in accepted and a["finished_at"]}
    if not accepted or accepted != finished.keys():
        return None
    return max(finished.values())


def _coverage(days: list[str], windows: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    eligible = _windows_completed_before(windows, cutoff)
    gaps = []
    for day in days:
        candidates = windows.loc[(windows["bgn"] <= day) & (windows["end"] >= day)]
        for kind in alerts.DISCLOSURE_TYPES:
            timely = eligible.loc[(eligible["pblntf_ty"] == kind) & (eligible["bgn"] <= day) & (eligible["end"] >= day)]
            if not timely.empty:
                continue
            seen = candidates.loc[candidates["pblntf_ty"] == kind]
            completed = seen.loc[seen["state"] == "complete"]
            late = [ts for window in completed.itertuples(index=False)
                    if (ts := _finished_at(window)) is not None and ts >= cutoff]
            gap = {"date": day, "type": kind,
                   "status": "late" if late else "missing" if seen.empty else "incomplete"}
            if late:
                gap["completed_at"] = min(late).tz_convert(alerts.KST).isoformat()
            elif not seen.empty:
                gap["window_states"] = sorted(seen["state"].unique())
            gaps.append(gap)
    return {"ready": not gaps, "required_slots": len(days) * len(alerts.DISCLOSURE_TYPES), "gaps": gaps}


def check_disclosure_inputs(signal_date: date) -> dict:
    days = alerts.load_trading_days()
    day = pd.Timestamp(signal_date)
    position = days.get_indexer([day])[0]
    if position < 0:
        raise ValueError(f"{signal_date}: 거래일 달력에 신호일이 없습니다")
    # signal_day_disclosures와 같은 t−19~t 거래일 창에서 t일은 제외한다.
    required = days[max(0, position - 19):position].strftime("%Y-%m-%d").tolist()
    cutoff = day.tz_localize(alerts.KST) + _CLOSE
    windows = _load_windows(io.ROOT / "data" / "raw")
    return {"signal_date": signal_date.isoformat(), "cutoff_at": cutoff.isoformat(),
            "required_dates": required, **_coverage(required, windows, cutoff)}


def _print_report(report: dict) -> None:
    print(f"{report['signal_date']} 공시 사전점검 | 마감 {report['cutoff_at']}")
    days = report["required_dates"]
    if days:
        print(f"대상 {days[0]} ~ {days[-1]}의 {len(days)}거래일 · B/I")
    missing = len(report["gaps"])
    total = report["required_slots"]
    print(f"{'통과' if report['ready'] else '미충족'} | 컷오프 전 확인 {total - missing}/{total}")
    explanations = {"missing": "원본 수집 창 없음", "incomplete": "수집 완료 미확인", "late": "마감 정각 또는 이후 완료"}
    for gap in report["gaps"]:
        detail = gap.get("completed_at") or ", ".join(gap.get("window_states", []))
        print(f"- {gap['date']} {gap['type']} {explanations[gap['status']]}" + (f" ({detail})" if detail else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, required=True, help="신호일 YYYY-MM-DD")
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")
    args = parser.parse_args(argv)
    try:
        report = check_disclosure_inputs(args.date)
    except (OSError, ValueError, KeyError, TypeError, PublishRefused) as error:
        print(f"공시 원본 점검 오류 | {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
