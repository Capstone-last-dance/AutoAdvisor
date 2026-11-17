"""PM-05 초기 CLI: 공식 네이버 원본 표본 1회 수집. 최종 news.parquet은 쓰지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from .disclosure_archive import atomic_write
from .naver_client import KST, NaverError, NaverNewsClient, NewsSearchRequest
from .news import NewsRecordError, parse_naver_time


def _json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def capture_naver_sample(request, client, *, run_id, data_root=Path("data/raw"), reports_root=Path("reports")):
    if re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run_id는 ASCII 영숫자·밑줄·하이픈만 허용합니다")
    today = datetime.now(KST).date()
    raw_dir = Path(data_root) / today.isoformat() / run_id / "naver_search"
    report_dir = Path(reports_root) / run_id
    if raw_dir.exists() or report_dir.exists():
        raise FileExistsError("같은 run_id의 자료가 있어 새 수집을 거부합니다")
    raw_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    version = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    report = {"run_id": run_id, "source": "naver", "request": request.params(),
              "code_commit": version.stdout.strip() if version.returncode == 0 else None,
              "status": "planned", "max_calls_this_execution": client.max_calls,
              "calls_attempted": 0, "coverage_claim": "one_search_snapshot_only",
              "historical_complete": False, "ticker_mapping_performed": False,
              "news_parquet_written": False}
    atomic_write(report_dir / "config.json", _json({"request":request.params(), "max_calls_this_execution":client.max_calls}))
    try:
        page = client.search(request)
        atomic_write(raw_dir / "response_0001.json", page.body)
        metadata = {"request": request.params(), "fetched_at":page.fetched_at.isoformat(),
                    "http_status":page.http_status, "elapsed_ms":page.elapsed_ms,
                    "sha256":hashlib.sha256(page.body).hexdigest(), "bytes":len(page.body)}
        atomic_write(raw_dir / "response_0001.meta.json", _json(metadata))
        report.update(http_status=page.http_status, fetched_at=page.fetched_at.isoformat())
        payload = page.payload()
        valid_dates, invalid_dates = [], 0
        for item in payload["items"]:
            try:
                timestamp = parse_naver_time(item.get("pubDate"))
                if timestamp > page.fetched_at:
                    raise NewsRecordError("발행 시각이 관측 시각보다 늦은 기사")
                valid_dates.append(timestamp)
            except NewsRecordError:
                invalid_dates += 1
        report.update(status="captured", authentication_verified=True, items=len(payload["items"]),
                      search_total=payload["total"], unmapped_items=len(payload["items"]),
                      invalid_or_future_dates=invalid_dates,
                      published_min=min(valid_dates).isoformat() if valid_dates else None,
                      published_max=max(valid_dates).isoformat() if valid_dates else None)
    except NaverError as exc:
        report.update(status="failed", authentication_verified=False, error_type=type(exc).__name__)
    finally:
        report["calls_attempted"] = client.calls_attempted
        atomic_write(raw_dir / "manifest.json", _json(report))
        atomic_write(report_dir / "summary.json", _json(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="PM-05 네이버 뉴스 원본 표본 1회 수집")
    parser.add_argument("--query", required=True)
    parser.add_argument("--display", type=int, default=100)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--sort", choices=("date", "sim"), default="date")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(".env"), override=False)
        request = NewsSearchRequest(args.query, args.display, args.start, args.sort)
        run_id = args.run_id or datetime.now(KST).strftime("%Y%m%d_PM_news_naver_%H%M%S")
        report = capture_naver_sample(request, NaverNewsClient(max_calls=1), run_id=run_id)
        print(f"run_id={run_id} status={report['status']} calls={report['calls_attempted']}")
        return 0 if report["status"] == "captured" else 2
    except (NaverError, ValueError, OSError) as exc:
        print(f"뉴스 수집 준비 실패 ({type(exc).__name__})")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
