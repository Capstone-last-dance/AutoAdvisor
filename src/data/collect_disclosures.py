"""PM-04 CLI: python -m src.data.collect_disclosures --help."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

from .dart_client import KST, CallBudget, DartClient, DartError
from .disclosure_archive import RawArchive
from .disclosures import (PublishRefused, build_disclosures, collect_daily, collect_run,
                          new_archive, plan_run, publish_disclosures, run_coverage, write_build_report)


def _types(value):
    items = value.split(",")
    if not items or len(set(items)) != len(items) or any(t not in tuple("ABCDEFGHIJ") for t in items):
        raise argparse.ArgumentTypeError("유형은 중복 없이 B,I 형식으로 입력합니다")
    return tuple(items)


def _find_archive(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id는 ASCII 영숫자, 밑줄, 하이픈만 허용합니다")
    matches = list(Path("data/raw").glob(f"*/{run_id}/opendart_list/manifest.json"))
    if len(matches) != 1:
        raise ValueError("run_id에 해당하는 수집 실행이 없거나 여러 개입니다")
    return RawArchive.open(matches[0].parent)


class _SecretFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        key = os.environ.get("OPENDART_API_KEY", "")
        if key:
            message = message.replace(key, "***")
        record.msg = re.sub(r"crtfc_key=[^&\s]+", "crtfc_key=***", message)
        record.args = ()
        return True


def _configure_log(run_id):
    folder = Path("logs")
    folder.mkdir(exist_ok=True)
    logger = logging.getLogger("src.data")
    logger.setLevel(logging.INFO)
    for old in list(logger.handlers):
        if getattr(old, "_pm04", False):
            logger.removeHandler(old)
            old.close()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(folder / f"disclosures_{run_id}.log", encoding="utf-8")):
        handler._pm04 = True
        handler.addFilter(_SecretFilter())
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parser():
    root = argparse.ArgumentParser(description="PM-04 OpenDART 공시 수집기")
    subs = root.add_subparsers(dest="command", required=True)
    today = datetime.now(KST).date()
    for name in ("plan", "collect"):
        command = subs.add_parser(name)
        command.add_argument("--start", type=date.fromisoformat)
        command.add_argument("--end", type=date.fromisoformat)
        command.add_argument("--types", type=_types, default=("B", "I"))
        command.add_argument("--window-months", type=int, choices=(1, 3), default=3)
        command.add_argument("--run-id")
        command.add_argument("--daily-budget", type=int, default=18000)
        if name == "collect":
            command.add_argument("--max-attempts", type=int, default=3)
            command.add_argument("--no-split", action="store_true")
    daily = subs.add_parser("daily")
    daily.add_argument("--date", type=date.fromisoformat, default=today)
    daily.add_argument("--lookback-days", type=int, default=3)
    daily.add_argument("--types", type=_types, default=("B", "I"))
    daily.add_argument("--daily-budget", type=int, default=18000)
    status = subs.add_parser("status")
    status.add_argument("--run-id", required=True)
    build = subs.add_parser("build")
    build.add_argument("--runs", default="all")
    build.add_argument("--schema-mode", choices=("v03", "extended"), default="v03")
    build.add_argument("--publish", action="store_true")
    build.add_argument("--allow-shrink", metavar="사유")
    coverage = subs.add_parser("coverage")
    coverage.add_argument("--quarters")
    coverage.add_argument("--types", type=_types)
    coverage.add_argument("--run-id")
    coverage.add_argument("--daily-budget", type=int, default=18000)
    return root


def _exit_code(manifest):
    if manifest["status"] == "failed":
        return 1
    if manifest["status"] == "incomplete":
        return 2
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(dotenv_path=Path(".env"), override=False)
    today = datetime.now(KST).date()
    try:
        if args.command == "status":
            archive = _find_archive(args.run_id)
            manifest = archive.load_manifest()
            _configure_log(manifest["run_id"])
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return _exit_code(manifest)
        if args.command == "build":
            run_dirs = None if args.runs == "all" else [_find_archive(i).root for i in args.runs.split(",")]
            _configure_log(f"{today:%Y%m%d}_PM_disclosures_build")
            result = build_disclosures(run_dirs, schema_mode=args.schema_mode, today=today)
            result.allow_shrink_reason = args.allow_shrink
            report = write_build_report(result)
            print(f"report={report} rows={len(result.df)} conflicts={len(result.conflicts)}")
            if args.publish:
                path = publish_disclosures(result, allow_shrink_reason=args.allow_shrink)
                print(f"published={path}")
            return 0
        if args.command == "daily":
            _configure_log(f"{args.date:%Y%m%d}_PM_disclosures_daily")
            result = collect_daily(args.date, lookback_days=args.lookback_days, types=args.types,
                                   client=DartClient(budget=CallBudget(args.daily_budget)))
            print(json.dumps(asdict(result), ensure_ascii=False, default=str))
            return _exit_code({"status": result.status})
        if args.command == "coverage":
            client = DartClient(budget=CallBudget(args.daily_budget))
            archive = _find_archive(args.run_id) if args.run_id else new_archive("coverage", today=today)
            if args.run_id and archive.load_manifest()["mode"] != "coverage":
                raise ValueError("coverage 모드 실행만 coverage 명령으로 재개할 수 있습니다")
            saved = archive.load_manifest()["params"] if args.run_id else {}
            quarters = args.quarters.split(",") if args.quarters else saved.get("quarters", ["2020Q1", "2023Q2"])
            types = args.types or saved.get("coverage_types", ("A", "C", "D", "E", "F", "J"))
            _configure_log(archive.root.parent.name)
            result, report = run_coverage(client=client,
                                          archive=archive, today=today, quarters=quarters, types=types)
            print(f"status={result.status} report={report}")
            return _exit_code({"status": result.status})
        client = DartClient(budget=CallBudget(args.daily_budget))
        existing = list(Path("data/raw").glob(f"*/{args.run_id}/opendart_list/manifest.json")) if args.run_id else []
        if args.run_id and (args.command == "collect" or existing):
            if args.start is not None or args.end is not None:
                raise ValueError("기존 run_id 재개에는 --start/--end를 함께 지정하지 않습니다")
            archive = _find_archive(args.run_id)
            manifest = archive.load_manifest()
            if manifest["mode"] != "backfill":
                raise ValueError("daily/coverage 실행은 원래 하위 명령으로 재개합니다")
            params = manifest["params"]
            start, end, types, months = date.fromisoformat(params["start"]), date.fromisoformat(params["end"]), params["types"], params["window_months"]
        else:
            archive = new_archive("backfill", today=today, run_id=args.run_id)
            start, end, types, months = (args.start or date(2010, 1, 1)), (args.end or today - timedelta(days=1)), args.types, args.window_months
        _configure_log(archive.root.parent.name)
        summary = plan_run(start, end, types=types, months=months, client=client, archive=archive, today=today)
        print(json.dumps(asdict(summary), ensure_ascii=False))
        manifest = archive.load_manifest()
        if args.command == "plan" or manifest["stop_reason"]:
            print(f"run_id={manifest['run_id']} status={manifest['status']}")
            return _exit_code(manifest)
        result = collect_run(client=client, archive=archive, today=today,
                             max_attempts=args.max_attempts, split_on_failure=not args.no_split)
        print(json.dumps(asdict(result), ensure_ascii=False, default=str))
        return _exit_code({"status": result.status})
    except PublishRefused as exc:
        print(f"저장 거부: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("중단됨. 기존 run_id로 재개할 수 있습니다.", file=sys.stderr)
        return 2
    except (DartError, ValueError, OSError, ImportError) as exc:
        # 외부 라이브러리 예외의 URL 또는 인증 정보를 출력하지 않는다.
        message = str(exc) if isinstance(exc, (DartError, ValueError)) else type(exc).__name__
        key = os.environ.get("OPENDART_API_KEY", "")
        if key:
            message = message.replace(key, "***")
        message = re.sub(r"crtfc_key=[^&\s]+", "crtfc_key=***", message)
        print(f"실패: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
