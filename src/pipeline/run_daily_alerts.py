"""일간 경보 실행기 (M11, 작업 D-06, 명세서 6.10절). D 소유.

    python -m src.pipeline.run_daily_alerts [--date YYYY-MM-DD]
    python -m src.pipeline.run_daily_alerts --from YYYY-MM-DD --to YYYY-MM-DD   # 기간 백필

달력일 d 하루의 M9 경보를 만들어 alerts/{d}.parquet에 저장한다 (PM 11/10 합의).
- d는 거래일이 아니라 달력일이다. 토요일 공시는 토요일 파일에만 들어간다.
- d가 끝난 다음 날 오전 08:00 KST쯤 돌린다. OpenDART는 접수일만 줘서 장 마감 직후의 0건은 확정이 아니다.
  --date를 빼면 어제(KST)를 쓴다.
- 공시 수집·종목 연결·거래일 달력·같은 사건의 기존 이력은 PM의 daily_alerts(d)가 처리한다.
  D는 날짜로 그 함수를 부르고 결과를 공용 I/O로 저장만 한다.
- 정상 수집 0건이면 0행 파일을 저장한다. 미수집(M9 AlertsNotCollected·AlertsNotStarted)·실패면 파일을 쓰지 않고 종료 코드 1이다.
  그래서 "파일 없음 = 미수집"이다.
- 같은 날짜를 다시 돌리면 성공한 결과로만 교체한다. 저장 중 실패하면 기존 파일은 그대로다 (io.write_table).
- 주간 실행(run_weekly)은 신호일 t의 경보 파일까지 읽는다. "일간 경보(t일까지) → 주간" 순서로 돌린다.
- --from/--to는 그 기간의 달력일을 하루씩 차례로 돌린다(백필). 하루가 미수집·실패여도 다음 날은 계속 돌리고,
  끝에 파일을 쓰지 못한 날짜를 알려 준다. 하루라도 실패하면 종료 코드 1이다. 날짜별 로그는 하루 실행과 같다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import traceback
from collections.abc import Callable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from src.common import io
from src.events import alerts as m9
from src.pipeline import runlog

KST = ZoneInfo("Asia/Seoul")


def default_date(now: dt.datetime | None = None) -> str:
    """--date를 빼면 쓰는 날짜: 어제(KST)."""
    now = now or dt.datetime.now(KST)
    return (now.astimezone(KST).date() - dt.timedelta(days=1)).isoformat()


def run(date: str, daily_alerts: Callable[[str], pd.DataFrame] | None = None) -> int:
    """d 하루의 경보를 저장한다. 성공하면 0, 미수집·실패면 1 (파일을 쓰지 않는다)."""
    io._check_date_arg(date)
    log = _setup_logger(date)
    try:
        log.info("일간 경보 시작: %s (달력일)", date)
        alerts = (daily_alerts or m9.daily_alerts)(date)
        path = io.write_table("alerts", alerts, date=date)
        log.info("확정 경보 %d건을 %s에 저장했다", len(alerts), path.relative_to(io.ROOT).as_posix())
        return 0
    except Exception as exc:
        if isinstance(exc, m9.AlertsNotCollected):   # AlertsNotStarted 포함
            log.error("미수집: %s 경보를 만들지 못했다 (%s). 파일을 쓰지 않았다", date, exc)
        else:
            log.error("실패 (%s: %s). 파일을 쓰지 않았다\n%s", type(exc).__name__, exc, traceback.format_exc())
        return 1
    finally:
        runlog.close_log(log)


def backfill(start: str, end: str, daily_alerts: Callable[[str], pd.DataFrame] | None = None) -> int:
    """start~end(양끝 포함) 달력일을 차례로 run한다. 모두 성공하면 0, 하루라도 미수집·실패면 1."""
    io._check_date_arg(start)
    io._check_date_arg(end)
    if start > end:
        raise ValueError(f"--from({start})이 --to({end})보다 늦다")
    days = pd.date_range(start, end, freq="D").strftime("%Y-%m-%d")
    failed = [day for day in days if run(day, daily_alerts) != 0]
    print(f"백필 {start} ~ {end}: {len(days) - len(failed)}/{len(days)}일 저장", file=sys.stderr)
    if failed:
        print(f"파일을 쓰지 못한 날 {len(failed)}일 (logs/alerts_{{날짜}}.log 확인): {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def _setup_logger(date: str) -> logging.Logger:
    return runlog.open_log(f"run_daily_alerts.{date}", io.ROOT / "logs" / f"alerts_{date}.log")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="일간 경보 실행기 (명세서 6.10절, 작업 D-06)")
    parser.add_argument("--date", help="경보를 만들 달력일 YYYY-MM-DD. 빼면 어제(KST)")
    parser.add_argument("--from", dest="start", help="백필 시작 달력일 YYYY-MM-DD (--to와 함께)")
    parser.add_argument("--to", dest="end", help="백필 끝 달력일 YYYY-MM-DD, 포함 (--from과 함께)")
    args = parser.parse_args(argv)
    if args.start or args.end:
        if args.date or not (args.start and args.end):
            parser.error("--from과 --to는 함께 쓰고, --date와는 함께 쓸 수 없다")
        try:
            return backfill(args.start, args.end)
        except ValueError as exc:
            print(f"오류: {exc}", file=sys.stderr)
            return 1
    date = args.date or default_date()
    try:
        io._check_date_arg(date)
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return run(date)


if __name__ == "__main__":
    sys.exit(main())
