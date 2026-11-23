"""M7 forward test 기록 (명세서 6.8절). D 소유.

    git switch -c forward/YYYYMMDD origin/main
    python -m src.pipeline.forward_test --date YYYY-MM-DD --source-commit <신호를 만든 코드 커밋>

신호일 t의 signals/{t}.parquet(있으면 positions/{t}.parquet도)를 git에 커밋하고,
그 커밋 해시를 forward_log.csv에 적은 뒤 forward_log.csv를 따로 커밋한다.
신호 커밋에 태그 ft-{YYYYMMDD}를 붙이고 브랜치와 태그를 origin에 push한다 (DEC-04).
run_weekly 7단계도 같은 record()를 부른다.

- forward/{YYYYMMDD} 브랜치에서만 기록한다. main에 직접 커밋하지 않는다 (DEC-04).
- 마감은 신호일 다음 거래일 09:00 KST다 (DEC-04). 지나면 기록하지 않고, 커밋·로그 기록·push 직전마다 다시 확인한다.
- 날짜별 입력을 커밋 전에 검사한다 (6.8절): 11/20 전 baseline만, 11/20~26 baseline+positions, 11/27부터 ML+positions.
- 11/27부터 B 병행 모델(T7-80-C53) 비중 positions/wf_{YYYYMMDD}_B_t7c53.parquet을 같은 신호 커밋·태그에 넣는다
  (PM 11/23). 단 이번 주간 실행의 5b가 성공적으로 쓴 파일(ctx.memory["parallel_written"])만 넣는다. 5b를 건너뛰거나
  실패한 경우, 기록기를 직접 실행한 경우에는 같은 날짜 파일이 있어도 공식 신호·비중만 기록한다.
- 이미 기록한 신호일은 다시 기록하지 않는다 (커밋한 신호는 고치지 않는다).
- 스텁(_stub/)은 읽지 않는다. 실제 폴더의 파일만 커밋한다.
- 마감 후 기록(--late, PM 승인 11/23): 마감이 지난 신호일을 PM이 승인한 경우에만, deviation log에 그 신호일의
  "마감 후 기록" 행이 있을 때 기록한다. 태그는 ft-late-{YYYYMMDD}로 정식 기록(ft-*)과 구분하고, 커밋 메시지에
  마감과 실제 기록 시각을 남긴다. forward_log.csv에는 적지 않는다(마감 준수 기록만 둔다).
  past_positions 등 정식 태그 읽기에는 들어가지 않는다.
- 신호를 만든 코드의 커밋 해시를 두 커밋 메시지에 적는다 (DEC-04).
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.common import io

KST = "Asia/Seoul"
LABEL_H = 20  # 4.6절 ER_i(t,20). 라벨은 t+1+h일 종가에 확정된다

# 평일 중 KRX 휴장일. 거래일 캘린더(A)가 main에 들어오면 그것으로 바꾼다.
KRX_CLOSED_DAYS = pd.to_datetime(["2026-12-25", "2026-12-31", "2027-01-01"])
CALENDAR_END = pd.Timestamp("2027-01-31")  # 위 휴장일 목록이 맞다고 확인한 마지막 날

# 6.8절 일정: 11/20부터 positions를 함께 기록하고, 11/27부터 ML 신호로 바꾼다
POSITIONS_FROM = pd.Timestamp("2026-11-20")
ML_FROM = pd.Timestamp("2026-11-27")

PUBLISHED_TAGS = "refs/forward-published"  # origin의 ft-* 태그를 받아 두는 곳
DEVIATION_LOG = "docs/deviation_log.md"
LATE_MARK = "마감 후 기록"  # deviation log 행과 커밋 메시지에 쓰는 표시

# B 병행 모델 비중 (PM 11/23): 이 날짜부터 5b가 이번 실행에서 쓴 파일을 공식 기록과 같은 커밋·태그에 남긴다.
# 경로 positions/wf_{YYYYMMDD}_B_{name}.parquet (io의 run_id 경로). None이면 넣지 않는다(보류).
PARALLEL_FROM: pd.Timestamp | None = pd.Timestamp("2026-11-27")
PARALLEL_NAME = "t7c53"


def parallel_run_id(date: str, name: str = PARALLEL_NAME) -> str:
    return f"{date.replace('-', '')}_B_{name}"

log = logging.getLogger("forward_test")


class ForwardTestError(RuntimeError):
    """기록하면 안 되는 상태. 아무것도 커밋하지 않고 멈춘다."""


class ForwardTestPartial(ForwardTestError):
    """기록 도중 멈췄다. 메시지에 끝난 단계와 남은 일을 적는다."""


def expected_input(day: pd.Timestamp) -> tuple[str, bool]:
    """신호일에 기록해야 하는 (signal_source, positions 포함 여부) (6.8절)."""
    if day < POSITIONS_FROM:
        return "baseline", False
    if day < ML_FROM:
        return "baseline", True
    return "ml", True


def label_ready_date(date: str | pd.Timestamp, h: int = LABEL_H) -> pd.Timestamp:
    """신호일 t의 라벨이 확정되는 날 = t+1+h 거래일 (4.6절)."""
    days = _trading_days_after(date)
    if len(days) < 1 + h:
        raise ForwardTestError(f"{CALENDAR_END.date()} 이후 거래일을 알 수 없다. KRX_CLOSED_DAYS를 갱신한다")
    return days[h]


def commit_deadline(date: str | pd.Timestamp) -> pd.Timestamp:
    """기록 마감 = 신호일 다음 거래일 09:00 KST (DEC-04 변경). 이 시각 전에 커밋하고 push한다."""
    days = _trading_days_after(date)
    if len(days) == 0:
        raise ForwardTestError(f"{CALENDAR_END.date()} 이후 거래일을 알 수 없다. KRX_CLOSED_DAYS를 갱신한다")
    return days[0].replace(hour=9).tz_localize(KST)


def _trading_days_after(date: str | pd.Timestamp) -> pd.DatetimeIndex:
    days = pd.bdate_range(pd.Timestamp(date) + pd.Timedelta(days=1), CALENDAR_END)
    return days[~days.isin(KRX_CLOSED_DAYS)]


def record(date: str, source_commit: str, push: bool = True, memo: str | None = None,
           now: Callable[[], pd.Timestamp] | None = None,
           late: str | None = None, parallel: str | None = None) -> str:
    """신호일 date의 신호를 커밋하고 해시를 forward_log.csv에 적는다. 신호 커밋 해시를 돌려준다.

    source_commit: 신호를 만든 코드의 커밋 (DEC-04).
    push: False면 커밋과 태그까지만 하고 push하지 않는다 (리허설용).
    memo: 두 커밋 메시지에 덧붙일 설명. 예: universe v1의 관리종목 단계 미적용 (DEC-24).
    late: 마감 후 기록 사유 (PM 승인). 주면 마감이 지난 경우에만, deviation log에 그 신호일의 "마감 후 기록"
        행이 있을 때 ft-late-{YYYYMMDD} 태그로 기록하고 forward_log.csv에는 적지 않는다.
    parallel: 이번 주간 실행의 5b가 성공적으로 쓴 B 병행 비중 파일 경로 (ctx.memory["parallel_written"]).
        신호일 경로와 같을 때만 같은 신호 커밋·태그에 넣는다. 기본값 None(직접 실행 등)이면 공식 신호·비중만 기록한다.
    """
    io._check_date_arg(date)
    now = now or (lambda: pd.Timestamp.now(tz=KST))
    deadline = commit_deadline(date)
    if late is not None:
        _check_late(date, late, deadline, now())

    def check_deadline(done: str | None = None) -> None:
        if late is not None or now() < deadline:
            return
        passed = f"마감({deadline:%Y-%m-%d %H:%M} KST)이 지났다 (DEC-04)"
        if done is None:
            raise ForwardTestError(f"{passed}. 아무것도 기록하지 않았다")
        raise ForwardTestPartial(f"{passed}. {done}. 남은 단계는 하지 않았다. PM에게 알리고 deviation log에 남긴다")

    check_deadline()

    stamp = date.replace("-", "")
    branch, tag = f"forward/{stamp}", (f"ft-late-{stamp}" if late is not None else f"ft-{stamp}")
    _git("rev-parse", "--is-inside-work-tree")
    current = _git("branch", "--show-current")
    if current != branch:
        raise ForwardTestError(f"{branch} 브랜치에서 실행한다 (지금: {current or '브랜치 없음'}). "
                               f"git switch -c {branch} origin/main")
    for existing in {tag, f"ft-{stamp}"}:
        if _git("tag", "--list", existing):
            raise ForwardTestError(f"태그 {existing}가 이미 있다. 커밋한 신호는 고치지 않는다 (6.8절)")
    try:
        code_commit = _git("rev-parse", "--verify", "--quiet", f"{source_commit}^{{commit}}")
    except ForwardTestError:
        raise ForwardTestError(f"신호를 만든 코드 커밋 {source_commit}을 찾을 수 없다. git fetch 후 다시 실행한다") from None

    forward_log = io.read_table("forward_log")
    day = pd.Timestamp(date)
    if (forward_log["signal_date"] == day).any():
        raise ForwardTestError(f"{date}는 이미 forward_log.csv에 있다. 커밋한 신호는 고치지 않는다 (6.8절)")

    signals = io.read_table("signals", date=date)
    if signals.empty:
        raise ForwardTestError(f"signals/{date}.parquet에 행이 없다")
    if not (signals["date"] == day).all():
        raise ForwardTestError(f"signals/{date}.parquet에 다른 날짜의 행이 있다")
    source = _signal_source(signals)

    paths = [f"signals/{date}.parquet"]
    has_positions = False
    try:
        io.read_table("positions", date=date)
        paths.append(f"positions/{date}.parquet")
        has_positions = True
    except FileNotFoundError:
        pass
    want_source, want_positions = expected_input(day)
    if source != want_source:
        raise ForwardTestError(f"{date}은 {want_source} 신호를 기록하는 날인데 {source} 신호다 (6.8절)")
    if has_positions != want_positions:
        raise ForwardTestError(f"{date}은 positions {'포함' if want_positions else '없이'} 기록하는 날이다 (6.8절)")
    attached = _parallel_path(date, day, parallel)
    if attached:
        paths.append(attached)

    note = f"\n\n신호를 만든 코드 커밋: {code_commit}" + (f"\n{memo}" if memo else "")
    if late is not None:
        note += (f"\n{LATE_MARK}: 마감 {deadline:%Y-%m-%d %H:%M} KST, 기록 시각 {now():%Y-%m-%d %H:%M} KST "
                 f"(PM 승인, {DEVIATION_LOG})\n사유: {late}")
        commit_hash = _commit_signals(date, paths, note, title=f"forward test ({LATE_MARK}): {date} 신호 커밋")
        _tag_and_push(commit_hash, branch, tag, push, check_deadline)
        log.warning("%s: %s으로 %s 태그를 남겼다. forward_log.csv에는 적지 않았다", date, LATE_MARK, tag)
        return commit_hash
    check_deadline()
    commit_hash = _commit_signals(date, paths, note)
    check_deadline(f"신호 커밋 {commit_hash}까지 끝났고 forward_log.csv에는 적지 않았다")

    row = pd.DataFrame({
        "signal_date": [day],
        "commit_hash": [commit_hash],
        "signal_source": [source],
        "has_positions": pd.array([int(has_positions)], dtype="int8"),
        "label_ready_date": [label_ready_date(date)],
        # 사후 열은 확정 전까지 비워 둔다 (9.3절)
        "mean_er_top20": [float("nan")],
        "mean_er_flagged": [float("nan")],
        "portfolio_return": [float("nan")],
        "bm_return": [float("nan")],
    })
    out = row if forward_log.empty else pd.concat([forward_log, row], ignore_index=True)
    try:
        io.write_table("forward_log", out)
        _git("add", "--", "forward_log.csv")
        _git("commit", "-m", f"forward test: {date} 해시 기록\n\n신호 커밋: {commit_hash}{note}",
             "--", "forward_log.csv")
    except (ForwardTestError, ValueError, OSError) as exc:
        # 파일에 행이 남으면 다시 실행할 때 '이미 기록됨'으로 막힌다
        _git("restore", "--source=HEAD", "--staged", "--worktree", "--", "forward_log.csv")
        raise ForwardTestPartial(f"신호 커밋 {commit_hash}까지 끝났고 forward_log.csv 기록은 되돌렸다. "
                                 f"다시 실행하면 이어서 기록한다 ({exc})") from None
    _tag_and_push(commit_hash, branch, tag, push, check_deadline)
    return commit_hash


def _tag_and_push(commit_hash: str, branch: str, tag: str, push: bool,
                  check_deadline: Callable[[str | None], None]) -> None:
    try:
        _git("tag", tag, commit_hash)
    except ForwardTestError as exc:
        raise ForwardTestPartial(f"커밋은 끝났고 태그만 실패했다. 직접 실행한다: git tag {tag} {commit_hash} "
                                 f"&& git push origin {branch} {tag} ({exc})") from None
    if push:
        check_deadline(f"커밋과 태그까지 끝났고 push는 하지 않았다 ({branch}, {tag})")
        try:
            _git("push", "origin", branch, f"refs/tags/{tag}")
        except ForwardTestError as exc:
            raise ForwardTestPartial(f"커밋과 태그는 끝났고 push만 실패했다. 직접 push한다: "
                                     f"git push origin {branch} {tag} ({exc})") from None
        log.info("기록 완료: 신호 커밋 %s, %s·%s push", commit_hash, branch, tag)
    else:
        log.info("기록 완료: 신호 커밋 %s (push 안 함)", commit_hash)


def _check_late(date: str, reason: str, deadline: pd.Timestamp, now: pd.Timestamp) -> None:
    """마감 후 기록의 조건: 사유가 있고, 마감이 실제로 지났고, deviation log에 그 신호일의 행이 있다."""
    if not reason.strip():
        raise ForwardTestError("마감 후 기록에는 사유(PM 승인 내용)를 적는다")
    if now < deadline:
        raise ForwardTestError(f"마감({deadline:%Y-%m-%d %H:%M} KST) 전이다. --late 없이 정식으로 기록한다")
    path = io.ROOT / DEVIATION_LOG
    rows = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if not any(line.startswith("|") and date in line and LATE_MARK in line for line in rows):
        raise ForwardTestError(f"{DEVIATION_LOG}에 {date} '{LATE_MARK}' 행이 없다. PM 승인을 deviation log에 먼저 남긴다")


def _parallel_path(date: str, day: pd.Timestamp, written: str | None) -> str | None:
    """함께 커밋할 B 병행 모델 비중 파일 경로. 이번 실행의 5b가 쓴 신호일 파일이 아니면 None (공식 기록만 한다)."""
    run_id = parallel_run_id(date)
    rel = f"positions/wf_{run_id}.parquet"
    exists = (io.ROOT / rel).is_file()
    if written is None:
        if exists:
            log.warning("이번 실행의 5b가 쓴 파일이 아니라 %s는 커밋하지 않는다 (공식 기록만 한다)", rel)
        return None
    if Path(written).as_posix() != rel:
        log.warning("5b가 쓴 경로 %s가 신호일 경로 %s와 달라 병행 파일을 커밋하지 않는다", written, rel)
        return None
    if PARALLEL_FROM is None or day < PARALLEL_FROM:
        log.warning("병행 모델 기록 시작일 전이라 %s는 커밋하지 않는다", rel)
        return None
    if not exists:
        log.warning("병행 모델 비중 %s가 없어 공식 기록만 한다", rel)
        return None
    try:
        df = io.read_table("positions", run_id=run_id)
        if df.empty or not (df["date"] == day).all():
            raise ValueError("행이 없거나 다른 날짜의 행이 있다")
    except Exception as exc:  # 병행 기록 문제로 공식 기록을 막지 않는다
        log.warning("병행 모델 비중 %s를 쓸 수 없어 공식 기록만 한다 (%s: %s)", rel, type(exc).__name__, exc)
        return None
    return rel


def past_positions(t: str) -> dict[str, pd.DataFrame]:
    """t보다 앞선 기록의 positions를 태그 ft-*에서 읽어 {신호일 "YYYY-MM-DD": 표}로 날짜순으로 돌려준다.

    positions가 없는 기록(11/20 전)은 뺀다. 병합을 기다리지 않도록 작업 폴더가 아니라 태그에서 읽고,
    origin에 게시된 태그만 전용 ref로 받아 읽는다. push에 실패해 로컬에만 남은 태그는 기록이 아니다.
    태그의 파일이 스키마를 어기면 예외를 낸다.
    """
    return past_tables(t, "positions")


def past_parallel_positions(t: str, name: str = PARALLEL_NAME) -> dict[str, pd.DataFrame]:
    """t보다 앞선 기록의 B 병행 모델 비중(positions/wf_{YYYYMMDD}_B_{name}.parquet). 없는 기록은 뺀다."""
    return past_tables(t, "positions", lambda date: f"positions/wf_{parallel_run_id(date, name)}.parquet")


def past_tables(t: str, table: str, path: Callable[[str], str] | None = None) -> dict[str, pd.DataFrame]:
    """t보다 앞선 기록의 파일을 게시된 태그 ft-*에서 읽어 {신호일: 표}로 날짜순으로 돌려준다.

    table: 스키마 검사에 쓸 9.3절 테이블 이름 (signals 또는 positions).
    path: 신호일 "YYYY-MM-DD"를 저장소 경로로 바꾸는 함수. 빼면 "{table}/{신호일}.parquet".
    그 파일이 없는 기록은 뺀다. 파일이 스키마를 어기거나 다른 날짜의 행이 있으면 예외를 낸다.
    """
    io._check_date_arg(t)
    path = path or (lambda date: f"{table}/{date}.parquet")
    _git("fetch", "-q", "--prune", "--no-tags", "origin", f"+refs/tags/ft-*:{PUBLISHED_TAGS}/ft-*")
    dates = sorted(f"{m[1]}-{m[2]}-{m[3]}"
                   for tag in _git("for-each-ref", "--format=%(refname:strip=2)", PUBLISHED_TAGS).splitlines()
                   if (m := re.fullmatch(r"ft-(\d{4})(\d{2})(\d{2})", tag)))
    out: dict[str, pd.DataFrame] = {}
    for date in dates:
        rel = path(date)
        tag = f"{PUBLISHED_TAGS}/ft-{date.replace('-', '')}"
        if date >= t or not _git("ls-tree", "--name-only", tag, "--", rel):
            continue
        result = subprocess.run(["git", "-C", str(io.ROOT), "cat-file", "blob", f"{tag}:{rel}"], capture_output=True)
        if result.returncode != 0:
            raise ForwardTestError(f"{tag}의 {rel}를 읽지 못했다")
        df = io._normalize_dates(table, pd.read_parquet(BytesIO(result.stdout)))
        io.validate(table, df)
        if not (df["date"] == pd.Timestamp(date)).all():
            raise ForwardTestError(f"{tag}의 {rel}에 다른 날짜의 행이 있다")
        out[date] = df
    return out


def _signal_source(signals: pd.DataFrame) -> str:
    baseline = signals["model_version"] == "baseline"
    if baseline.all():
        return "baseline"
    if not baseline.any():
        return "ml"
    raise ForwardTestError("한 신호일에 baseline과 ML 신호가 섞여 있다")


def _commit_signals(date: str, paths: list[str], note: str, title: str | None = None) -> str:
    """paths만 커밋하고 그 커밋 해시를 돌려준다. 다른 변경은 커밋에 넣지 않는다."""
    _git("add", "--", *paths)
    if not _git("diff", "--cached", "--name-only", "--", *paths):
        # 지난 실행이 신호 커밋 뒤, 해시 기록 전에 멈춘 경우: 모든 파일이 든 그 커밋을 그대로 쓴다
        commits = {_git("log", "-1", "--format=%H", "--", path) for path in paths}
        if len(commits) != 1:
            raise ForwardTestError(f"{', '.join(paths)}가 서로 다른 커밋에 있어 하나의 해시로 기록할 수 없다")
        last = commits.pop()
        log.warning("%s는 이미 커밋돼 있어 그 커밋(%s)의 해시를 기록한다", ", ".join(paths), last)
        return last
    committed = [path for path in paths if _git("ls-tree", "--name-only", "HEAD", "--", path)]
    if committed:
        _git("restore", "--staged", "--", *paths)
        raise ForwardTestError(f"이미 커밋된 파일이 달라졌다: {', '.join(committed)}. "
                               "커밋한 신호는 고치지 않는다 (6.8절)")
    _git("commit", "-m", f"{title or f'forward test: {date} 신호 커밋'}{note}", "--", *paths)
    return _git("rev-parse", "HEAD")


def _git(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(io.ROOT), *args], capture_output=True, text=True,
                            encoding="utf-8")
    if result.returncode != 0:
        raise ForwardTestError(f"git {args[0]} 실패: {result.stderr.strip()}")
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="forward test 기록 (명세서 6.8절)")
    parser.add_argument("--date", required=True, help="신호일 YYYY-MM-DD")
    parser.add_argument("--source-commit", required=True, help="신호를 만든 코드의 커밋 해시 (DEC-04)")
    parser.add_argument("--no-push", action="store_true", help="커밋과 태그까지만 한다 (리허설용)")
    parser.add_argument("--memo", help="커밋 메시지에 덧붙일 설명")
    parser.add_argument("--late", metavar="사유",
                        help="마감 후 기록 (PM 승인 필요, deviation log에 그 신호일의 '마감 후 기록' 행이 있어야 한다)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        commit_hash = record(args.date, source_commit=args.source_commit, push=not args.no_push, memo=args.memo,
                             late=args.late)
    except (ForwardTestError, FileNotFoundError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    print(f"{args.date} 기록 완료. 신호 커밋 {commit_hash}" + ("" if args.no_push else ", push 완료"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
