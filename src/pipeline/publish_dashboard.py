"""대시보드 배포 (작업 D-08, Q-17 결정). D 소유.

    python -m src.pipeline.publish_dashboard --date YYYY-MM-DD --target ../배포저장소 [--no-push]

공식 주간 기록(forward test)이 성공한 뒤 수동으로 돌린다 (PM 11/16 Q-17).
실행에 쓴 앱 코드와 검증된 신호·비중·경보·공시 근거를 비공개 배포 저장소에 한 커밋으로 올린다.
Streamlit Community Cloud는 그 저장소의 app/Home.py를 띄운다.

- 신호·비중은 origin에 게시된 forward 태그(ft-*)에서 읽는다. 공식 기록이 아닌 진단용 신호는 올리지 않는다.
- 신호일 t가 forward_log.csv에 있고 태그 ft-t가 게시돼 있어야 한다.
- 경보(alerts)는 t까지의 날짜별 파일, 공시 근거(evidence)는 공식 기록 신호일의 파일만 올린다.
- 뉴스 원문(news, news_daily)과 data/raw/, .env 등은 올리지 않는다. 올리는 파일은 아래 목록으로 고정한다.
- 대상 저장소의 origin이 이 저장소(공개)와 같으면 거부한다. 팀 전용 실데이터는 비공개 저장소로만 보낸다.
- 앱 코드는 이 저장소 HEAD의 파일이어야 한다. 앱 코드 경로에 커밋하지 않은 변경이 있으면 거부한다.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.common import io
from src.pipeline import forward_test

# 대시보드가 import하는 코드만 올린다 (app/ → src.common.io, src.events.alerts → triggers,
# src.pipeline.forward_test, src.portfolio.params). 늘어나면 tests가 import 실패로 알려 준다.
CODE_FILES = (
    "src/__init__.py",
    "src/common/__init__.py", "src/common/io.py",
    "src/events/__init__.py", "src/events/alerts.py", "src/events/triggers.py",
    "src/pipeline/__init__.py", "src/pipeline/forward_test.py",
    "src/portfolio/__init__.py", "src/portfolio/params.py",
    "configs/prereg.yaml",
)
CODE_DIRS = ("app",)
# 배포 저장소에서 이 스크립트가 관리하는 경로. 다시 돌리면 모두 지우고 새로 쓴다.
MANAGED = ("app", "src", "configs", "signals", "positions", "alerts", "evidence", "data", "reports",
           "forward_log.csv", "requirements.txt")
# Cloud에 설치할 패키지 (requirements.txt의 고정 버전을 그대로 쓴다)
PACKAGES = ("numpy", "pandas", "pyarrow", "streamlit", "PyYAML")


class PublishError(RuntimeError):
    pass


def publish(date: str, target: Path, push: bool = True) -> str:
    """신호일 date까지의 공식 결과를 target 저장소에 커밋한다. 커밋 해시를 돌려준다."""
    io._check_date_arg(date)
    target = target.resolve()
    _check_target(target)
    _check_code_clean()
    code_commit = _git(io.ROOT, "rev-parse", "HEAD")

    official = _official_dates(date)
    for path in MANAGED:
        _remove(target / path)

    for rel in CODE_FILES:
        _copy_head(rel, target)
    for folder in CODE_DIRS:
        for rel in _git(io.ROOT, "ls-files", "--", folder).splitlines():
            _copy_head(rel, target)

    for day in official:
        tag = f"{forward_test.PUBLISHED_TAGS}/ft-{day.replace('-', '')}"
        for folder in ("signals", "positions"):
            rel = f"{folder}/{day}.parquet"
            if _git(io.ROOT, "ls-tree", "--name-only", tag, "--", rel):
                _write(target / rel, _blob(f"{tag}:{rel}"))
        _copy_file(io.ROOT / "evidence" / f"{day}.jsonl", target, required=False)

    for day in io.list_dates("alerts"):
        if day <= date:
            _copy_file(io.ROOT / "alerts" / f"{day}.parquet", target, required=True)
    _copy_file(io.ROOT / "forward_log.csv", target, required=True)
    _copy_file(io.ROOT / "data" / "processed" / "index.parquet", target, required=True)
    final = io.ROOT / "reports" / "final"
    for name in io.list_final_reports():
        _copy_file(final / name, target, required=True)
    (target / "requirements.txt").write_text(_requirements(), encoding="utf-8")

    # 지금 있거나 전에 올렸던 경로만 (없는 경로를 주면 git add가 실패한다)
    paths = [p for p in MANAGED if (target / p).exists() or _git(target, "ls-files", "--", p)]
    _git(target, "add", "-A", "--", *paths)
    if not _git(target, "diff", "--cached", "--name-only"):
        print(f"바뀐 내용이 없어 커밋하지 않았다 ({date})", file=sys.stderr)
        return _git(target, "rev-parse", "HEAD")
    _git(target, "commit", "-q", "-m", f"배포: 신호일 {date} (코드 {code_commit[:7]})")
    commit = _git(target, "rev-parse", "HEAD")
    if push:
        _git(target, "push", "-q", "origin", "HEAD")
    return commit


def _official_dates(date: str) -> list[str]:
    """date까지의 공식 기록 신호일. date가 기록돼 있지 않으면 거부한다."""
    log = io.read_table("forward_log")
    recorded = sorted(log["signal_date"].dt.strftime("%Y-%m-%d"))
    if date not in recorded:
        raise PublishError(f"{date}가 forward_log.csv에 없다. 공식 주간 기록이 성공한 뒤 배포한다")
    _git(io.ROOT, "fetch", "-q", "--prune", "--no-tags", "origin",
         f"+refs/tags/ft-*:{forward_test.PUBLISHED_TAGS}/ft-*")
    published = {f"{m[1]}-{m[2]}-{m[3]}"
                 for ref in _git(io.ROOT, "for-each-ref", "--format=%(refname:strip=2)",
                                 forward_test.PUBLISHED_TAGS).splitlines()
                 if (m := re.fullmatch(r"ft-(\d{4})(\d{2})(\d{2})", ref))}
    if date not in published:
        raise PublishError(f"태그 ft-{date.replace('-', '')}가 origin에 없다. 공식 기록 push가 끝난 뒤 배포한다")
    return [day for day in recorded if day <= date and day in published]


def _check_target(target: Path) -> None:
    if not (target / ".git").exists():
        raise PublishError("--target은 배포 저장소를 clone한 폴더여야 한다")
    if target == io.ROOT.resolve():
        raise PublishError("--target이 이 저장소다. 별도 배포 저장소를 쓴다")
    mine = _normalize_url(_git(io.ROOT, "remote", "get-url", "origin"))
    theirs = _normalize_url(_git(target, "remote", "get-url", "origin"))
    if mine == theirs:
        raise PublishError("배포 저장소의 origin이 공개 저장소와 같다. 팀 전용 실데이터는 비공개 저장소로만 올린다")


def _normalize_url(url: str) -> str:
    url = url.strip().removesuffix("/").removesuffix(".git").lower()
    return re.sub(r"^(https?://|git@)", "", url).replace(":", "/")


def _check_code_clean() -> None:
    paths = [*CODE_FILES, *CODE_DIRS]
    if _git(io.ROOT, "status", "--porcelain", "--", *paths):
        raise PublishError("앱 코드에 커밋하지 않은 변경이 있다. 실행에 쓴 커밋의 코드만 배포한다")


def _requirements() -> str:
    pins = {}
    for line in (io.ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if m:
            pins[m[1].lower()] = f"{m[1]}=={m[2]}"
    missing = [p for p in PACKAGES if p.lower() not in pins]
    if missing:
        raise PublishError(f"requirements.txt에 버전이 없다: {missing}")
    return "# 대시보드 배포용 (src/pipeline/publish_dashboard.py가 만든다)\n" + \
        "\n".join(pins[p.lower()] for p in PACKAGES) + "\n"


def _copy_head(rel: str, target: Path) -> None:
    _write(target / rel, _blob(f"HEAD:{rel}"))


def _copy_file(src: Path, target: Path, required: bool) -> None:
    if not src.is_file():
        if required:
            raise PublishError(f"{src.relative_to(io.ROOT).as_posix()}가 없다")
        return
    dest = target / src.relative_to(io.ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)


def _write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _blob(spec: str) -> bytes:
    result = subprocess.run(["git", "-C", str(io.ROOT), "cat-file", "blob", spec], capture_output=True)
    if result.returncode != 0:
        raise PublishError(f"git에서 {spec}를 읽지 못했다")
    return result.stdout


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise PublishError(f"git {args[0]} 실패: {result.stderr.strip()}")
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="대시보드 배포 (작업 D-08)")
    parser.add_argument("--date", required=True, help="공식 기록을 마친 신호일 YYYY-MM-DD")
    parser.add_argument("--target", required=True, help="비공개 배포 저장소를 clone한 폴더")
    parser.add_argument("--no-push", action="store_true", help="배포 저장소에 커밋까지만 한다")
    args = parser.parse_args(argv)
    try:
        commit = publish(args.date, Path(args.target), push=not args.no_push)
    except (PublishError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    print(f"배포 커밋 {commit[:7]} ({args.date}{', push 안 함' if args.no_push else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
