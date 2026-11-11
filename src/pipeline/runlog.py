"""실행기 로그 (M11). D 소유. run_weekly·run_daily_alerts가 같이 쓴다.

로그 파일(logs/ 아래)과 표준 오류에 같은 내용을 남긴다.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def open_log(name: str, path: Path) -> logging.Logger:
    """path에 쓰는 로거. 같은 이름의 이전 핸들러는 닫고 새로 붙인다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False
    close_log(log)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stderr)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def close_log(log: logging.Logger) -> None:
    for handler in list(log.handlers):
        handler.close()
        log.removeHandler(handler)
