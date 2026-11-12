"""주간 실행기 (M11, 명세서 5.2절, 6.12절). D 소유.

    python -m src.pipeline.run_weekly --date YYYY-MM-DD

5.2절 1~7단계를 순서대로 실행한다.
- 각 단계는 해당 모듈의 공개 함수만 호출한다. 연결된 함수가 없는 단계는 9.3절 스키마를 만족하는
  빈 스텁을 _stub/ 아래에 쓴다 (6.12절).
- 한 단계라도 실패하면 이후 단계를 실행하지 않고, 실패 단계와 오류를 logs/{date}.log에 남긴 뒤
  종료 코드 1로 끝낸다.
- 스텁이 하나라도 쓰인 실행은 forward test 커밋을 하지 않는다 (7단계).

모듈 연결: 각 단계의 `entry`에 담당 모듈의 공개 함수를 "모듈경로:함수이름"으로 적는다.
함수는 `fn(ctx: StepContext) -> None` 형태이며, 입력은 ctx.read_table로 읽고 출력은 ctx.write_table로 쓴다.
담당자와 형태를 합의하기 전에는 entry를 None으로 둔다 (R-08).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import re
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.common import io
from src.pipeline import forward_test

PROCESSED_TABLES = (
    "prices", "index", "tickers", "status", "universe", "delisted", "delisted_reused", "sector", "disclosures", "news_daily", "news",
)


@dataclass
class StepContext:
    """한 번의 주간 실행에서 단계들이 공유하는 상태."""

    date: str
    stubbed: list[str] = field(default_factory=list)   # 스텁으로 대신한 단계
    stub_tables: set[str] = field(default_factory=set)  # 이번 실행에서 _stub/에 쓴 테이블
    memory: dict = field(default_factory=dict)          # 저장하지 않는 중간값 (4단계 w_raw)
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("run_weekly"))
    source_commit: str | None = None                    # 신호를 만든 코드의 커밋 (DEC-04)

    @property
    def uses_stub(self) -> bool:
        return bool(self.stubbed)

    def read_table(self, name: str, **kwargs) -> pd.DataFrame:
        # 이번 실행에서 스텁으로 쓴 테이블은 _stub/에서 읽는다
        return io.read_table(name, stub=name in self.stub_tables, **kwargs)

    def write_table(self, name: str, df: pd.DataFrame, **kwargs) -> Path:
        # 앞 단계에 스텁이 있으면 그 결과로 만든 출력도 _stub/에 쓴다 (실제 폴더에 섞이지 않게)
        stub = self.uses_stub
        path = io.write_table(name, df, stub=stub, **kwargs)
        if stub:
            self.stub_tables.add(name)
        return path

    def read_evidence(self) -> list[dict]:
        return io.read_evidence(self.date, stub="evidence" in self.stub_tables)

    def write_evidence(self, records: list[dict]) -> Path:
        stub = self.uses_stub
        path = io.write_evidence(self.date, records, stub=stub)
        if stub:
            self.stub_tables.add("evidence")
        return path


@dataclass(frozen=True)
class Step:
    no: int
    module: str                         # 5.2절 모듈
    name: str
    entry: str | Callable[[StepContext], None] | None
    stub: Callable[[StepContext], None]


# ---------------------------------------------------------------------------
# 스텁 (9.3절 스키마를 만족하는 빈 출력)
# ---------------------------------------------------------------------------

def empty_frame(name: str) -> pd.DataFrame:
    """9.3절 스키마의 열과 자료형을 가진 0행 DataFrame."""
    dtypes = {
        "date": "datetime64[ns]", "datetime_kst": pd.DatetimeTZDtype("ns", "Asia/Seoul"),
        "str": "str", "ticker": "str", "ticker_or_cash": "str",
        "float64": "float64", "int64": "int64", "int32": "int32", "int8": "int8",
        "nullable_int8": "Int8", "bool": "bool", "nullable_bool": "boolean",
    }
    return pd.DataFrame({col: pd.Series(dtype=dtypes[rule.kind]) for col, rule in io.SCHEMAS[name].items()})


def _stub_tables(*names: str, dated: bool) -> Callable[[StepContext], None]:
    def stub(ctx: StepContext) -> None:
        for name in names:
            io.write_table(name, empty_frame(name), date=ctx.date if dated else None, stub=True)
            ctx.stub_tables.add(name)
    return stub


def _stub_w_raw(ctx: StepContext) -> None:
    ctx.memory["w_raw"] = pd.Series(dtype="float64")  # 종목 없음 = 현금 100%


def _stub_evidence(ctx: StepContext) -> None:
    io.write_evidence(ctx.date, [], stub=True)
    ctx.stub_tables.add("evidence")


# ---------------------------------------------------------------------------
# 7단계 M7 (D 소유)
# ---------------------------------------------------------------------------

def forward_test_commit(ctx: StepContext) -> None:
    if ctx.uses_stub:
        ctx.log.info("7단계: 스텁이 쓰인 실행이라 forward test 커밋을 하지 않는다 (스텁 단계: %s)",
                     ", ".join(ctx.stubbed))
        return
    if ctx.source_commit is None:
        raise forward_test.ForwardTestError("forward test 커밋에는 --source-commit이 필요하다 (DEC-04)")
    deviations = ctx.memory.get("deviations") or []   # 예: M9가 없어 쿨다운 없이 기록한 주 (M4·M8)
    for note in deviations:  # 상세 사유(예외 문장 포함)는 로컬 로그에만 남긴다
        ctx.log.warning("7단계: 기록 시 이탈 사항 상세: %s", note)
    commit_hash = forward_test.record(ctx.date, source_commit=ctx.source_commit, memo=deviation_memo(ctx.date, deviations))
    ctx.log.info("7단계: 신호 커밋 %s를 forward_log.csv에 기록하고 push했다", commit_hash)


# 커밋 메시지는 push되어 공개된다. 이탈 문장에는 예외 메시지(로컬 경로 등)가 섞일 수 있어
# 정해진 사유로만 바꿔 쓰고, 상세는 logs/{date}.log에 둔다 (PM #38 리뷰)
_DEVIATION_REASONS = (
    (re.compile(r"M9 쿨다운 없이 계산"), "M9 쿨다운 없이 계산 (PM 11/9 결정)"),
    (re.compile(r"M9 체결 지도 없이 계산"), "M9 체결 지도 없이 계산 (PM 11/9 결정)"),
)
_OTHER_DEVIATION = "그 밖의 이탈"


def deviation_memo(date: str, deviations: Sequence[str]) -> str | None:
    """이탈 문장 목록을 공개 커밋용 메모로 바꾼다. 이탈이 없으면 None."""
    if not deviations:
        return None
    reasons: list[str] = []
    for note in deviations:
        reason = next((r for pattern, r in _DEVIATION_REASONS if pattern.search(str(note))), _OTHER_DEVIATION)
        if reason not in reasons:
            reasons.append(reason)
    return "\n".join([f"기록 시 이탈 사항 (상세는 실행 로그 logs/{date}.log):", *(f"- {r}" for r in reasons)])


def _no_stub(ctx: StepContext) -> None:
    raise AssertionError("7단계는 스텁으로 대신하지 않는다")


# 5.2절 1~7단계. entry가 None인 단계는 담당자와 함수 형태를 합의하기 전이다.
STEPS: tuple[Step, ...] = (
    Step(1, "M1", "데이터 갱신", None, _stub_tables(*PROCESSED_TABLES, dated=False)),
    Step(2, "M2", "피처 계산", None, _stub_tables("features", dated=True)),
    Step(3, "M3a·M3b", "급락 확률·순위 점수", "src.models.signals:m3_step", _stub_tables("signals", dated=True)),
    Step(4, "M4", "원시 목표 비중 w_raw", "src.portfolio.weekly:m4_step", _stub_w_raw),
    Step(5, "M8", "리스크 레이어", "src.portfolio.weekly:m8_step", _stub_tables("positions", dated=True)),
    Step(6, "M6", "근거 요약", None, _stub_evidence),
    Step(7, "M7", "forward test 커밋", forward_test_commit, _no_stub),
)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def resolve(entry: str | Callable | None) -> Callable[[StepContext], None] | None:
    """entry를 함수로 바꾼다. 모듈이나 함수가 아직 없으면 None (스텁 대상)."""
    if entry is None or callable(entry):
        return entry
    module_name, _, func_name = entry.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and module_name.startswith(exc.name):
            return None  # 모듈이 아직 없다
        raise            # 모듈 안에서 다른 import가 실패한 것은 오류다
    fn = getattr(module, func_name, None)
    return fn if callable(fn) else None


def run(date: str, steps: Sequence[Step] = STEPS, source_commit: str | None = None) -> int:
    """주간 실행. 성공하면 0, 한 단계라도 실패하면 1."""
    io._check_date_arg(date)
    log = _setup_logger(date)
    ctx = StepContext(date=date, log=log, source_commit=source_commit)
    log.info("주간 실행 시작: 신호일 %s", date)

    for step in steps:
        label = f"{step.no}단계 {step.module} {step.name}"
        try:
            fn = resolve(step.entry)
            if fn is None:
                log.warning("%s: 연결된 함수가 없어 스텁으로 대신한다", label)
                step.stub(ctx)
                ctx.stubbed.append(f"{step.no}({step.module})")
            else:
                log.info("%s: 시작", label)
                fn(ctx)
            log.info("%s: 완료", label)
        except Exception as exc:  # 어떤 오류든 이후 단계를 막는다
            log.error("%s: 실패 (%s: %s)\n%s", label, type(exc).__name__, exc, traceback.format_exc())
            log.error("주간 실행 중단: %s 이후 단계는 실행하지 않았다", label)
            _close_logger(log)
            return 1

    log.info("주간 실행 완료 (스텁 사용: %s)", "예, " + ", ".join(ctx.stubbed) if ctx.uses_stub else "아니오")
    _close_logger(log)
    return 0


def _setup_logger(date: str) -> logging.Logger:
    log_path = io.ROOT / "logs" / f"{date}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"run_weekly.{date}")
    log.setLevel(logging.INFO)
    log.propagate = False
    _close_logger(log)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stderr)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def _close_logger(log: logging.Logger) -> None:
    for handler in list(log.handlers):
        handler.close()
        log.removeHandler(handler)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="주간 실행기 (명세서 5.2절, 6.12절)")
    parser.add_argument("--date", required=True, help="신호일 YYYY-MM-DD")
    parser.add_argument("--source-commit", help="신호를 만든 코드의 커밋 해시. forward test 커밋에 필요하다 (DEC-04)")
    args = parser.parse_args(argv)
    try:
        io._check_date_arg(args.date)
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return run(args.date, source_commit=args.source_commit)


if __name__ == "__main__":
    sys.exit(main())
