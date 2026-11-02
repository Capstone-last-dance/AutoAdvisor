"""src/pipeline/run_weekly.py 단위 테스트 (작업 D-03, 6.12절 완료 기준 2).

실행: python -m pytest tests/pipeline
"""

import dataclasses
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.common import io
from src.pipeline import run_weekly as rw

DAY = "2026-11-06"


class TempRootTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._root = io.ROOT
        io.ROOT = self.tmp

    def tearDown(self):
        io.ROOT = self._root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def log_text(self):
        return (self.tmp / "logs" / f"{DAY}.log").read_text(encoding="utf-8")


def with_entries(**entries):
    """STEPS에서 일부 단계의 entry만 바꾼 목록. 키는 s1~s7."""
    return tuple(dataclasses.replace(s, entry=entries.get(f"s{s.no}", s.entry)) for s in rw.STEPS)


class EmptyFrameTest(unittest.TestCase):
    def test_stub_frames_satisfy_schema(self):
        for name in io.SCHEMAS:
            with self.subTest(name=name):
                io.validate(name, rw.empty_frame(name))


class AllStubRunTest(TempRootTest):
    """지금처럼 연결된 모듈이 없으면 모든 단계를 스텁으로 대신하고 정상 종료한다."""

    def test_all_stub_run(self):
        self.assertEqual(rw.run(DAY), 0)
        stub = self.tmp / "_stub"
        for name in rw.PROCESSED_TABLES:
            self.assertTrue((stub / "data" / "processed" / f"{name}.parquet").exists(), name)
        for folder in ("features", "signals", "positions"):
            self.assertTrue((stub / folder / f"{DAY}.parquet").exists(), folder)
        self.assertTrue((stub / "evidence" / f"{DAY}.jsonl").exists())
        # 실제 폴더에는 아무것도 쓰지 않는다
        for folder in ("signals", "positions", "features", "evidence", "data"):
            self.assertFalse((self.tmp / folder).exists(), folder)
        log = self.log_text()
        self.assertIn("스텁으로 대신한다", log)
        self.assertIn("forward test 커밋을 하지 않는다", log)
        self.assertIn("주간 실행 완료", log)


class FailureTest(TempRootTest):
    """6.12절 완료 기준 (2): 한 단계에 오류를 넣으면 이후 단계가 실행되지 않고 종료 코드가 1이다."""

    def test_failure_stops_later_steps(self):
        calls = []

        def ok(no):
            def fn(ctx):
                calls.append(no)
            return fn

        def boom(ctx):
            calls.append(3)
            raise RuntimeError("일부러 넣은 오류")

        steps = with_entries(s1=ok(1), s2=ok(2), s3=boom, s4=ok(4), s5=ok(5), s6=ok(6), s7=ok(7))
        self.assertEqual(rw.run(DAY, steps), 1)
        self.assertEqual(calls, [1, 2, 3])
        log = self.log_text()
        self.assertIn("3단계 M3a·M3b 급락 확률·순위 점수: 실패 (RuntimeError: 일부러 넣은 오류)", log)
        self.assertIn("Traceback", log)
        self.assertIn("이후 단계는 실행하지 않았다", log)
        self.assertNotIn("4단계 M4 원시 목표 비중 w_raw: 시작", log)

    def test_stub_failure_also_stops(self):
        def bad_stub(ctx):
            raise ValueError("스텁 쓰기 실패")
        steps = tuple(dataclasses.replace(s, stub=bad_stub) if s.no == 2 else s for s in rw.STEPS)
        self.assertEqual(rw.run(DAY, steps), 1)
        self.assertFalse((self.tmp / "_stub" / "signals" / f"{DAY}.parquet").exists())

    def test_all_real_steps_reach_unimplemented_commit(self):
        # 스텁이 없으면 7단계 커밋을 해야 하는데, D-04 전에는 조용히 넘어가지 않고 실패한다
        noop = lambda ctx: None
        steps = with_entries(s1=noop, s2=noop, s3=noop, s4=noop, s5=noop, s6=noop)
        self.assertEqual(rw.run(DAY, steps), 1)
        self.assertIn("NotImplementedError", self.log_text())


class StubPropagationTest(TempRootTest):
    """앞 단계가 스텁이면 뒤 단계의 실제 출력도 _stub/에 쓴다."""

    def test_real_step_after_stub_reads_and_writes_stub(self):
        seen = {}

        def features(ctx):
            seen["prices_rows"] = len(ctx.read_table("prices"))  # 1단계 스텁을 읽는다
            ctx.write_table("features", rw.empty_frame("features"), date=ctx.date)

        self.assertEqual(rw.run(DAY, with_entries(s2=features)), 0)
        self.assertEqual(seen["prices_rows"], 0)
        self.assertTrue((self.tmp / "_stub" / "features" / f"{DAY}.parquet").exists())
        self.assertFalse((self.tmp / "features" / f"{DAY}.parquet").exists())


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "d03_broken_mod.py").write_text("import d03_not_installed_pkg\n", encoding="utf-8")
        sys.path.insert(0, str(self.tmp))

    def tearDown(self):
        sys.path.remove(str(self.tmp))
        sys.modules.pop("d03_broken_mod", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_resolve(self):
        self.assertIsNone(rw.resolve(None))
        self.assertIsNone(rw.resolve("src.models.not_yet_written:generate"))   # 모듈 없음
        self.assertIsNone(rw.resolve("src.common.io:no_such_function"))         # 함수 없음
        self.assertIs(rw.resolve("src.common.io:validate"), io.validate)
        with self.assertRaises(ModuleNotFoundError):                            # 모듈 안의 import 실패는 오류
            rw.resolve("d03_broken_mod:run")


class MainTest(TempRootTest):
    def test_invalid_date(self):
        self.assertEqual(rw.main(["--date", "2026-13-01"]), 1)
        self.assertFalse((self.tmp / "logs").exists())

    def test_cli_runs(self):
        self.assertEqual(rw.main(["--date", DAY]), 0)


if __name__ == "__main__":
    unittest.main()
