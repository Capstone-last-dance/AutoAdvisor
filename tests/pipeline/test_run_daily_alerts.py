"""src/pipeline/run_daily_alerts.py 단위 테스트 (작업 D-06).

실행: python -m pytest tests/pipeline/test_run_daily_alerts.py
"""

import datetime as dt
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.common import io
from src.pipeline import run_daily_alerts as rda

SAT = "2026-11-14"  # 토요일도 달력일로 돌린다


def alerts_frame(day: str, n: int) -> pd.DataFrame:
    at = pd.to_datetime([f"{day} 23:59:59"] * n).tz_localize("Asia/Seoul")
    return pd.DataFrame({
        "date": pd.to_datetime([day] * n),
        "ticker": pd.Series([f"{i:06d}" for i in range(1, n + 1)], dtype="str"),
        "trigger_type": pd.Series(["T1"] * n, dtype="str"),
        "rcept_no": pd.Series([f"{day.replace('-', '')}{i:06d}" for i in range(1, n + 1)], dtype="str"),
        "report_nm": pd.Series(["샘플 공시"] * n, dtype="str"),
        "rcept_dt": pd.to_datetime([pd.NaT] * n).tz_localize("Asia/Seoul"),
        "available_at": at,
        "generated_at": at,
    })


class AlertsNotCollected(RuntimeError):
    """PM M9의 전용 예외를 흉내 낸다 (이름으로 구분한다)."""


class TempRootTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._root, io.ROOT = io.ROOT, self.tmp

    def tearDown(self):
        io.ROOT = self._root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, day=SAT):
        return self.tmp / "alerts" / f"{day}.parquet"

    def log_text(self, day=SAT):
        return (self.tmp / "logs" / f"alerts_{day}.log").read_text(encoding="utf-8")


class RunTest(TempRootTest):
    def test_saves_alerts_of_calendar_day(self):
        calls = []

        def fn(d):
            calls.append(d)
            return alerts_frame(d, 2)

        self.assertEqual(rda.run(SAT, fn), 0)
        self.assertEqual(calls, [SAT])
        self.assertEqual(len(io.read_table("alerts", date=SAT)), 2)
        self.assertIn("확정 경보 2건", self.log_text())

    def test_zero_alerts_writes_empty_file(self):
        # 정상 수집 0건은 0행 파일로 남긴다. 파일 없음은 미수집이다
        self.assertEqual(rda.run(SAT, lambda d: alerts_frame(d, 0)), 0)
        self.assertTrue(self.path().exists())
        self.assertEqual(len(io.read_table("alerts", date=SAT)), 0)
        self.assertIn("확정 경보 0건", self.log_text())

    def test_not_collected_writes_nothing(self):
        def fn(d):
            raise AlertsNotCollected(f"{d}가 아직 끝나지 않았다")

        self.assertEqual(rda.run(SAT, fn), 1)
        self.assertFalse(self.path().exists())
        self.assertIn("미수집", self.log_text())

    def test_failure_keeps_previous_result(self):
        self.assertEqual(rda.run(SAT, lambda d: alerts_frame(d, 2)), 0)
        before = self.path().read_bytes()
        for fn in (lambda d: (_ for _ in ()).throw(AlertsNotCollected("수집 실패")),
                   lambda d: (_ for _ in ()).throw(RuntimeError("일부러 넣은 오류")),
                   lambda d: alerts_frame("2026-11-13", 1)):           # 다른 날짜 행은 write_table이 막는다
            with self.subTest(fn=fn):
                self.assertEqual(rda.run(SAT, fn), 1)
                self.assertEqual(self.path().read_bytes(), before)
        self.assertIn("Traceback", self.log_text())

    def test_rerun_replaces_on_success(self):
        rda.run(SAT, lambda d: alerts_frame(d, 2))
        self.assertEqual(rda.run(SAT, lambda d: alerts_frame(d, 1)), 0)
        self.assertEqual(len(io.read_table("alerts", date=SAT)), 1)

    def test_missing_m9_function_fails(self):
        with mock.patch.object(rda, "M9_FUNCTION", "no_such_function"):
            self.assertEqual(rda.run(SAT), 1)
        self.assertIn("DailyAlertsUnavailable", self.log_text())
        self.assertFalse(self.path().exists())


class DateTest(TempRootTest):
    def test_default_is_yesterday_kst(self):
        utc = dt.timezone.utc
        # 11/15 08:00 KST = 11/14 23:00 UTC
        self.assertEqual(rda.default_date(dt.datetime(2026, 11, 14, 23, 0, tzinfo=utc)), "2026-11-14")
        self.assertEqual(rda.default_date(dt.datetime(2026, 11, 14, 14, 0, tzinfo=utc)), "2026-11-13")

    def test_invalid_date(self):
        self.assertEqual(rda.main(["--date", "2026-13-01"]), 1)
        self.assertFalse((self.tmp / "logs").exists())

    def test_main_uses_default_date(self):
        with mock.patch.object(rda, "default_date", return_value=SAT), \
                mock.patch.object(rda, "run", return_value=0) as run:
            self.assertEqual(rda.main([]), 0)
        run.assert_called_once_with(SAT)


if __name__ == "__main__":
    unittest.main()
