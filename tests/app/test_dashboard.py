"""app/ 대시보드 테스트 (작업 D-05, 6.11절 완료 기준 1~4).

실행: python -m pytest tests/app
"""

import ast
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.common import io
from src.pipeline.make_dashboard_sample import SIGNAL_DATES, write_sample
from tests.common.test_io import sample

APP = Path(__file__).resolve().parents[2] / "app"
PAGES = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]
sys.path.insert(0, str(APP))

import dashboard as ui  # noqa: E402


class RootTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._env = os.environ.get("DASHBOARD_DATA_ROOT")
        os.environ["DASHBOARD_DATA_ROOT"] = str(self.tmp)

    def tearDown(self):
        if self._env is None:
            del os.environ["DASHBOARD_DATA_ROOT"]
        else:
            os.environ["DASHBOARD_DATA_ROOT"] = self._env
        io.ROOT = ui.REPO_ROOT
        shutil.rmtree(self.tmp, ignore_errors=True)

    def open(self, page):
        at = AppTest.from_file(str(page), default_timeout=60).run()
        self.assertFalse(at.exception, f"{page.name}: {at.exception}")
        return at

    def captions(self, at):
        return " ".join(c.value for c in at.caption)


class SamplePagesTest(RootTest):
    def setUp(self):
        super().setUp()
        write_sample(self.tmp)

    def test_all_pages_open_with_header_and_disclaimer(self):
        for page in PAGES:
            with self.subTest(page=page.name):
                text = self.captions(self.open(page))
                self.assertIn(f"최신 신호일: {SIGNAL_DATES[-1]}", text)
                self.assertIn("파일 생성 시각", text)
                self.assertIn(io.DISCLAIMER, text)

    def test_weekly_signals_tables(self):
        at = self.open(APP / "pages" / "1_weekly_signals.py")
        flagged, candidates, portfolio = [d.value for d in at.dataframe]
        self.assertEqual(len(flagged), 4)
        self.assertEqual(len(candidates), 20)
        self.assertEqual(list(candidates["rank"]), sorted(candidates["rank"]))
        self.assertTrue(candidates["종목명"].str.startswith("샘플종목").all())
        self.assertEqual(len(portfolio), 5)

    def test_alert_names_use_prereg(self):
        at = self.open(APP / "pages" / "2_alerts.py")
        table = at.dataframe[0].value
        names = ui.trigger_names()
        self.assertEqual(set(table["경보 유형"]), {names["T1"], names["T5"]})

    def test_evidence_shows_summary(self):
        at = self.open(APP / "pages" / "3_evidence.py")
        self.assertTrue(any("샘플 요약" in m.value for m in at.markdown))

    def test_new_signal_file_shows_after_rerun(self):
        at = self.open(APP / "Home.py")
        saved, io.ROOT = io.ROOT, self.tmp
        new = "2026-11-20"
        df = sample("signals"); df["date"] = pd.Timestamp(new)
        io.write_table("signals", df, date=new)
        io.ROOT = saved
        at.run()
        self.assertFalse(at.exception)
        self.assertIn(f"최신 신호일: {new}", self.captions(at))


class EmptyPagesTest(RootTest):
    def test_all_pages_open_without_data(self):
        for page in PAGES:
            with self.subTest(page=page.name):
                text = self.captions(self.open(page))
                self.assertIn("최신 신호일: 없음", text)
                self.assertIn(io.DISCLAIMER, text)

    def test_performance_without_reports(self):
        at = self.open(APP / "pages" / "4_performance.py")
        self.assertEqual(at.info[0].value, "확정된 결과 없음")


class EntryCandidatesTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "ticker": ["A00001", "A00002", "A00003"], "rank": [3, 1, 2], "crash_flag": [0, 1, 0],
        })

    def test_excludes_flag_and_cooldown(self):
        self.assertEqual(list(ui.entry_candidates(self.frame(), set())["ticker"]), ["A00003", "A00001"])
        self.assertEqual(list(ui.entry_candidates(self.frame(), {"A00003"})["ticker"]), ["A00001"])


class StaticRulesTest(unittest.TestCase):
    """6.11절 금지 사항과 R-12."""

    FILES = sorted(APP.rglob("*.py"))

    def test_no_network_libraries(self):
        banned = {"requests", "urllib", "urllib3", "httpx", "aiohttp", "socket", "http"}
        for path in self.FILES:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    self.assertNotIn(name.split(".")[0], banned, f"{path.name}: {name}")

    def test_no_forbidden_words_or_stub(self):
        for path in self.FILES:
            text = path.read_text(encoding="utf-8")
            for word in ("매수", "매도", "추천", "_stub"):
                self.assertNotIn(word, text, f"{path.name}: {word}")


class IoHelpersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._root = io.ROOT
        io.ROOT = self.tmp

    def tearDown(self):
        io.ROOT = self._root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_dates_skips_wf_files(self):
        self.assertEqual(io.list_dates("signals"), [])
        io.write_table("signals", sample("signals"), date="2026-11-06")
        io.write_table("signals", sample("signals"), run_id="20261106_D_x")
        self.assertEqual(io.list_dates("signals"), ["2026-11-06"])

    def test_file_mtime(self):
        self.assertIsNone(io.file_mtime("signals", "2026-11-06"))
        io.write_table("signals", sample("signals"), date="2026-11-06")
        self.assertIsNotNone(io.file_mtime("signals", "2026-11-06"))

    def test_list_final_reports_ignores_gitkeep(self):
        final = self.tmp / "reports" / "final"
        self.assertEqual(io.list_final_reports(), [])
        final.mkdir(parents=True)
        (final / ".gitkeep").write_text("")
        self.assertEqual(io.list_final_reports(), [])
        (final / "kpi.csv").write_text("a")
        self.assertEqual(io.list_final_reports(), ["kpi.csv"])


if __name__ == "__main__":
    unittest.main()
