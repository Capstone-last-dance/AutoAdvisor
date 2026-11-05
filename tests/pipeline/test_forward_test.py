"""src/pipeline/forward_test.py 단위 테스트 (작업 D-04, 6.8절).

임시 폴더에 git 저장소를 만들어 실제로 커밋한다.
실행: python -m pytest tests/pipeline
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.common import io
from src.pipeline import forward_test as ft
from tests.common.test_io import sample

DAY = "2026-11-06"
BEFORE = lambda: pd.Timestamp(f"{DAY} 19:30", tz=ft.KST)   # 마감 전
AFTER = lambda: pd.Timestamp("2026-11-07 00:00:01", tz=ft.KST)


class RepoTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.tmp = self.base / "repo"
        self.remote = self.base / "remote.git"
        self.tmp.mkdir()
        subprocess.run(["git", "init", "-q", "--bare", str(self.remote)], check=True)
        self._root = io.ROOT
        io.ROOT = self.tmp
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "test")
        self.git("config", "user.email", "test@example.com")
        self.git("remote", "add", "origin", str(self.remote))
        (self.tmp / "forward_log.csv").write_text("", encoding="utf-8")   # main과 같은 빈 파일
        self.git("add", ".")
        self.git("commit", "-q", "-m", "init")
        self.source = self.git("rev-parse", "HEAD")                      # 신호를 만든 코드 커밋 역할
        self.git("switch", "-q", "-c", "forward/20261106")

    def tearDown(self):
        io.ROOT = self._root
        shutil.rmtree(self.base, ignore_errors=True)

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.tmp), *args], check=True, capture_output=True,
                              text=True, encoding="utf-8").stdout.strip()

    def commits(self):
        return self.git("log", "--format=%s").splitlines()

    def files_in(self, rev):
        return set(self.git("show", "--name-only", "--format=", rev).splitlines())

    def remote_ref(self, ref):
        return subprocess.run(["git", "-C", str(self.remote), "rev-parse", ref], check=True,
                              capture_output=True, text=True).stdout.strip()

    def record(self, date=DAY, **kwargs):
        kwargs.setdefault("source_commit", self.source)
        kwargs.setdefault("now", BEFORE)
        return ft.record(date, **kwargs)


class RecordTest(RepoTest):
    def test_first_record_signals_only(self):
        io.write_table("signals", sample("signals"), date=DAY)
        commit_hash = self.record()

        self.assertEqual(self.commits()[:2], [f"forward test: {DAY} 해시 기록", f"forward test: {DAY} 신호 커밋"])
        self.assertEqual(self.files_in(commit_hash), {f"signals/{DAY}.parquet"})
        self.assertEqual(self.files_in("HEAD"), {"forward_log.csv"})
        self.assertEqual(self.git("status", "--porcelain"), "")
        # 두 커밋 메시지에 신호를 만든 코드 커밋을 적는다 (DEC-04)
        self.assertIn(self.source, self.git("log", "-1", "--format=%B", commit_hash))
        self.assertIn(self.source, self.git("log", "-1", "--format=%B", "HEAD"))
        # 태그는 신호 커밋에 붙고, 브랜치와 태그를 push한다
        self.assertEqual(self.git("rev-parse", "ft-20261106^{commit}"), commit_hash)
        self.assertEqual(self.remote_ref("refs/heads/forward/20261106"), self.git("rev-parse", "HEAD"))
        self.assertEqual(self.remote_ref("ft-20261106^{commit}"), commit_hash)

        log = io.read_table("forward_log")
        self.assertEqual(len(log), 1)
        row = log.iloc[0]
        self.assertEqual(row["commit_hash"], commit_hash)
        self.assertEqual(row["signal_source"], "baseline")
        self.assertEqual(row["has_positions"], 0)
        self.assertEqual(row["label_ready_date"], pd.Timestamp("2026-12-07"))
        self.assertTrue(pd.isna(row["mean_er_top20"]))

    def test_with_positions_and_second_record(self):
        io.write_table("signals", sample("signals"), date=DAY)
        self.record()

        nxt = "2026-11-13"
        self.git("switch", "-q", "-c", "forward/20261113")
        sig = sample("signals"); sig["date"] = pd.Timestamp(nxt); sig["model_version"] = "lgbm_v1"
        pos = sample("positions"); pos["date"] = pd.Timestamp(nxt)
        io.write_table("signals", sig, date=nxt)
        io.write_table("positions", pos, date=nxt)
        commit_hash = self.record(nxt, now=lambda: pd.Timestamp(f"{nxt} 20:00", tz=ft.KST))

        self.assertEqual(self.files_in(commit_hash), {f"signals/{nxt}.parquet", f"positions/{nxt}.parquet"})
        log = io.read_table("forward_log")
        self.assertEqual(list(log["signal_date"]), [pd.Timestamp(DAY), pd.Timestamp(nxt)])
        self.assertEqual(list(log["signal_source"]), ["baseline", "ml"])
        self.assertEqual(list(log["has_positions"]), [0, 1])
        self.assertEqual(log.iloc[1]["label_ready_date"], pd.Timestamp("2026-12-14"))

    def test_other_changes_are_not_committed(self):
        (self.tmp / "notes.txt").write_text("작업 중", encoding="utf-8")
        self.git("add", "notes.txt")
        io.write_table("signals", sample("signals"), date=DAY)
        commit_hash = self.record()
        self.assertNotIn("notes.txt", self.files_in(commit_hash) | self.files_in("HEAD"))
        self.assertIn("A  notes.txt", self.git("status", "--porcelain"))

    def test_resume_after_signal_commit(self):
        # 신호 커밋 뒤 해시 기록 전에 멈췄다면, 다시 실행할 때 그 커밋 해시를 쓴다
        io.write_table("signals", sample("signals"), date=DAY)
        self.git("add", "signals")
        self.git("commit", "-q", "-m", "신호만 커밋됨")
        signal_commit = self.git("rev-parse", "HEAD")
        self.assertEqual(self.record(), signal_commit)
        self.assertEqual(io.read_table("forward_log").iloc[0]["commit_hash"], signal_commit)


class RefuseTest(RepoTest):
    def assertRefused(self, exc=ft.ForwardTestError, **kwargs):
        head = self.git("rev-parse", "HEAD")
        with self.assertRaises(exc):
            self.record(**kwargs)
        self.assertEqual(self.git("rev-parse", "HEAD"), head)   # 아무것도 커밋하지 않았다

    def test_after_deadline(self):
        io.write_table("signals", sample("signals"), date=DAY)
        self.assertRefused(now=AFTER)

    def test_already_recorded(self):
        io.write_table("signals", sample("signals"), date=DAY)
        self.record()
        self.assertRefused()

    def test_wrong_branch(self):
        io.write_table("signals", sample("signals"), date=DAY)
        self.git("switch", "-q", "main")
        self.assertRefused()

    def test_unknown_source_commit(self):
        io.write_table("signals", sample("signals"), date=DAY)
        self.assertRefused(source_commit="0123456789abcdef0123456789abcdef01234567")

    def test_push_failure_keeps_commits(self):
        io.write_table("signals", sample("signals"), date=DAY)
        self.git("remote", "set-url", "origin", str(self.base / "no_such_remote.git"))
        with self.assertRaisesRegex(ft.ForwardTestError, "push만 실패"):
            self.record()
        self.assertEqual(self.commits()[0], f"forward test: {DAY} 해시 기록")

    def test_no_signals(self):
        self.assertRefused(FileNotFoundError)

    def test_stub_signals_are_ignored(self):
        io.write_table("signals", sample("signals"), date=DAY, stub=True)
        self.assertRefused(FileNotFoundError)

    def test_empty_signals(self):
        io.write_table("signals", sample("signals").iloc[0:0], date=DAY)
        self.assertRefused()

    def test_mixed_sources(self):
        sig = sample("signals"); sig.loc[1, "model_version"] = "lgbm_v1"
        io.write_table("signals", sig, date=DAY)
        self.assertRefused()


class LabelReadyDateTest(unittest.TestCase):
    def test_dates(self):
        # t+1+20 거래일 (4.6절)
        self.assertEqual(ft.label_ready_date("2026-11-06"), pd.Timestamp("2026-12-07"))
        self.assertEqual(ft.label_ready_date("2026-11-20"), pd.Timestamp("2026-12-21"))
        # 12/25, 12/31, 1/1 휴장을 건너뛴다
        self.assertEqual(ft.label_ready_date("2026-12-04"), pd.Timestamp("2027-01-07"))

    def test_beyond_calendar(self):
        with self.assertRaises(ft.ForwardTestError):
            ft.label_ready_date("2027-01-15")


class MainTest(RepoTest):
    def test_cli_error_exit_code(self):
        self.assertEqual(ft.main(["--date", DAY, "--source-commit", self.source]), 1)   # 신호 파일 없음
        self.assertEqual(ft.main(["--date", "2026-13-01", "--source-commit", self.source]), 1)


if __name__ == "__main__":
    unittest.main()
