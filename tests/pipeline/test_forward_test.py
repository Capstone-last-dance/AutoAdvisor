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
AFTER = lambda: pd.Timestamp("2026-11-09 09:00:00", tz=ft.KST)   # 다음 거래일 09:00 정각부터 마감


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

    def prepare(self, date, version="baseline", positions=False):
        """date의 forward 브랜치로 옮겨 입력 파일을 만든다."""
        self.git("switch", "-q", "-c", f"forward/{date.replace('-', '')}")
        sig = sample("signals"); sig["date"] = pd.Timestamp(date); sig["model_version"] = version
        io.write_table("signals", sig, date=date)
        if positions:
            pos = sample("positions"); pos["date"] = pd.Timestamp(date)
            io.write_table("positions", pos, date=date)


class RecordTest(RepoTest):
    def test_first_record_signals_only(self):
        io.write_table("signals", sample("signals"), date=DAY)
        commit_hash = self.record(memo="universe v1: 관리종목 단계 미적용 (DEC-24)")

        self.assertIn("관리종목 단계 미적용", self.git("log", "-1", "--format=%B", commit_hash))
        self.assertIn("관리종목 단계 미적용", self.git("log", "-1", "--format=%B", "HEAD"))
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

    def test_schedule_6_8(self):
        io.write_table("signals", sample("signals"), date=DAY)
        self.record()
        last = None
        for date, version, positions in [("2026-11-13", "baseline", False), ("2026-11-20", "baseline", True),
                                         ("2026-11-27", "lgbm_v1", True)]:
            self.prepare(date, version, positions)
            last = self.record(date, now=lambda: pd.Timestamp(f"{date} 20:00", tz=ft.KST))

        self.assertEqual(self.files_in(last), {"signals/2026-11-27.parquet", "positions/2026-11-27.parquet"})
        log = io.read_table("forward_log")
        self.assertEqual(list(log["signal_source"]), ["baseline", "baseline", "baseline", "ml"])
        self.assertEqual(list(log["has_positions"]), [0, 0, 1, 1])
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


def clock(*times):
    """부를 때마다 다음 시각을 돌려주는 시계. 마지막 시각은 계속 유지한다."""
    stamps = [pd.Timestamp(t, tz=ft.KST) for t in times]
    return lambda: stamps.pop(0) if len(stamps) > 1 else stamps[0]


OK, LATE = "2026-11-09 08:59:58", "2026-11-09 09:00:01"


class DeadlineDuringRecordTest(RepoTest):
    """시작, 커밋, 로그 기록, push 직전에 마감을 확인한다."""

    def setUp(self):
        super().setUp()
        io.write_table("signals", sample("signals"), date=DAY)
        self.head = self.git("rev-parse", "HEAD")

    def test_before_commit_nothing_done(self):
        with self.assertRaises(ft.ForwardTestError) as ctx:
            self.record(now=clock(OK, LATE))
        self.assertNotIsInstance(ctx.exception, ft.ForwardTestPartial)
        self.assertEqual(self.git("rev-parse", "HEAD"), self.head)

    def test_before_log_is_partial(self):
        with self.assertRaisesRegex(ft.ForwardTestPartial, "신호 커밋"):
            self.record(now=clock(OK, OK, LATE))
        self.assertEqual(self.commits()[0], f"forward test: {DAY} 신호 커밋")
        self.assertTrue(io.read_table("forward_log").empty)
        self.assertEqual(self.git("tag", "--list", "ft-20261106"), "")

    def test_before_push_is_partial(self):
        with self.assertRaisesRegex(ft.ForwardTestPartial, "push는 하지 않았다"):
            self.record(now=clock(OK, OK, OK, LATE))
        self.assertNotEqual(self.git("tag", "--list", "ft-20261106"), "")
        result = subprocess.run(["git", "-C", str(self.remote), "rev-parse", "--verify", "--quiet",
                                 "refs/heads/forward/20261106"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


class RefuseTest(RepoTest):
    def assertRefused(self, exc=ft.ForwardTestError, **kwargs):
        head = self.git("rev-parse", "HEAD")
        with self.assertRaises(exc):
            self.record(**kwargs)
        self.assertEqual(self.git("rev-parse", "HEAD"), head)   # 아무것도 커밋하지 않았다

    def test_schedule_violations(self):
        cases = [("2026-11-06", "lgbm_v1", False), ("2026-11-13", "baseline", True),
                 ("2026-11-20", "baseline", False), ("2026-11-27", "baseline", True)]
        for date, version, positions in cases:
            with self.subTest(date=date):
                self.prepare_case(date, version, positions)
                self.assertRefused(date=date, now=lambda: pd.Timestamp(f"{date} 20:00", tz=ft.KST))
                self.assertEqual(self.git("diff", "--cached", "--name-only"), "")
                shutil.rmtree(self.tmp / "signals")
                shutil.rmtree(self.tmp / "positions", ignore_errors=True)

    def prepare_case(self, date, version, positions):
        if date == DAY:   # setUp에서 이미 이 브랜치에 있다
            self.git("switch", "-q", "main")
            self.git("branch", "-q", "-D", "forward/20261106")
        self.prepare(date, version, positions)

    def test_signals_and_positions_in_different_commits(self):
        date = "2026-11-20"
        self.prepare(date, "baseline", True)
        self.git("add", "signals")
        self.git("commit", "-q", "-m", "신호 먼저")
        self.git("add", "positions")
        self.git("commit", "-q", "-m", "비중 나중")
        self.assertRefused(date=date, now=lambda: pd.Timestamp(f"{date} 20:00", tz=ft.KST))
        self.assertTrue(io.read_table("forward_log").empty)

    def test_committed_signals_changed(self):
        io.write_table("signals", sample("signals"), date=DAY)
        self.git("add", "signals")
        self.git("commit", "-q", "-m", "신호 커밋됨")
        sig = sample("signals"); sig["rank_score"] = [0.8, 0.2]
        io.write_table("signals", sig, date=DAY)
        self.assertRefused()
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")

    def test_log_commit_failure_restores_csv(self):
        # 커밋이 실패해도 forward_log.csv에 행이 남지 않아 다시 실행할 수 있다
        io.write_table("signals", sample("signals"), date=DAY)
        hook = self.tmp / ".git" / "hooks" / "pre-commit"
        hook.write_text('#!/bin/sh\ngit diff --cached --name-only | grep -q forward_log && exit 1\nexit 0\n',
                        encoding="utf-8", newline="\n")
        hook.chmod(0o755)
        with self.assertRaisesRegex(ft.ForwardTestPartial, "되돌렸다"):
            self.record()
        self.assertTrue(io.read_table("forward_log").empty)
        hook.unlink()
        self.record()
        self.assertEqual(len(io.read_table("forward_log")), 1)

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


class CommitDeadlineTest(unittest.TestCase):
    def test_next_trading_day_0900(self):
        self.assertEqual(ft.commit_deadline("2026-11-06"), pd.Timestamp("2026-11-09 09:00", tz=ft.KST))   # 주말
        self.assertEqual(ft.commit_deadline("2026-11-11"), pd.Timestamp("2026-11-12 09:00", tz=ft.KST))
        self.assertEqual(ft.commit_deadline("2026-12-24"), pd.Timestamp("2026-12-28 09:00", tz=ft.KST))   # 12/25 휴장


class PastPositionsTest(RepoTest):
    def setUp(self):
        super().setUp()
        io.write_table("signals", sample("signals"), date=DAY)
        self.record()                                                    # 11/6: positions 없음
        for date, version in [("2026-11-20", "baseline"), ("2026-11-27", "lgbm_v1")]:
            self.prepare(date, version, positions=True)
            self.record(date, now=lambda: pd.Timestamp(f"{date} 20:00", tz=ft.KST))

    def test_only_earlier_records_with_positions(self):
        self.assertEqual(ft.past_positions("2026-11-20"), {})            # 11/20은 빈 보유로 시작
        self.assertEqual(list(ft.past_positions("2026-11-27")), ["2026-11-20"])
        got = ft.past_positions("2026-12-04")
        self.assertEqual(list(got), ["2026-11-20", "2026-11-27"])        # 날짜순
        io.validate("positions", got["2026-11-27"])
        self.assertTrue((got["2026-11-27"]["date"] == pd.Timestamp("2026-11-27")).all())

    def test_reads_tag_not_working_tree(self):
        self.git("switch", "-q", "main")                                 # main에는 기록 파일이 없다
        self.assertEqual(list(ft.past_positions("2026-12-04")), ["2026-11-20", "2026-11-27"])

    def test_fetches_tags_from_origin(self):
        self.git("tag", "-d", "ft-20261120")
        self.assertIn("2026-11-20", ft.past_positions("2026-12-04"))

    def test_ignores_other_tags(self):
        self.git("tag", "ft-rehearsal")
        self.git("push", "-q", "origin", "ft-rehearsal")
        self.assertEqual(list(ft.past_positions("2026-11-27")), ["2026-11-20"])

    def test_excludes_tag_left_by_failed_push(self):
        date = "2026-12-04"
        self.prepare(date, "lgbm_v1", positions=True)
        url = self.git("remote", "get-url", "origin")
        self.git("remote", "set-url", "origin", str(self.base / "no_such_remote.git"))
        with self.assertRaises(ft.ForwardTestPartial):
            self.record(date, now=lambda: pd.Timestamp(f"{date} 20:00", tz=ft.KST))
        self.git("remote", "set-url", "origin", url)
        self.assertTrue(self.git("tag", "--list", "ft-20261204"))        # 로컬에는 남아 있다
        self.assertEqual(list(ft.past_positions("2026-12-11")), ["2026-11-20", "2026-11-27"])

    def test_tag_removed_from_origin_is_dropped(self):
        self.assertIn("2026-11-27", ft.past_positions("2026-12-04"))
        subprocess.run(["git", "-C", str(self.remote), "tag", "-d", "ft-20261127"], check=True, capture_output=True)
        self.assertEqual(list(ft.past_positions("2026-12-04")), ["2026-11-20"])

    def test_invalid_file_in_tag(self):
        bad = sample("positions"); bad["date"] = pd.Timestamp("2026-12-04")
        (self.tmp / "positions").mkdir(exist_ok=True)
        bad.drop(columns="exposure").to_parquet(self.tmp / "positions" / "2026-12-04.parquet", index=False)
        self.git("add", "positions")
        self.git("commit", "-q", "-m", "bad")
        self.git("tag", "ft-20261204")
        self.git("push", "-q", "origin", "ft-20261204")
        with self.assertRaises(io.SchemaError):
            ft.past_positions("2026-12-11")

    def test_invalid_date(self):
        with self.assertRaises(ValueError):
            ft.past_positions("2026-13-01")


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
