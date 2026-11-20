"""src/pipeline/publish_dashboard.py 테스트 (작업 D-08).

임시 git 저장소 세 개(원본, 원본의 origin, 비공개 배포 저장소)로 실제 git 명령을 돌린다.
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.common import io
from src.pipeline import publish_dashboard as pd_
from src.pipeline.make_dashboard_sample import write_sample

REPO = Path(__file__).resolve().parents[2]
DAY = "2026-11-06"          # 샘플 forward_log에 기록된 신호일
DIAGNOSTIC = "2026-11-13"   # 샘플에 신호 파일은 있지만 공식 기록이 아닌 날


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()


class PublishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._root = io.ROOT
        self.src = self.tmp / "src"
        write_sample(self.src)
        (self.src / "data" / "processed" / "news.parquet").write_bytes(b"news")   # 올리면 안 되는 파일
        (self.src / "reports" / "final").mkdir(parents=True, exist_ok=True)
        (self.src / "reports" / "final" / "kpi.csv").write_text("gate\n", encoding="utf-8")
        for rel in [*pd_.CODE_FILES, "requirements.txt",
                    *subprocess.run(["git", "-C", str(REPO), "ls-files", "app"], capture_output=True,
                                    text=True, encoding="utf-8", check=True).stdout.split()]:
            (self.src / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, self.src / rel)

        git(self.tmp, "init", "-q", "--bare", "origin.git")
        git(self.src, "init", "-q")
        git(self.src, "remote", "add", "origin", str(self.tmp / "origin.git"))
        git(self.src, "add", *pd_.CODE_FILES, "app", "requirements.txt")
        git(self.src, "commit", "-q", "-m", "code")
        git(self.src, "add", f"signals/{DAY}.parquet")
        git(self.src, "commit", "-q", "-m", "signal")
        git(self.src, "tag", "ft-20261106")
        git(self.src, "push", "-q", "origin", "HEAD:refs/heads/main", "refs/tags/ft-20261106")

        git(self.tmp, "init", "-q", "--bare", "deploy.git")
        git(self.tmp, "clone", "-q", str(self.tmp / "deploy.git"), "deploy")
        self.target = self.tmp / "deploy"
        git(self.target, "config", "user.name", "t")
        git(self.target, "config", "user.email", "t@t")
        io.ROOT = self.src

    def tearDown(self):
        io.ROOT = self._root
        shutil.rmtree(self.tmp, ignore_errors=True)

    def deployed(self):
        return set(git(self.tmp / "deploy.git", "ls-tree", "-r", "--name-only", "HEAD").splitlines())

    def test_publishes_code_and_official_outputs_only(self):
        pd_.publish(DAY, self.target)
        files = self.deployed()
        self.assertIn("app/Home.py", files)
        self.assertIn("src/common/io.py", files)
        self.assertIn(f"signals/{DAY}.parquet", files)
        self.assertNotIn(f"signals/{DIAGNOSTIC}.parquet", files)        # 진단용 신호는 올리지 않는다
        self.assertNotIn(f"positions/{DIAGNOSTIC}.parquet", files)
        self.assertNotIn(f"evidence/{DIAGNOSTIC}.jsonl", files)
        self.assertIn(f"alerts/{DAY}.parquet", files)
        self.assertFalse(any(f.startswith("alerts/") and f > f"alerts/{DAY}.parquet" for f in files))
        self.assertIn("data/processed/index.parquet", files)
        self.assertNotIn("data/processed/news.parquet", files)           # 뉴스 원문은 빼야 한다
        self.assertIn("reports/final/kpi.csv", files)
        self.assertIn("forward_log.csv", files)
        req = (self.target / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("streamlit==", req)
        self.assertNotIn("torch", req)
        self.assertIn(f"배포: 신호일 {DAY}", git(self.tmp / "deploy.git", "log", "-1", "--format=%s"))

    def test_deployed_dashboard_imports(self):
        # 올린 코드만으로 대시보드가 import되는지 본다 (CODE_FILES 누락 확인)
        pd_.publish(DAY, self.target, push=False)
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, 'app'); import dashboard; "
                                   "from src.events import alerts; alerts.load_trading_days()"],
            cwd=self.target, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_rerun_without_changes_makes_no_commit(self):
        first = pd_.publish(DAY, self.target)
        self.assertEqual(pd_.publish(DAY, self.target), first)

    def test_rejects_unrecorded_date(self):
        with self.assertRaisesRegex(pd_.PublishError, "forward_log"):
            pd_.publish(DIAGNOSTIC, self.target)

    def test_rejects_unpublished_tag(self):
        git(self.tmp / "origin.git", "tag", "-d", "ft-20261106")
        with self.assertRaisesRegex(pd_.PublishError, "origin에 없다"):
            pd_.publish(DAY, self.target)

    def test_rejects_public_repo_as_target(self):
        git(self.target, "remote", "set-url", "origin", str(self.tmp / "origin.git") + "/")
        with self.assertRaisesRegex(pd_.PublishError, "공개 저장소"):
            pd_.publish(DAY, self.target)
        self.assertEqual(self.deployed_or_empty(), set())

    def test_rejects_uncommitted_app_code(self):
        (self.src / "app" / "Home.py").write_text("# 수정\n", encoding="utf-8")
        with self.assertRaisesRegex(pd_.PublishError, "커밋하지 않은 변경"):
            pd_.publish(DAY, self.target)

    def test_cli_no_push(self):
        self.assertEqual(pd_.main(["--date", DAY, "--target", str(self.target), "--no-push"]), 0)
        self.assertEqual(self.deployed_or_empty(), set())
        self.assertIn(f"signals/{DAY}.parquet", git(self.target, "ls-tree", "-r", "--name-only", "HEAD"))

    def deployed_or_empty(self):
        try:
            return self.deployed()
        except subprocess.CalledProcessError:
            return set()


if __name__ == "__main__":
    unittest.main()
