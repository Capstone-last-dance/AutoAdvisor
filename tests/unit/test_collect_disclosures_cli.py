"""실제 네트워크 없이 CLI 명령과 종료 코드를 검증한다."""

import json
from datetime import date
from pathlib import Path

import pytest

from conftest import item, page
from src.data import collect_disclosures as cli
from src.data.disclosures import plan_run
from src.data.disclosure_archive import RawArchive


def test_help_without_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENDART_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "coverage" in capsys.readouterr().out


def test_cli_plan_collect_build_and_publish_refusal(tmp_path, monkeypatch, client_factory):
    monkeypatch.chdir(tmp_path)
    client, session, _ = client_factory([page([item(1)], total=2, pages=2),
                                        page([item(2)], total=2, pages=2, number=2)])
    monkeypatch.setattr(cli, "DartClient", lambda **kwargs: client)
    assert cli.main(["plan", "--start", "2025-01-01", "--end", "2025-03-31", "--types", "B", "--run-id", "20261028_PM_disclosures_backfill"]) == 0
    assert len(session.requests) == 1
    assert cli.main(["collect", "--run-id", "20261028_PM_disclosures_backfill"]) == 0
    assert [r[-1] for r in session.requests] == [1, 2]
    assert cli.main(["status", "--run-id", "20261028_PM_disclosures_backfill"]) == 0
    assert cli.main(["build"]) == 0
    assert len(list(Path("reports").glob("*/metrics.json"))) == 1
    assert not Path("data/processed/disclosures.parquet").exists()
    assert cli.main(["build", "--publish"]) == 3
    assert not Path("data/processed/disclosures.parquet").exists()


@pytest.mark.parametrize("failure,code,reason", [(KeyboardInterrupt(), 2, "interrupted"),
                                                  (page(status="020"), 2, "rate_limit"),
                                                  (page(status="010"), 1, "auth"),
                                                  (page(status="100"), 1, "request_error")])
def test_cli_stop_codes(tmp_path, monkeypatch, client_factory, failure, code, reason):
    monkeypatch.chdir(tmp_path)
    client, _, _ = client_factory([page([item(1)], total=2, pages=2), failure])
    monkeypatch.setattr(cli, "DartClient", lambda **kwargs: client)
    assert cli.main(["collect", "--start", "2025-01-01", "--end", "2025-03-31", "--types", "B"]) == code
    manifest = json.loads(next(Path("data/raw").rglob("manifest.json")).read_text(encoding="utf-8"))
    assert manifest["stop_reason"] == reason


def test_schema_snapshot_matches_appendix():
    import re
    spec = Path("docs/프로젝트_구조_및_역할분담_v0.3.md").read_text(encoding="utf-8")
    expected = re.search(r"## 부록 1\..*?```yaml\n(.*?)```", spec, re.S).group(1)
    assert Path("configs/prereg.yaml").read_text(encoding="utf-8") == expected


def test_complete_build_is_diagnostic_and_publish_refuses_missing_io(tmp_path, monkeypatch, client_factory, capsys):
    from src.data import disclosures
    monkeypatch.chdir(tmp_path)
    client, _, _ = client_factory([page([item(day="20100104")])])
    monkeypatch.setattr(cli, "DartClient", lambda **kwargs: client)
    monkeypatch.setattr(disclosures, "_common_io", lambda: None)
    assert cli.main(["collect", "--start", "2010-01-01", "--end", "2010-03-31", "--types", "B"]) == 0
    assert cli.main(["build"]) == 0
    metrics = json.loads(next(Path("reports").glob("*/metrics.json")).read_text(encoding="utf-8"))
    assert metrics["diagnostic_only"] and not metrics["public_publish_enabled"]
    assert cli.main(["build", "--publish"]) == 3
    assert "R-18" in capsys.readouterr().err
    assert not Path("data/processed/disclosures.parquet").exists()
