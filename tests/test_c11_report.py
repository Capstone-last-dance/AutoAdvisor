"""2026 개봉 가드, C 엔진 재생, 현금 보정과 실행 지문의 핵심 경계."""

import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest.accounting import CostRates
from src.backtest.c11_report import check_open, replay_candidate, write_run_record
from src.backtest.engine import ExecutionEngine
from src.common.guards import GuardViolation


def test_dec38_open_requires_date_and_explicit_flag():
    cfg = {"splits": {"holdout_signal_range": ["2026-01-01", "2026-10-31"],
                      "holdout_open_date": "2026-12-01"}}
    for day, flag in [(30, True), (1, False)]:
        month = 11 if day == 30 else 12
        with pytest.raises(GuardViolation):
            check_open(cfg, run_at=datetime(2026, month, day, 9, tzinfo=ZoneInfo("Asia/Seoul")),
                       allow_holdout=flag)
    check_open(cfg, run_at=datetime(2026, 12, 1, 9, tzinfo=ZoneInfo("Asia/Seoul")),
               allow_holdout=True)


def test_replay_uses_same_engine_and_keeps_both_cash_variants():
    days = pd.bdate_range("2026-01-05", periods=6)
    close = pd.DataFrame({"005930": [100.0] * 6}, index=days)
    bm = pd.Series([100.0] * 6, index=days)
    engine = ExecutionEngine(close, bm, CostRates(0, 0, 0), record_ledger=True)
    positions = pd.DataFrame([
        {"date": days[0], "ticker": "005930", "target_weight": 0.5},
        {"date": days[0], "ticker": "CASH", "target_weight": 0.5},
        {"date": days[2], "ticker": "005930", "target_weight": 0.5},
        {"date": days[2], "ticker": "CASH", "target_weight": 0.5},
    ])
    monthly = pd.Series([0.0365], index=pd.PeriodIndex(["2026-01"], freq="M"))
    report, daily = replay_candidate(positions, engine, monthly,
                                     end_date=days[-1].strftime("%Y-%m-%d"), benchmark=bm)
    assert report["signal_days"] == 2
    assert report["last_day"] == days[-1].strftime("%Y-%m-%d")
    assert report["cash0"]["final_nav"] == pytest.approx(1.0)
    assert report["cd91"]["final_nav"] > 1.0
    assert len(daily) == 5
    assert daily["cash_fraction"].iloc[0] == pytest.approx(0.5)


def test_last_signal_is_not_filled_when_its_fill_is_end_close():
    days = pd.bdate_range("2026-11-24", periods=5)
    close = pd.DataFrame({"005930": [100.0] * 5}, index=days)
    bm = pd.Series([100.0] * 5, index=days)
    engine = ExecutionEngine(close, bm, CostRates(0, 0.01, 0), record_ledger=True)
    positions = pd.DataFrame([
        {"date": days[0], "ticker": "CASH", "target_weight": 1.0},
        {"date": days[-2], "ticker": "005930", "target_weight": 1.0},
        {"date": days[-2], "ticker": "CASH", "target_weight": 0.0},
    ])
    monthly = pd.Series([0.03], index=pd.PeriodIndex(["2026-10"], freq="M"))
    report, daily = replay_candidate(positions, engine, monthly,
                                     end_date=days[-1].strftime("%Y-%m-%d"), benchmark=bm)
    assert report["last_signal_unexecuted_at_end"] is True
    assert report["executed_signals"] == 1
    assert report["cd91_filled_months"] == ["2026-11"]
    assert report["cash0"]["final_nav"] == pytest.approx(1.0)
    assert daily["cash_fraction"].iloc[-1] == pytest.approx(1.0)


def test_record_is_fingerprinted_and_never_overwritten(tmp_path):
    cfg = tmp_path / "prereg.yaml"
    source = tmp_path / "prices.parquet"
    cfg.write_text("holdout_open_date: 2026-12-01\n", encoding="utf-8")
    source.write_bytes(b"fixed input")
    out = tmp_path / "reports" / "20261201_C_c11"
    run_at = datetime(2026, 12, 1, 8, tzinfo=ZoneInfo("Asia/Seoul"))
    daily = pd.DataFrame({"date": ["2026-11-30"], "nav_cd91": [1.0]})
    kwargs = dict(config_path=cfg, inputs={"prices": source}, metrics={"candidate": {"cagr": 0.1}},
                  daily_nav={"candidate": daily}, run_at=run_at, allow_holdout=True)
    write_run_record(out, **kwargs)
    record = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert record["run_at_kst"] == run_at.isoformat()
    assert record["inputs_sha256"]["prices"] == hashlib.sha256(b"fixed input").hexdigest()
    assert (out / "config.yaml").read_bytes() == cfg.read_bytes()
    with pytest.raises(FileExistsError):
        write_run_record(out, **kwargs)
