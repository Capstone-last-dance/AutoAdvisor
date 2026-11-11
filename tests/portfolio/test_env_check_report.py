"""B-05 리포트 (src/portfolio/env_check_report.py): 실제 A 베이스라인 신호로 완료 기준 (1)·(2)를 남기는 실행."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.portfolio import env_check_report as report
from src.portfolio.risk_layer import market_state_table
from tests.portfolio.conftest import VOL_TARGET_FOR_TESTS
from tests.portfolio.fakes import market_tables

HAS_A_FEATURES = importlib.util.find_spec("src.features.price_features") is not None


@pytest.fixture(scope="module")
def cfg_with_vol_target(cfg):
    c = copy.deepcopy(cfg)
    c["risk_layer"]["vol_target_annual"] = VOL_TARGET_FOR_TESTS  # 테스트용 임의값 (conftest와 같다)
    return c


@pytest.fixture(scope="module")
def synthetic(market):
    tables, kept = market_tables(market)
    tables = {
        **tables,
        "delisted_reused": pd.DataFrame(columns=["ticker", "delist_date", "last_price"]).astype({"last_price": "float64"}),
        "universe": kept.universe, "sector": kept.sector,
    }
    return tables, kept


def test_signal_dates_when_the_end_is_a_holiday(market):
    """끝날이 휴장일이면(실제 2025-12-31) 그 주의 마지막 거래일은 끝날 뒤라 신호일에 들지 않는다.
    거래일을 끝날에서 자르고 주마다 마지막 날을 세면 그 주 화·수요일을 신호일로 잘못 센다."""
    days = market.bm_close.index.drop(pd.Timestamp("2012-12-27"))  # 목요일 휴장, 그 주 마지막 거래일은 12/28
    got = report.signal_dates(market.universe, days, "2011-01-01", "2012-12-27")
    assert got == [d for d in market.signal_dates if d <= "2012-12-27"]
    assert got[-1] == "2012-12-21"


def test_signal_dates_stop_when_universe_differs_from_the_calendar(market):
    days = market.bm_close.index
    u = market.universe
    missing = u[pd.to_datetime(u["date"]) != pd.Timestamp(market.signal_dates[3])]
    with pytest.raises(ValueError, match="빠진 신호일"):
        report.signal_dates(missing, days, "2011-01-01", "2011-12-31")
    extra = pd.concat([u, u.head(1).assign(date=pd.Timestamp("2011-03-02").date())], ignore_index=True)  # 수요일
    with pytest.raises(ValueError, match="신호일이 아닌 날짜: \\['2011-03-02'\\]"):
        report.signal_dates(extra, days, "2011-01-01", "2011-12-31")


def test_metrics_pass_both_criteria_on_a_synthetic_market(synthetic, cfg_with_vol_target):
    tables, kept = synthetic
    dates = kept.signal_dates
    m = report.env_check_metrics(tables, cfg_with_vol_target, dates, kept.signals, kept.features, seed=3)
    assert m["criteria"]["1_episode_runs_to_the_end"] and m["criteria"]["2_same_weights_same_fills_and_costs"]
    for e in m["episodes"].values():
        assert (e["n_steps"], e["first_date"], e["last_date"]) == (len(dates), dates[0], dates[-1])
    assert m["episodes"]["zero_action"]["nav_end"] != m["episodes"]["random_action"]["nav_end"]
    assert m["buffer_vs_env"]["ledger_counts"]["rejected_halted"] > 0  # 정지 거부가 실제로 일어났다
    assert m["random_action_seed"] == 3 and "M9" in m["m9"]


def test_a_difference_fails_criterion_2(synthetic, cfg_with_vol_target, monkeypatch):
    tables, kept = synthetic
    real = report.compare_buffer_and_env

    def off_by_a_little(*args, **kwargs):
        return {**real(*args, **kwargs), "max_fill_diff": 1e-9}

    monkeypatch.setattr(report, "compare_buffer_and_env", off_by_a_little)
    dates = kept.signal_dates[:30]
    m = report.env_check_metrics(tables, cfg_with_vol_target, dates, kept.signals, kept.features)
    assert m["criteria"]["1_episode_runs_to_the_end"] and not m["criteria"]["2_same_weights_same_fills_and_costs"]


def _priced(tables, seed=0):
    rng = np.random.default_rng(seed)
    p = tables["prices"].copy()
    p["value"] = rng.uniform(1e8, 1e9, len(p))
    p["mcap"] = rng.uniform(1e11, 1e12, len(p))
    return p


@pytest.mark.skipif(not HAS_A_FEATURES, reason="A PR #35(src.features.price_features) 병합 전")
def test_feature_table_uses_the_a_definitions(synthetic, risk_params, monkeypatch):
    """vol_20 = 20일 로그수익률 표준편차(ddof=1) × √252의 U_t 안 백분위(결측 0.5) (6.2절). 종목을 나눠 계산해도 같다."""
    tables, kept = synthetic
    prices = _priced(tables)
    dates = kept.signal_dates[:40]
    members = report.members_by_date(kept.universe, dates)
    got = report.feature_table(prices, kept.bm_close, members, risk_params)

    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    vol = np.log(close / close.shift(1)).rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    bm_ret = np.log(kept.bm_close / kept.bm_close.shift(1))
    regime = market_state_table(kept.bm_close, risk_params, mode="no_exposure")["regime_mc"]
    for t, tickers in members.items():
        g = got[got["date"] == t].set_index("ticker").loc[tickers]
        want = vol.loc[t].reindex(tickers).rank(pct=True, method="average").fillna(0.5)
        np.testing.assert_allclose(g["vol_20"], want, rtol=0, atol=1e-12)
        assert g["mkt_vol_20"].iloc[0] == pytest.approx(bm_ret.loc[:t].iloc[-20:].std(ddof=1) * np.sqrt(252), abs=1e-12)
        assert g["mkt_ret_20"].iloc[0] == pytest.approx(kept.bm_close.loc[t] / kept.bm_close.loc[:t].iloc[-21] - 1, abs=1e-12)
        assert (g["regime_mc"] == int(regime.loc[t])).all()

    monkeypatch.setattr(report, "CHUNK", 7)
    pd.testing.assert_frame_equal(report.feature_table(prices, kept.bm_close, members, risk_params), got)


def test_input_fingerprints_are_file_sha256(tmp_path):
    folder = tmp_path / "data" / "processed"
    folder.mkdir(parents=True)
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_parquet(folder / "index.parquet")
    got = report.input_fingerprints({"index": df}, tmp_path)
    assert got == {"index": {"path": "data/processed/index.parquet", "rows": 3,
                             "sha256": hashlib.sha256((folder / "index.parquet").read_bytes()).hexdigest()}}


def test_main_writes_the_run_folder_once(synthetic, cfg_with_vol_target, tmp_path, monkeypatch):
    tables, kept = synthetic
    config = tmp_path / "prereg.yaml"
    config.write_text(yaml.safe_dump(cfg_with_vol_target, allow_unicode=True), encoding="utf-8")
    by_date = {pd.Timestamp(d): g for d, g in kept.signals.groupby(pd.to_datetime(kept.signals["date"]))}
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "TABLES", tuple(tables))
    monkeypatch.setattr(report.io, "read_table", lambda name: tables[name])
    monkeypatch.setattr(report, "_a_signal_function", lambda: lambda close, tickers, t: by_date[t])
    monkeypatch.setattr(report, "feature_table", lambda prices, bm, members, risk: kept.features)
    monkeypatch.setattr(report, "input_fingerprints", lambda t, root: {n: {"rows": len(df)} for n, df in t.items()})
    def commit():
        assert not (tmp_path / "reports" / "20261109_B_test").exists()  # 결과 폴더가 새 파일 목록에 들지 않게
        return "abc\n"

    monkeypatch.setattr(report, "_git_commit", commit)
    monkeypatch.setattr(report, "LAST_SIGNAL_DATE", "2011-12-31")  # 합성 시장 1년만 (빠르게)

    out = report.main(["--run-id", "20261109_B_test", "--config", str(config)])
    assert sorted(p.name for p in out.iterdir()) == ["config.yaml", "git_commit.txt", "metrics.json"]
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["task"] == "B-05" and all(v for k, v in m["criteria"].items() if k != "tolerance")
    assert m["signal_dates"]["last"] <= "2011-12-31" and m["generated"]["signals"]["model_version"] == ["synthetic"]
    with pytest.raises(SystemExit, match="R-11"):
        report.main(["--run-id", "20261109_B_test", "--config", str(config)])
    with pytest.raises(SystemExit, match="형식"):
        report.main(["--run-id", "20261109_A_test", "--config", str(config)])
