"""B-08 G5·G7 리포트 (src/portfolio/g5_g7_report.py).

test_buffer_wf_report.py와 같은 합성 시장(시험 연도 2012·2013, 시드 0·1, M9 켬)에서 G5 비교 표를 만든 뒤 리포트를 만든다.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics as m5
from src.portfolio import buffer_wf_report as bwf
from src.portfolio import g5_g7_report as rep
from src.portfolio.engine_adapter import cost_rates
from tests.portfolio.test_buffer_wf_report import (  # noqa: F401 (pytest fixture)
    RUN as BUFFER_RUN,
    _schedule,
    ppo_env,
    rl_root,
    setup,
    synthetic,
)
from tests.portfolio.test_rl_wf_report import SEEDS, YEARS

RUN = "20261118_B_g5g7"
FILES = ["config.yaml", "decomposition.csv", "folds.csv", "g5_judgment.csv", "g7_check.csv", "git_commit.txt",
         "metrics.json", "rebound.csv", "report.md", "seeds.csv", "yearly.csv"]


@pytest.fixture
def b08(setup, monkeypatch):
    buffer_args, _, root, kept = setup
    bwf.main(buffer_args)
    schedule = _schedule(kept)
    monkeypatch.setattr(rep, "ROOT", root)
    monkeypatch.setattr(rep, "TABLES", bwf.TABLES)
    monkeypatch.setattr(rep, "input_fingerprints", bwf.input_fingerprints)
    monkeypatch.setattr(rep, "_git_commit", lambda: "abc\n")
    monkeypatch.setattr(rep, "LAST_SIGNAL_DATE", kept.signal_dates[-2])
    monkeypatch.setattr(rep, "m9_schedule", lambda run_id, first, last: schedule)
    args = ["--run-id", RUN, "--buffer-run-id", BUFFER_RUN, "--config", str(root / "prereg.yaml")]
    return args, root, kept


def _every_third_day(bm, risk, mode):
    """합성 시장 2012~2013에는 국면 1(급락 뒤 반등)인 날이 없다. 정렬을 보려고 사흘에 한 번 1로 둔다."""
    assert mode == "no_exposure"
    return pd.DataFrame({"regime_mc": (np.arange(len(bm)) % 3 == 0).astype(np.int8)}, index=bm.index)


def test_report_tables_follow_the_g5_table_and_c_functions(b08, monkeypatch):
    args, root, kept = b08
    monkeypatch.setattr(rep, "market_state_table", _every_third_day)
    out = rep.main(args)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert sorted(p.name for p in out.iterdir()) == FILES
    c = m["criteria"]
    assert c.pop("c07_g7_received") is False  # G7 판정은 C-07을 받아야 끝난다
    assert all(c.values()), c
    buf = json.loads((root / "reports" / BUFFER_RUN / "metrics.json").read_text(encoding="utf-8"))
    weekly = pd.read_csv(root / "reports" / BUFFER_RUN / "weekly_excess.csv")
    nav = pd.read_csv(root / "reports" / BUFFER_RUN / "daily_nav.csv", index_col="date", parse_dates=["date"])

    # G5 판정 표는 G5 비교 표의 판정을 그대로 옮긴다
    g5 = pd.read_csv(out / "g5_judgment.csv")
    assert list(g5["pass"]) == [buf["g5"]["pass_by_ir"], buf["g5"]["cagr_not_lower"],
                                buf["g5"]["turnover_ratio"] <= buf["g5"]["max_turnover_ratio"],
                                buf["g5"]["pass_by_alternative"], buf["g5"]["pass"]]
    assert g5.loc[0, "value"] == pytest.approx(buf["g5"]["ir_gain"])
    g7 = pd.read_csv(out / "g7_check.csv")
    assert g7.loc[0, "value"] == pytest.approx(buf["g7_check"]["mdd_reduction_rel"])
    assert bool(g7.loc[2, "pass"]) == buf["g7_check"]["pass"]

    # 반등 포착률 = C rebound_capture (일간 수익률 d−1→d, d−1 종가 regime_mc)
    days = kept.closes.index
    regime = _every_third_day(pd.Series(0.0, index=days), None, "no_exposure")["regime_mc"]
    regime = regime.reindex(nav.index).shift(1).iloc[1:]
    for name in ("buffer_full", "rl_s1"):
        p, b = nav[name].pct_change().iloc[1:], nav["KOSPI200"].pct_change().iloc[1:]
        want = m5.rebound_capture(p.to_numpy(), b.to_numpy(), regime.to_numpy().astype(int))
        assert m["rebound"][name]["rebound_capture"] == pytest.approx(want)
        assert m["rebound"][name]["days"] == int((regime == 1).sum()) > 0

    # 원인 분해: 세 몫의 합 = 주간 a_k 평균 × 52, 비용은 회전율 × 왕복 비용 근처
    rt = cost_rates(rep.load_prereg(root / "prereg.yaml"))
    round_trip = 2 * (rt.commission_per_side + rt.slippage_per_side) + rt.sell_tax
    for name, d in m["decomposition"].items():
        a = weekly.loc[weekly["portfolio"] == name, "a_k"]
        assert d["excess"] == pytest.approx(a.mean() * 52)
        assert d["cash"] + d["cost"] + d["selection"] == pytest.approx(d["excess"])
        assert 0.5 < -d["cost"] / (buf["runs"][name]["annual_turnover"] * round_trip) < 1.5
    assert m["decomposition"]["buffer_off"]["cash"] == pytest.approx(0.0, abs=1e-12)  # 미적용 버퍼는 현금이 없다
    assert m["decomposition"]["buffer_full"]["cash"] != 0.0

    # 시드별 결과와 연도별·폴드 표
    seeds = pd.read_csv(out / "seeds.csv", dtype={"seed": str}, keep_default_na=False)
    assert list(seeds["portfolio"]) == ["rl"] * (len(SEEDS) + 1) + list(rep.BUFFER_NAMES)
    assert list(seeds["seed"][: len(SEEDS) + 1]) == [str(s) for s in SEEDS] + ["mean"]
    assert seeds.loc[len(SEEDS), "rebound_capture"] == pytest.approx(seeds.loc[: len(SEEDS) - 1, "rebound_capture"].mean())
    years = pd.read_csv(out / "yearly.csv", index_col="year")
    assert list(years.index) == list(YEARS)
    assert np.prod(1 + years["buffer_full_return"]) == pytest.approx(nav["buffer_full"].iloc[-1])
    assert ((years["buffer_full_mean_exposure"] > 0) & (years["buffer_full_mean_exposure"] <= 1)).all()
    folds = pd.read_csv(out / "folds.csv")
    assert list(folds["fold"]) == list(YEARS)
    assert folds["val_ir_mean"].tolist() == pytest.approx([np.mean([0.1 * s for s in SEEDS])] * len(YEARS))
    fsi = pd.read_csv(root / "reports" / buf["rl_run"]["run_id"] / "fold_seed_ir.csv")
    assert folds["test_ir_mean"].tolist() == pytest.approx(fsi.groupby("fold")["ir"].mean().tolist())
    assert m["val_test_ir_correlation"]["folds"] is None  # 폴드 2개라 정의하지 않는다
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "## G5 판정 표 (7장)" in text and "| 2012 |" in text  # 연도는 정수로 쓴다

    with pytest.raises(SystemExit, match="R-11"):
        rep.main(args)


def test_a_replay_that_differs_from_the_g5_table_is_reported(b08):
    args, root, _ = b08
    path = root / "reports" / BUFFER_RUN / "weekly_excess.csv"
    w = pd.read_csv(path, dtype={"signal_date": str, "week": str})
    w.loc[w["portfolio"] == "buffer_off", "a_k"] += 1e-6
    w.to_csv(path, index=False)
    m = json.loads((rep.main(args) / "metrics.json").read_text(encoding="utf-8"))
    assert not m["criteria"]["replay_matches_g5_table"] and not m["criteria"]["b08_tables_complete"]
    assert m["replay"]["max_abs_diff"]["buffer_off"] == pytest.approx(1e-6, rel=1e-3)


def test_other_positions_inputs_or_config_stop_the_report(b08, monkeypatch):
    args, root, _ = b08
    with monkeypatch.context() as mp:
        mp.setattr(rep, "input_fingerprints", lambda t, r: {n: {"rows": len(df) + (n == "prices")} for n, df in t.items()})
        with pytest.raises(ValueError, match=r"입력 표.*prices"):
            rep.main(args)
    config = root / "prereg.yaml"
    original = config.read_bytes()
    config.write_bytes(original + b"\n# other\n")
    with pytest.raises(ValueError, match="config.yaml"):
        rep.main(args)
    config.write_bytes(original)
    path = root / "positions" / f"wf_{BUFFER_RUN}_off.parquet"
    pos = pd.read_parquet(path)
    pos.loc[pos.index[0], "target_weight"] *= 0.5
    pos.to_parquet(path)
    with pytest.raises(ValueError, match="buffer_off positions 지문"):
        rep.main(args)
    with pytest.raises(SystemExit, match="결과가 없다"):
        rep.main([*args[:3], "20261117_B_none", *args[4:]])
    assert not (root / "reports" / RUN).exists()


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------


def test_rebound_uses_the_previous_close_regime():
    days = pd.DatetimeIndex(["2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07"])
    nav = pd.Series([1.0, 1.1, 1.21, 1.0], index=days)
    bm = pd.Series([1.0, 1.05, 1.1025, 1.0], index=days)
    regime = pd.Series([0, 1, 1, 0], index=days)  # 1/5·1/6 종가에 국면 1 → 1/6·1/7 수익률을 쓴다
    r = rep.rebound(nav, bm, regime)
    want = ((1.21 / 1.1) * (1.0 / 1.21) - 1) / ((1.1025 / 1.05) * (1.0 / 1.1025) - 1)
    assert r["days"] == 2 and r["rebound_capture"] == pytest.approx(want)
    with pytest.raises(ValueError, match="regime_mc"):
        rep.rebound(nav, bm, regime.iloc[:2])


def test_decomposition_and_weekly_benchmark():
    d = rep.decomposition(np.array([0.01, -0.02]), np.array([0.5, 0.0]), np.array([0.02, -0.01]), np.array([0.001, 0.002]))
    assert d["cash"] == pytest.approx(-0.5 * 0.02 / 2 * 52) and d["cost"] == pytest.approx(-0.0015 * 52)
    assert d["cash"] + d["cost"] + d["selection"] == pytest.approx(d["excess"]) and d["mean_cash"] == 0.25
    days = pd.DatetimeIndex(["2016-01-08", "2016-01-11", "2016-01-15", "2016-01-18", "2016-01-22", "2016-01-25"])
    bm = pd.Series([100.0, 101.0, 102.0, 104.04, 103.0, 100.0], index=days)
    # 신호일 1/8·1/15 체결 → 1/11·1/18, 마지막 신호일 1/22의 다음 거래일 1/25까지
    assert rep.benchmark_weekly(bm, ["2016-01-08", "2016-01-15"], "2016-01-22").tolist() == pytest.approx(
        [104.04 / 101.0 - 1, 100.0 / 104.04 - 1])


def test_correlation_and_markdown_helpers():
    assert rep._corr([1, 2], [3, 4]) is None and rep._corr([1, 1, 1], [1, 2, 3]) is None
    assert rep._corr([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    table = rep.md_table(pd.DataFrame({"year": [2016], "cagr": [-0.0504], "ok": [True], "x": [None]}), pct=("cagr",))
    assert table.splitlines()[2] == "| 2016 | -5.0% | 통과 | − |"
