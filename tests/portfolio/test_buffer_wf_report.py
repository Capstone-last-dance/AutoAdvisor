"""B-07 G5 비교 표 (src/portfolio/buffer_wf_report.py): 버퍼 규칙 기준선 walk-forward와 RL 시험 운용 비교.

test_rl_wf_report.py와 같은 합성 시장(시험 연도 2012·2013, 시드 0·1)에서 RL 시험 운용을 모듈마다 한 번 돌리고,
테스트마다 그 결과를 복사해 같은 입력으로 버퍼 규칙 기준선과 G5 표를 만든다. M9는 합성 쿨다운·이벤트 청산 일정으로 켠다.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics as m5
from src.portfolio import buffer_wf_report as bwf
from src.portfolio import rl_wf_report as wf
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.rl_fold_report import ENGINE_TABLES, M9Schedule
from tests.portfolio.fakes import FakeEngine
from tests.portfolio.test_rl_wf_report import (  # noqa: F401 (ppo_env, synthetic는 pytest fixture)
    FOLD_RUNS,
    SEEDS,
    YEARS,
    _config,
    _fold_run,
    _test_signals,
    ppo_env,
    synthetic,
)

RL_RUN = "20261114_B_rl_wf"
RUN = "20261117_B_buffer_wf"
TEST_SIGNALS = "20261110_A_wfpm"
M9_RECORD = {"applied": True, "alerts_run_id": "20261112_PM_m9_test", "sha256": "ff", "alerts": 3}
MODES = ("full", "off", "no_exposure")
KPI = {"G5": {"min_ir_gain": 0.10, "alternative": {"cagr_not_lower": True, "max_turnover_ratio": 0.7},
              "aggregate": "mean_over_seeds"},
       "G7": {"min_mdd_reduction_rel": 0.25, "max_cagr_giveup": 0.02}}


def _schedule(kept, record=None) -> M9Schedule:
    """합성 M9 일정. 신호일 쿨다운은 합성 시장 값, 엔진 쿨다운은 그 다음 거래일, 이벤트 청산은 시험 신호일 네 번에
    한 번 rank 1위 종목(체결일 다음 거래일)이다."""
    days = kept.closes.index
    after = {d: days[days.searchsorted(pd.Timestamp(d), side="right"):] for d in kept.signal_dates[:-1]}
    cooldowns = {after[d][0].strftime("%Y-%m-%d"): frozenset(v) for d, v in kept.cooldown.items() if d in after}
    sig = kept.signals.assign(day=pd.to_datetime(kept.signals["date"]).dt.strftime("%Y-%m-%d"))
    liquidations = {}
    for d in [d for d in kept.signal_dates[:-1] if int(d[:4]) in YEARS][::4]:
        top = sig[(sig["day"] == d) & (sig["ticker"] != "000007")].sort_values("rank")["ticker"].iloc[0]
        liquidations[after[d][1].strftime("%Y-%m-%d")] = frozenset({top})
    return M9Schedule(lambda d: kept.cooldown.get(d, ()), cooldowns, liquidations, dict(record or M9_RECORD))


def _patch(mp, root, tables, kept, schedule):
    real_read = wf.io.read_table

    def read_table(name, date=None, run_id=None, stub=False):
        if date is None and run_id is None and not stub:
            return tables[name]
        return real_read(name, date=date, run_id=run_id, stub=stub)

    for mod in (wf, bwf):
        mp.setattr(mod, "ROOT", root)
        mp.setattr(mod, "TABLES", tuple(tables))
        mp.setattr(mod, "feature_table", lambda prices, bm, members, risk: kept.features)
        mp.setattr(mod, "input_fingerprints", lambda t, r: {n: {"rows": len(df)} for n, df in t.items()})
        mp.setattr(mod, "_git_commit", lambda: "abc\n")
        mp.setattr(mod, "LAST_SIGNAL_DATE", kept.signal_dates[-2])  # 합성 시장의 마지막 주는 주 중간에서 끝난다
        mp.setattr(mod, "m9_schedule", lambda run_id, first, last: schedule)
    mp.setattr(wf.io, "ROOT", root)
    mp.setattr(wf.io, "read_table", read_table)


@pytest.fixture(scope="module")
def rl_root(synthetic, cfg, tmp_path_factory, ppo_env):
    """M9를 켜고 학습한(것으로 꾸민) 폴드 결과로 RL 시험 운용을 끝까지 돌린 저장소."""
    tables, kept = synthetic
    root = tmp_path_factory.mktemp("rl")
    config = _config(cfg, root)
    with pytest.MonkeyPatch.context() as mp:
        _patch(mp, root, tables, kept, _schedule(kept))
        test_dates = [d for d in kept.signal_dates[:-1] if int(d[:4]) in YEARS]
        for y in YEARS:
            _fold_run(root, y, config, ppo_env, m9=dict(M9_RECORD))
        _test_signals(kept, test_dates)
        wf.main(["--run-id", RL_RUN, "--fold-runs", FOLD_RUNS, "--test-signals-run-id", TEST_SIGNALS,
                 "--config", str(config)])
    return root


@pytest.fixture
def setup(rl_root, synthetic, tmp_path, monkeypatch):
    tables, kept = synthetic
    root = tmp_path / "repo"
    shutil.copytree(rl_root, root)
    _patch(monkeypatch, root, tables, kept, _schedule(kept))
    test_dates = [d for d in kept.signal_dates[:-1] if int(d[:4]) in YEARS]
    args = ["--run-id", RUN, "--rl-run-id", RL_RUN, "--config", str(root / "prereg.yaml")]
    return args, test_dates, root, kept


def _read(out, name, **kw):
    return pd.read_csv(out / name, **kw)


def test_buffer_baseline_and_g5_table_use_the_rl_inputs(setup, synthetic):
    args, test_dates, root, kept = setup
    tables = synthetic[0]
    out = bwf.main(args)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert all(m["criteria"].values()), m["criteria"]
    assert m["replay"]["max_abs_diff"]["rl_s0"] <= bwf.REPLAY_TOLERANCE
    assert sorted(p.name for p in out.iterdir()) == ["config.yaml", "daily_nav.csv", "g5_table.csv", "git_commit.txt",
                                                     "metrics.json", "weekly_excess.csv"]

    pos = {mode: bwf.io.read_table("positions", run_id=f"{RUN}_{mode}") for mode in MODES}
    for mode, p in pos.items():
        assert sorted(p["date"].dt.strftime("%Y-%m-%d").unique()) == test_dates
        assert set(p["constructor"]) == {"buffer"}
        assert np.allclose(p.groupby("date")["target_weight"].sum(), 1.0)
        h = m["handoff"]["positions"][f"buffer_{mode}"]
        assert (h["run_id"], h["m8_mode"], h["rows"]) == (f"{RUN}_{mode}", mode, len(p))
        assert h["m9_alerts_run_id"] == (None if mode == "off" else M9_RECORD["alerts_run_id"])
    assert m["handoff"]["positions"]["rl_s1"]["m8_mode"] == "full"
    assert m["handoff"]["m9"]["alerts_run_id"] == M9_RECORD["alerts_run_id"] and "sector" in m["handoff"]["inputs"]
    # E_t: full만 노출 조절을 하고, no_exposure와 off는 1이다
    assert (pos["full"]["exposure"] < 1.0).any()
    assert (pos["no_exposure"]["exposure"] == 1.0).all() and (pos["off"]["exposure"] == 1.0).all()

    # M9 쿨다운: full·no_exposure는 쿨다운 종목을 목표에 넣지 않고, off(미적용)는 넣는다
    def cooled(p):
        stocks = p[p["ticker"] != "CASH"]
        day = stocks["date"].dt.strftime("%Y-%m-%d")
        return sum(t in kept.cooldown[d] for d, t in zip(day, stocks["ticker"]))
    assert cooled(pos["full"]) == 0 and cooled(pos["no_exposure"]) == 0 and cooled(pos["off"]) > 0

    # 엔진의 M9(이벤트 청산·체결일 쿨다운): full은 켜고 off(미적용)는 끈다
    weekly = _read(out, "weekly_excess.csv", dtype={"signal_date": str, "week": str})
    assert set(weekly["portfolio"]) == {*(f"buffer_{x}" for x in MODES), *(f"rl_s{s}" for s in SEEDS)}
    inputs = engine_inputs(*(tables[n] for n in ENGINE_TABLES))
    cfg_now, sched = bwf.load_prereg(root / "prereg.yaml"), _schedule(kept)
    for mode in ("full", "off"):
        got = weekly.loc[weekly["portfolio"] == f"buffer_{mode}", "a_k"].to_numpy()
        plain = bwf.replay(pos[mode], test_dates, build_engine(inputs, cfg_now, record_ledger=True)).excess
        m9 = bwf.replay(pos[mode], test_dates, build_engine(inputs, cfg_now, record_ledger=True, cooldowns=sched.cooldowns,
                                                            event_liquidations=sched.liquidations)).excess
        same, other = (m9, plain) if mode == "full" else (plain, m9)
        np.testing.assert_allclose(got, same, rtol=0, atol=1e-12)
        assert np.abs(got - other).max() > 1e-6

    # RL을 다시 체결한 a_k는 RL 기록과 같다
    recorded = _read(root / "reports" / RL_RUN, "weekly_excess.csv")
    for s in SEEDS:
        np.testing.assert_allclose(weekly.loc[weekly["portfolio"] == f"rl_s{s}", "a_k"],
                                   recorded.loc[recorded["seed"] == s, "a_k"], rtol=0, atol=1e-12)

    # 표의 지표는 C의 M5 함수로 다시 계산한 값과 같다
    nav = _read(out, "daily_nav.csv", index_col="date")
    days = kept.closes.index
    assert nav.index[0] == wf._next_trading_day(days, test_dates[0])  # 첫 체결일
    assert nav.index[-1] == wf._next_trading_day(days, test_dates[-1])  # 마지막 평가일
    n = len(nav) - 1
    table = _read(out, "g5_table.csv", dtype={"seed": str}, keep_default_na=False)
    assert list(table["portfolio"]) == ["rl"] * (len(SEEDS) + 1) + ["buffer"]
    assert list(table["seed"]) == [str(s) for s in SEEDS] + ["mean", ""]
    buf = table[table["portfolio"] == "buffer"].iloc[0]
    a = weekly[weekly["portfolio"] == "buffer_full"]
    assert len(a) == len(test_dates) - 1  # 마지막 시험 신호일은 체결하지 않는다
    assert buf["ir"] == pytest.approx(m5.information_ratio(a["a_k"]))
    assert buf["cagr"] == pytest.approx(nav["buffer_full"].iloc[-1] ** (252 / n) - 1)
    assert buf["annual_turnover"] == pytest.approx(a["turnover"].sum() / (n / 252))
    assert buf["mdd"] == pytest.approx(m5.max_drawdown(np.r_[1.0, nav["buffer_full"]]))
    bm_cagr = (nav["KOSPI200"].iloc[-1] / nav["KOSPI200"].iloc[0]) ** (252 / n) - 1
    assert buf["annual_excess"] == pytest.approx(buf["cagr"] - bm_cagr)
    mean = table[table["seed"] == "mean"].iloc[0]
    assert mean["ir"] == pytest.approx(table.loc[table["seed"].isin(["0", "1"]), "ir"].mean())

    g5 = m["g5"]
    assert g5["ir_gain"] == pytest.approx(mean["ir"] - buf["ir"])
    assert g5["turnover_ratio"] == pytest.approx(mean["annual_turnover"] / buf["annual_turnover"])
    assert g5["pass"] == (g5["pass_by_ir"] or g5["pass_by_alternative"])
    g7 = m["g7_check"]
    assert g7["mdd_applied"] == pytest.approx(m["runs"]["buffer_full"]["mdd"])
    assert g7["mdd_unapplied"] == pytest.approx(m["runs"]["buffer_off"]["mdd"])
    assert m["g0"]["violations"] == {n: 0 for n in m["g0"]["violations"]} and len(m["g0"]["violations"]) == 5

    with pytest.raises(SystemExit, match="R-11"):
        bwf.main(args)


def test_a_changed_rl_record_is_reported_not_hidden(setup):
    """RL 기록의 a_k와 다시 체결한 a_k가 다르면 멈추지 않고 완료 기준에 적는다."""
    args, _, root, _ = setup
    path = root / "reports" / RL_RUN / "weekly_excess.csv"
    w = pd.read_csv(path, dtype={"signal_date": str, "week": str})
    w.loc[w.index[-1], "a_k"] += 1e-6
    w.to_csv(path, index=False)
    out = bwf.main(args)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert not m["criteria"]["rl_replay_matches_record"] and not m["criteria"]["g5_table_complete"]
    assert m["replay"]["max_abs_diff"]["rl_s1"] == pytest.approx(1e-6, rel=1e-3)


def test_other_inputs_than_the_rl_run_stop_the_run(setup, monkeypatch, synthetic):
    args, _, root, kept = setup
    with monkeypatch.context() as mp:
        mp.setattr(bwf, "input_fingerprints", lambda t, r: {n: {"rows": len(df) + (n == "sector")} for n, df in t.items()})
        with pytest.raises(ValueError, match=r"입력 표.*sector"):
            bwf.main(args)
    with monkeypatch.context() as mp:
        other = kept.features.assign(vol_20=kept.features["vol_20"] * 0.5)
        mp.setattr(bwf, "feature_table", lambda prices, bm, members, risk: other)
        with pytest.raises(ValueError, match="features 지문"):
            bwf.main(args)
    with monkeypatch.context() as mp:
        schedule = _schedule(kept, {**M9_RECORD, "sha256": "ee"})
        mp.setattr(bwf, "m9_schedule", lambda run_id, first, last: schedule)
        with pytest.raises(ValueError, match="M9 경보 파일"):
            bwf.main(args)
    assert not (root / "reports" / RUN).exists()


def test_other_config_or_rl_positions_stop_the_run(setup):
    args, _, root, _ = setup
    config = root / "prereg.yaml"
    original = config.read_bytes()
    config.write_bytes(original + b"\n# other\n")
    with pytest.raises(ValueError, match="config.yaml"):
        bwf.main(args)
    config.write_bytes(original)
    path = root / "positions" / f"wf_{RL_RUN}_s0.parquet"
    pos = pd.read_parquet(path)
    pos.loc[pos.index[0], "target_weight"] *= 0.5
    pos.to_parquet(path)
    with pytest.raises(ValueError, match="RL positions 시드 0 지문"):
        bwf.main(args)
    with pytest.raises(SystemExit, match="결과가 없다"):
        bwf.main([*args[:3], "20261114_B_none", *args[4:]])
    assert not (root / "reports" / RUN).exists()


def test_rl_run_without_m9_is_not_a_g5_comparison(setup):
    args, _, root, _ = setup
    path = root / "reports" / RL_RUN / "metrics.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    m["m9"] = {"applied": False}
    path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="M9 없이"):
        bwf.main(args)


# ---------------------------------------------------------------------------
# 판정과 보조 함수
# ---------------------------------------------------------------------------


def _s(ir, cagr, turnover):
    return {"ir": ir, "cagr": cagr, "annual_turnover": turnover}


RL = {0: _s(0.5, 0.05, 4.0), 1: _s(0.3, 0.03, 6.0)}  # 5시드 평균 대신 2시드 평균: IR 0.4, CAGR 0.04, 회전율 5.0


@pytest.mark.parametrize(
    ("buffer", "by_ir", "by_alt"),
    [
        (_s(0.25, 0.06, 5.0), True, False),  # IR 차 0.15 ≥ 0.10
        (_s(0.35, 0.039, 8.0), False, True),  # CAGR 0.04 ≥ 0.039, 회전율 비 0.625 ≤ 0.7
        (_s(0.35, 0.041, 8.0), False, False),  # CAGR이 낮다
        (_s(0.35, 0.03, 7.0), False, False),  # 회전율 비 0.714 > 0.7
    ],
)
def test_g5_judgment(buffer, by_ir, by_alt):
    g = bwf.g5_judgment(RL, buffer, KPI)
    assert g["rl_mean_over_seeds"] == pytest.approx({"ir": 0.4, "cagr": 0.04, "annual_turnover": 5.0})
    assert (g["pass_by_ir"], g["pass_by_alternative"], g["pass"]) == (by_ir, by_alt, by_ir or by_alt)


def test_g5_judgment_needs_defined_ir_and_the_preregistered_aggregate():
    with pytest.raises(ValueError, match="정의되지 않은"):
        bwf.g5_judgment({0: _s(None, 0.0, 1.0)}, _s(0.1, 0.0, 1.0), KPI)
    other = {**KPI, "G5": {**KPI["G5"], "aggregate": "median_over_seeds"}}
    with pytest.raises(ValueError):
        bwf.g5_judgment(RL, _s(0.1, 0.0, 1.0), other)


def test_g7_check():
    g = bwf.g7_check({"mdd": 0.15, "cagr": 0.05}, {"mdd": 0.25, "cagr": 0.06}, KPI)
    assert g["mdd_reduction_rel"] == pytest.approx(0.4) and g["pass"]
    assert not bwf.g7_check({"mdd": 0.15, "cagr": 0.03}, {"mdd": 0.25, "cagr": 0.06}, KPI)["pass"]  # CAGR 포기 0.03
    assert not bwf.g7_check({"mdd": 0.20, "cagr": 0.06}, {"mdd": 0.25, "cagr": 0.06}, KPI)["pass"]  # 낙폭 감소 0.2


def test_daily_nav_takes_the_last_record_of_each_day():
    """구간 끝 평가일(체결 전 NAV)과 다음 구간 첫 체결일(체결 뒤 NAV)이 같은 날이면 뒤 기록을 쓴다."""
    ledger = [
        {"date": "2016-01-11", "nav_before": 1.0, "nav_after": 0.999},
        {"date": "2016-01-12", "nav_before": 1.01, "nav_after": 1.01},
        {"date": "2016-01-18", "nav_before": 1.02, "nav_after": 1.02},
        {"date": "2016-01-18", "nav_before": 1.02, "nav_after": 1.018},
    ]
    nav = bwf.daily_nav(ledger)
    assert list(nav.index.strftime("%Y-%m-%d")) == ["2016-01-11", "2016-01-12", "2016-01-18"]
    assert list(nav) == [0.999, 1.01, 1.018]


def test_replay_needs_a_fresh_engine_with_a_ledger(market, cfg):
    positions = pd.DataFrame({"date": pd.to_datetime(["2012-01-06"] * 2), "ticker": ["000001", "CASH"],
                              "target_weight": [0.5, 0.5]})
    with pytest.raises(ValueError, match="record_ledger"):
        bwf.replay(positions, ["2012-01-06"], FakeEngine(market.closes, market.bm_close, cfg["costs"]))
    assert bwf.targets_by_date(positions) == {"2012-01-06": {"000001": 0.5}}
