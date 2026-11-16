"""B-07 (src/portfolio/rl_wf_report.py): 10폴드 정책의 시험 연도 운용.

합성 시장(2011~2013)은 4.7절 폴드를 만들 수 없어, 시험 연도를 2012·2013으로 줄인 설정과 손으로 만든 폴드 결과
폴더(학습하지 않은 PPO 정책)를 쓴다. 학습 자체는 test_rl_fold_report.py와 test_rl_train.py가 본다.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from stable_baselines3 import PPO

from src.portfolio import rl_wf_report as wf
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_env import PortfolioEnv
from src.portfolio.rl_fold_report import M9_NONE_NOTE, checkpoint_fingerprints
from tests.portfolio.conftest import VOL_TARGET_FOR_TESTS
from tests.portfolio.fakes import FakeEngine, market_tables

YEARS = (2012, 2013)
SEEDS = (0, 1)
RUN = "20261112_B_wf"
FOLD_RUNS = "20261111_B_wf_y{Y}"


@pytest.fixture(scope="module")
def synthetic(market):
    tables, kept = market_tables(market)
    tables = {
        **tables,
        "delisted_reused": pd.DataFrame(columns=["ticker", "delist_date", "last_price"]).astype({"last_price": "float64"}),
        "universe": kept.universe, "sector": kept.sector,
    }
    return tables, kept


@pytest.fixture(scope="module")
def ppo_env(market, rl_params, risk_params, cfg):
    """정책을 만들 때만 쓰는 환경 (관측·행동 공간)."""
    dates = market.signal_dates[:60]
    snaps = build_snapshots(dates, market.signals, market.features, market.universe, market.sector, market.bm_close,
                            lambda d: (), risk_params)
    return PortfolioEnv(snaps, FakeEngine(market.closes, market.bm_close, cfg["costs"]), rl_params, risk_params, 0.0, "full")


def _config(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["risk_layer"]["vol_target_annual"] = VOL_TARGET_FOR_TESTS
    c["splits"]["walk_forward_test_years"] = list(YEARS)
    c["portfolio"]["rl"]["seeds"] = list(SEEDS)
    path = tmp_path / "prereg.yaml"
    path.write_text(yaml.safe_dump(c, allow_unicode=True), encoding="utf-8")
    return path


def _fold_run(root, year, config, env, commit="abc", m9=None):
    """rl_fold_report(B-07)가 남기는 폴드 폴더 모양. 정책은 (연도, 시드)마다 다른 초기 가중치다."""
    folder = root / "reports" / FOLD_RUNS.replace("{Y}", str(year))
    for seed in SEEDS:
        (folder / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
        PPO("MlpPolicy", env, seed=year * 10 + seed, device="cpu").save(folder / f"seed_{seed}" / "best.zip")
    (folder / "config.yaml").write_bytes(config.read_bytes())
    (folder / "git_commit.txt").write_text(f"{commit}\n", encoding="utf-8")
    metrics = {
        "task": "B-07",
        "criteria": {"lambda_selection_ran_to_the_end": True},
        "signals": {"run_id": f"20261110_A_rlsig_y{year}"},
        "label_end_dates": {"path": "end_dates.parquet", "sha256": "ab", "signal_days": 777},
        "lambda_search": {"chosen_lambda": 0.5},
        "main_training": {"seeds": list(SEEDS), "per_seed": [{"seed": s, "best_val_ir": 0.1 * s} for s in SEEDS]},
        "fold": {"test_year": year, "purge": {"h": 20, "delay_purged": {"train": 1, "val": 0}}},
        "m9": m9 or {"applied": False, "note": M9_NONE_NOTE},
        "checkpoints": {"files": checkpoint_fingerprints(folder)},
    }
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return folder


def _test_signals(kept, test_dates, run_id="20261110_A_wfpm", drop=None):
    sig = kept.signals.assign(date=pd.to_datetime(kept.signals["date"]))
    day = sig["date"].dt.strftime("%Y-%m-%d")
    keep = set(test_dates) - ({drop} if drop else set())
    sig = sig[day.isin(keep)].copy()
    sig["model_version"] = "20261110_A_wfpm_y" + sig["date"].dt.year.astype(str)
    sig = sig.astype({"rank": "int32", "crash_flag": "int8"}).reset_index(drop=True)
    wf.io.write_table("signals", sig, run_id=run_id)


@pytest.fixture
def setup(synthetic, cfg, tmp_path, monkeypatch, ppo_env):
    tables, kept = synthetic
    config = _config(cfg, tmp_path)
    real_read = wf.io.read_table

    def read_table(name, date=None, run_id=None, stub=False):
        if date is None and run_id is None and not stub:
            return tables[name]
        return real_read(name, date=date, run_id=run_id, stub=stub)

    monkeypatch.setattr(wf, "ROOT", tmp_path)
    monkeypatch.setattr(wf.io, "ROOT", tmp_path)
    monkeypatch.setattr(wf, "TABLES", tuple(tables))
    monkeypatch.setattr(wf.io, "read_table", read_table)
    monkeypatch.setattr(wf, "feature_table", lambda prices, bm, members, risk: kept.features)
    monkeypatch.setattr(wf, "input_fingerprints", lambda t, root: {n: {"rows": len(df)} for n, df in t.items()})
    monkeypatch.setattr(wf, "_git_commit", lambda: "abc\n")
    monkeypatch.setattr(wf, "LAST_SIGNAL_DATE", kept.signal_dates[-2])  # 합성 시장의 마지막 주는 주 중간에서 끝난다
    test_dates = [d for d in kept.signal_dates[:-1] if int(d[:4]) in YEARS]
    folders = {y: _fold_run(tmp_path, y, config, ppo_env) for y in YEARS}
    _test_signals(kept, test_dates)
    args = ["--run-id", RUN, "--fold-runs", FOLD_RUNS, "--test-signals-run-id", "20261110_A_wfpm", "--config", str(config)]
    return args, test_dates, folders, tmp_path, config


def test_main_runs_every_seed_over_the_test_years(setup, monkeypatch):
    args, test_dates, folders, root, _ = setup
    seen = {}
    real_load = wf.load_policies
    monkeypatch.setattr(wf, "load_policies", lambda runs, seeds: seen.setdefault("p", real_load(runs, seeds)))
    real_rollout = wf.rollout
    calls = []

    def spy(env, policy_for):
        def chosen(d):
            calls.append((d, policy_for(d)))
            return calls[-1][1]
        return real_rollout(env, chosen)

    monkeypatch.setattr(wf, "rollout", spy)
    out = wf.main(args)

    policies, _ = seen["p"]
    # 신호일 t의 정책은 t가 속한 해의 폴드 정책이다 (연도 경계에서 교체)
    assert len(calls) == len(SEEDS) * len(test_dates)
    for k, (d, model) in enumerate(calls):
        assert model is policies[(int(d[:4]), SEEDS[k // len(test_dates)])]

    for seed in SEEDS:
        pos = wf.io.read_table("positions", run_id=f"{RUN}_s{seed}")
        assert sorted(pos["date"].dt.strftime("%Y-%m-%d").unique()) == test_dates
        assert set(pos["constructor"]) == {"rl"}
        assert np.allclose(pos.groupby("date")["target_weight"].sum(), 1.0)
    weekly = pd.read_csv(out / "weekly_excess.csv", dtype={"signal_date": str, "week": str})
    assert len(weekly) == len(SEEDS) * (len(test_dates) - 1)  # 마지막 시험 신호일은 체결하지 않는다
    assert (weekly["fold"] == weekly["signal_date"].str[:4].astype(int)).all()
    assert (weekly["week"] > weekly["signal_date"]).all()
    scores = pd.read_csv(out / "fold_seed_ir.csv")
    assert list(scores.columns) == ["fold", "seed", "ir"] and len(scores) == len(YEARS) * len(SEEDS)
    assert set(scores["fold"]) == set(YEARS) and set(scores["seed"]) == set(SEEDS)

    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    c = m["criteria"]
    assert c["every_fold_and_seed_trained"] and c["fold_runs_same_commit"] and c["label_end_dates_applied"]
    assert c["every_test_date_in_positions"]
    assert not c["m9_applied"] and not c["b07_complete"]  # M9 없이 학습한 폴드라 '설계 그대로'가 아니다
    assert m["fold_runs"]["signals_run_id"] == "20261110_A_rlsig" and len(m["checkpoints"]) == 4
    assert all(ck["file_matches_record"] for ck in m["checkpoints"])
    assert m["test_signals"]["model_version"]["2013"] == {"20261110_A_wfpm_y2013": m["test_signals"]["model_version"]["2013"]["20261110_A_wfpm_y2013"]}
    assert set(m["ir"]["per_seed"]) == {"0", "1"} and m["test"]["signal_dates"] == len(test_dates)
    assert sorted(p.name for p in out.iterdir()) == ["config.yaml", "fold_seed_ir.csv", "git_commit.txt",
                                                     "metrics.json", "weekly_excess.csv"]

    with pytest.raises(SystemExit, match="R-11"):
        wf.main(args)


def test_main_refuses_an_existing_positions_file(setup):
    args, *_ , root, _ = setup
    (root / "positions").mkdir(exist_ok=True)
    (root / "positions" / f"wf_{RUN}_s1.parquet").write_bytes(b"")
    with pytest.raises(SystemExit, match="R-11"):
        wf.main(args)


def test_other_weights_stop_the_run(setup, ppo_env):
    """드라이브에서 받은 체크포인트가 폴드 기록과 다른 가중치면 멈춘다."""
    args, _, folders, root, _ = setup
    PPO("MlpPolicy", ppo_env, seed=999, device="cpu").save(folders[2013] / "seed_1" / "best.zip")
    with pytest.raises(ValueError, match="정책 파라미터 지문"):
        wf.main(args)
    assert not (root / "reports" / RUN).exists()


def test_missing_checkpoint_or_fold_stops_the_run(setup):
    args, _, folders, root, _ = setup
    (folders[2012] / "seed_0" / "best.zip").unlink()
    with pytest.raises(SystemExit, match="팀 드라이브"):
        wf.main(args)
    with pytest.raises(SystemExit, match="결과가 없다"):
        wf.main([*args[:3], "20261111_B_none_y{Y}", *args[4:]])
    with pytest.raises(SystemExit, match=r"\{Y\}"):
        wf.main([*args[:3], "20261111_B_wf_y2012", *args[4:]])


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda m: m.update(task="B-06"), "B-07"),
        (lambda m: m["criteria"].update(lambda_selection_ran_to_the_end=False), "완료 기준"),
        (lambda m: m["main_training"].update(seeds=[0]), "사전등록 시드"),
        (lambda m: m["signals"].update(run_id="20261110_A_other_y2013"), "학습 신호 run_id"),
        (lambda m: m["label_end_dates"].update(sha256="cd"), "라벨 종료일"),
        (lambda m: m["fold"].update(test_year=2012), "시험 연도"),
    ],
)
def test_folds_from_another_design_stop_the_run(setup, change, match):
    args, _, folders, root, _ = setup
    path = folders[2013] / "metrics.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    change(m)
    path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        wf.main(args)


def test_other_config_or_test_signals_stop_the_run(setup, synthetic):
    args, test_dates, folders, root, config = setup
    (folders[2012] / "config.yaml").write_bytes(config.read_bytes() + b"\n# other\n")
    with pytest.raises(ValueError, match="config.yaml"):
        wf.main(args)
    (folders[2012] / "config.yaml").write_bytes(config.read_bytes())
    _test_signals(synthetic[1], test_dates, run_id="20261110_A_short", drop=test_dates[10])
    with pytest.raises(ValueError, match="시험 신호일과 다르다"):
        wf.main([*args[:5], "20261110_A_short", *args[6:]])


def test_mixed_commits_are_recorded_not_hidden(setup):
    args, _, folders, root, _ = setup
    (folders[2013] / "git_commit.txt").write_text("def\n", encoding="utf-8")
    out = wf.main(args)
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["fold_runs"]["commits"] == ["abc", "def"] and not m["criteria"]["fold_runs_same_commit"]


def test_weekly_fold_is_the_year_of_the_signal_date():
    """2016-12-30 신호의 체결 주는 2017-01-02에 시작하지만 그 주의 비중은 폴드 2016 정책이 정했다."""
    days = pd.DatetimeIndex(["2016-12-29", "2016-12-30", "2017-01-02", "2017-01-03"])
    weekly = wf.weekly_table(3, ["2016-12-29", "2016-12-30"], np.array([0.01, -0.02]), days)
    assert weekly.to_dict("list") == {"seed": [3, 3], "fold": [2016, 2016], "signal_date": ["2016-12-29", "2016-12-30"],
                                      "week": ["2016-12-30", "2017-01-02"], "a_k": [0.01, -0.02]}
    with pytest.raises(ValueError, match="2017"):
        wf.fold_seed_ir(weekly, [2016, 2017])
