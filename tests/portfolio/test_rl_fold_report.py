"""B-06 (src/portfolio/rl_fold_report.py): 폴드 하나의 λ 탐색과 PPO 학습 (6.5절).

PPO를 실제로 학습하는 테스트는 slow로 표시했다. 나머지는 학습 함수를 가짜로 바꿔 연결과 완료 기준 판정을 본다.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from stable_baselines3.common.save_util import save_to_zip_file

from src.models.splits import Fold, make_folds
from src.portfolio import rl_fold_report as report
from src.portfolio.params import LambdaSearchParams, load_prereg
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_train import TrainResult
from tests.portfolio.conftest import VOL_TARGET_FOR_TESTS
from tests.portfolio.fakes import FakeEngine, fake_ir, market_tables


def _calendar():
    """2011~2025 금요일 신호일과 영업일 달력 (폴드 분할만 볼 때 쓴다)."""
    days = pd.bdate_range("2010-12-01", "2026-03-31")
    fridays = days[days.day_of_week == 4]
    sig = fridays[(fridays >= "2011-01-01") & (fridays <= "2025-12-31")]
    return [d.strftime("%Y-%m-%d") for d in sig], days


def test_fold_split_is_the_a_split(cfg):
    """A-06·A-07과 같은 학습·검증 신호일 (A의 make_folds 그대로, h = 20 purge)."""
    dates, days = _calendar()
    fold = report.fold_split(dates, days, 2016, cfg)
    (want,) = make_folds(pd.DatetimeIndex(dates), days, h=20, test_years=(2016,))
    for part in ("train", "val", "test"):
        pd.testing.assert_index_equal(getattr(fold, part), getattr(want, part))
    assert (fold.train.year.min(), fold.train.year.max()) == (2011, 2013)
    assert set(fold.val.year) == {2014, 2015} and set(fold.test.year) == {2016}
    # 남은 마지막 학습 신호일의 라벨(t+1+20 거래일)은 검증 첫 신호일 전에 끝나고, 빠진 신호일은 그렇지 않다
    dropped = [pd.Timestamp(d) for d in dates if d.startswith("2013") and pd.Timestamp(d) > fold.train[-1]]
    assert dropped and all(days[days.get_loc(d) + 21] >= fold.val[0] for d in dropped)
    assert days[days.get_loc(fold.train[-1]) + 21] < fold.val[0]
    s = report.fold_summary(fold, dates, "명목")
    assert s["train"]["purged"] == len(dropped) and s["val"]["purged"] > 0
    assert (s["test_year"], s["purge"]["h"]) == (2016, 20)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("splits", "validation_years"), 3, "검증 2년"),
        (("labels", "main_horizon_days"), 5, "h = 20"),
    ],
)
def test_fold_split_rejects_settings_the_code_does_not_implement(cfg, path, value, match):
    bad = copy.deepcopy(cfg)
    bad[path[0]][path[1]] = value
    dates, days = _calendar()
    with pytest.raises(ValueError, match=match):
        report.fold_split(dates, days, 2016, bad)


def test_fold_split_rejects_a_year_that_is_not_a_fold(cfg):
    dates, days = _calendar()
    with pytest.raises(ValueError, match="walk_forward_test_years"):
        report.fold_split(dates, days, 2015, cfg)


@pytest.fixture(scope="module")
def fold_market(market, risk_params, cfg):
    """합성 시장(3년)은 4.7절 폴드를 만들 수 없어 학습·검증을 직접 정한다. 사이의 4개 신호일은 purge로 빠진 것처럼 비운다."""
    dates = market.signal_dates
    snaps = build_snapshots(dates, market.signals, market.features, market.universe, market.sector, market.bm_close,
                            lambda d: market.cooldown[d], risk_params)
    fold = Fold(2014, pd.DatetimeIndex(dates[:96]), pd.DatetimeIndex(dates[100:150]), pd.DatetimeIndex(dates[150:]))
    return fold, snaps, FakeEngine(market.closes, market.bm_close, cfg["costs"]), dates


def test_fold_envs_use_the_fold_dates_and_the_calendar(fold_market, rl_params, risk_params):
    fold, snaps, engine, dates = fold_market
    envs = report.fold_envs(fold, snaps, engine, rl_params, risk_params, dates)
    assert (envs.train_dates, envs.val_dates) == (dates[:96], dates[100:150])
    env = envs.make_env(0.5)
    assert (env.lam, env.episode) == (0.5, "random") and [s.date for s in env.snapshots] == dates[:96]
    assert envs.val_env.episode == "full" and [s.date for s in envs.val_env.snapshots] == dates[100:150]

    # 학습 구간 안에 purge 자리가 있으면 그 자리를 건너뛰는 스텝이 없다 (DEC-21 (g), 달력을 넘겼기 때문)
    gap = Fold(2014, pd.DatetimeIndex(dates[:40] + dates[44:110]), fold.val, fold.test)
    env = report.fold_envs(gap, snaps, engine, rl_params, risk_params, dates).make_env(0.0)
    assert not env._linked[39] and env._linked[38] and env._linked[40]
    assert all(not (i <= 39 < i + rl_params.episode_steps) for i in env._starts)


def test_fold_envs_stop_without_snapshots_or_a_52_step_run(fold_market, rl_params, risk_params):
    fold, snaps, engine, dates = fold_market
    with pytest.raises(ValueError, match="스냅샷이 없는"):
        report.fold_envs(fold, snaps[:120], engine, rl_params, risk_params, dates)
    short = Fold(2014, pd.DatetimeIndex(dates[:50]), fold.val, fold.test)  # 52스텝 연속 구간이 없다
    with pytest.raises(ValueError):
        report.fold_envs(short, snaps, engine, rl_params, risk_params, dates)


def test_lambda_choice_takes_the_smaller_lambda_on_ties():
    table = pd.DataFrame({"lambda": [0.0, 0.5, 1.0, 2.0], "best_val_ir": [0.1, 0.3, 0.3, 0.2]})
    assert report.lambda_choice(table) == 0.5
    assert report.lambda_choice(table.assign(best_val_ir=[0.4, 0.3, 0.3, 0.2])) == 0.0


def _small(rl, grid=(0.0, 1.0)):
    return dataclasses.replace(
        rl, total_timesteps_per_seed=4, eval_every_steps=2, seeds=(0, 1),
        lambda_search=LambdaSearchParams(grid=grid, seed=0, total_timesteps=6),
    )


def _write_checkpoint(path, weight, note="test"):
    """SB3 저장 형식의 작은 체크포인트. note가 다르면 가중치가 같아도 파일 내용이 다르다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_to_zip_file(path, data={"note": note}, params={"policy": {"w": torch.full((2, 3), float(weight))}})


def _write_run(folder, total, every, irs):
    folder.mkdir(parents=True, exist_ok=True)
    log = pd.DataFrame({"timesteps": [every * k for k in range(1, total // every + 1)], "val_ir": irs})
    log.to_csv(folder / "eval_log.csv", index=False)
    _write_checkpoint(folder / "best.zip", max(irs))
    return log


def _fake_training(monkeypatch, scores, chosen=None, drop_eval=False):
    """select_lambda·train_one을 대신한다. λ마다 scores[λ] 검증 IR 기록과 빈 체크포인트를 쓴다."""
    def select(make_env, val_env, rl, ir_fn, out):
        s = rl.lambda_search
        rows = []
        for lam in s.grid:
            make_env(lam)  # 환경이 실제로 만들어지는지
            irs = [scores[lam] - 0.1 * k for k in range(s.total_timesteps // rl.eval_every_steps)][::-1]
            log = _write_run(out / f"lambda_{lam:g}", s.total_timesteps, rl.eval_every_steps, irs)
            if drop_eval and lam == s.grid[-1]:
                log.iloc[:-1].to_csv(out / f"lambda_{lam:g}" / "eval_log.csv", index=False)
            rows.append({"lambda": lam, "best_val_ir": max(irs), "best_timesteps": int(log["timesteps"].iloc[-1])})
        table = pd.DataFrame(rows)
        return (chosen if chosen is not None else report.lambda_choice(table)), table

    def one(make_env, val_env, lam, seed, total, every, ir_fn, out):
        log = _write_run(out, total, every, [0.1 * k for k in range(total // every)])
        return TrainResult(lam, seed, float(log["val_ir"].max()), int(log["timesteps"].iloc[-1]), out / "best.zip", log)

    monkeypatch.setattr(report, "select_lambda", select)
    monkeypatch.setattr(report, "train_one", one)


def test_run_fold_records_the_lambda_procedure(fold_market, rl_params, risk_params, tmp_path, monkeypatch):
    fold, snaps, engine, dates = fold_market
    rl = _small(rl_params, grid=(0.0, 0.5, 1.0, 2.0))
    envs = report.fold_envs(fold, snaps, engine, rl, risk_params, dates)
    _fake_training(monkeypatch, {0.0: 0.1, 0.5: 0.3, 1.0: 0.3, 2.0: 0.2})
    r = report.run_fold(envs, rl, fake_ir, tmp_path, [0])
    assert r["criteria"] == {"lambda_selection_ran_to_the_end": True, "every_lambda_trained": True,
                             "eval_logs_and_checkpoints_complete": True, "chosen_lambda_follows_dec21_f": True}
    assert r["lambda_search"]["chosen_lambda"] == 0.5 and r["main_training"]["lambda"] == 0.5
    assert [p["n_evals"] for p in r["lambda_search"]["per_lambda"]] == [3, 3, 3, 3]
    assert r["main_training"]["seeds"] == [0] and not r["main_training"]["all_prereg_seeds"]
    assert (tmp_path / "lambda_search.csv").exists() and (tmp_path / "seed_0" / "best.zip").exists()


@pytest.mark.parametrize(("chosen", "drop_eval", "failed"),
                         [(1.0, False, "chosen_lambda_follows_dec21_f"), (None, True, "eval_logs_and_checkpoints_complete")])
def test_run_fold_fails_the_criterion(fold_market, rl_params, risk_params, tmp_path, monkeypatch, chosen, drop_eval, failed):
    """고른 λ가 규칙과 다르거나(동률인데 큰 λ) 평가 기록이 빠지면 완료 기준을 통과하지 않는다."""
    fold, snaps, engine, dates = fold_market
    rl = _small(rl_params, grid=(0.0, 0.5, 1.0, 2.0))
    envs = report.fold_envs(fold, snaps, engine, rl, risk_params, dates)
    _fake_training(monkeypatch, {0.0: 0.1, 0.5: 0.3, 1.0: 0.3, 2.0: 0.2}, chosen=chosen, drop_eval=drop_eval)
    r = report.run_fold(envs, rl, fake_ir, tmp_path, [0])
    assert not r["criteria"][failed] and not r["criteria"]["lambda_selection_ran_to_the_end"]


def test_checkpoint_fingerprints_compare_weights_not_files(tmp_path):
    """파일 SHA-256은 드라이브 파일 대조용이고, 같은 가중치인지는 정책 파라미터 지문으로 본다 (PM 11/10)."""
    _write_checkpoint(tmp_path / "lambda_0" / "best.zip", 1.0, note="a")
    _write_checkpoint(tmp_path / "seed_0" / "best.zip", 1.0, note="b")  # 같은 가중치, 다른 파일 (저장 시각이 다른 것과 같다)
    _write_checkpoint(tmp_path / "lambda_0.5" / "best.zip", 2.0, note="a")
    (tmp_path / "lambda_1").mkdir()  # 체크포인트가 없는 폴더는 목록에 없다 (완료 기준이 따로 잡는다)
    fp = report.checkpoint_fingerprints(tmp_path)
    assert list(fp) == ["lambda_0.5/best.zip", "lambda_0/best.zip", "seed_0/best.zip"]
    same, other, base = fp["seed_0/best.zip"], fp["lambda_0.5/best.zip"], fp["lambda_0/best.zip"]
    assert base["sha256"] == hashlib.sha256((tmp_path / "lambda_0" / "best.zip").read_bytes()).hexdigest()
    assert base["bytes"] == (tmp_path / "lambda_0" / "best.zip").stat().st_size
    assert same["sha256"] != base["sha256"] and same["policy_sha256"] == base["policy_sha256"]
    assert other["policy_sha256"] != base["policy_sha256"]


@pytest.mark.slow
def test_run_fold_trains_ppo_end_to_end(fold_market, rl_params, risk_params, tmp_path):
    """실제 PPO로 λ 2개 탐색과 시드 하나 본 학습 (스텝 수는 테스트용 축소값)."""
    fold, snaps, engine, dates = fold_market
    rl = dataclasses.replace(rl_params, total_timesteps_per_seed=2048, eval_every_steps=1024,
                             lambda_search=LambdaSearchParams(grid=(0.0, 1.0), seed=0, total_timesteps=2048))
    envs = report.fold_envs(fold, snaps, engine, rl, risk_params, dates)
    r = report.run_fold(envs, rl, fake_ir, tmp_path, [0])
    assert r["criteria"]["lambda_selection_ran_to_the_end"]
    for name in ("lambda_0", "lambda_1", "seed_0"):
        assert (tmp_path / name / "best.zip").stat().st_size > 0
    assert pd.read_csv(tmp_path / "lambda_search.csv")["lambda"].tolist() == [0.0, 1.0]
    # 본 학습(시드 0, 같은 스텝 수)은 고른 λ의 탐색과 같은 학습이라 체크포인트 가중치가 같다. 파일은 저장 시각이 달라 다르다
    fp = report.checkpoint_fingerprints(tmp_path)
    searched = fp[f"lambda_{r['lambda_search']['chosen_lambda']:g}/best.zip"]
    assert searched["policy_sha256"] == fp["seed_0/best.zip"]["policy_sha256"]


@pytest.fixture(scope="module")
def synthetic(market):
    tables, kept = market_tables(market)
    tables = {
        **tables,
        "delisted_reused": pd.DataFrame(columns=["ticker", "delist_date", "last_price"]).astype({"last_price": "float64"}),
        "universe": kept.universe, "sector": kept.sector,
    }
    return tables, kept


def _main_setup(synthetic, cfg, tmp_path, monkeypatch, run_id="20261110_B_test"):
    """main을 합성 시장으로 돌리는 준비. 공용 표는 메모리에서 읽고, run_id·date 파일은 tmp_path의 io로 읽는다.
    합성 시장(3년)은 4.7절 폴드를 만들 수 없어 fold_split을 고정 폴드로 바꾼다. 실제 라벨 종료일을 넘기면
    (지연 purge를 흉내 내어) 학습 마지막 신호일 하나를 더 뺀다."""
    tables, kept = synthetic
    c = copy.deepcopy(cfg)
    c["risk_layer"]["vol_target_annual"] = VOL_TARGET_FOR_TESTS
    config = tmp_path / "prereg.yaml"
    config.write_text(yaml.safe_dump(c, allow_unicode=True), encoding="utf-8")
    dates = kept.signal_dates
    by_date = {pd.Timestamp(d): g for d, g in kept.signals.groupby(pd.to_datetime(kept.signals["date"]))}
    real_read = report.io.read_table

    def read_table(name, date=None, run_id=None, stub=False):
        if date is None and run_id is None and not stub:
            return tables[name]
        return real_read(name, date=date, run_id=run_id, stub=stub)

    def commit():
        assert not (tmp_path / "reports" / run_id).exists()  # 결과 폴더가 새 파일 목록에 들지 않게
        return "abc\n"

    def split(dates_, days, year, cfg_, actual_end_dates=None):
        train = dates[:95] if actual_end_dates else dates[:96]
        return Fold(year, pd.DatetimeIndex(train), pd.DatetimeIndex(dates[100:150]), pd.DatetimeIndex(dates[150:]))

    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report.io, "ROOT", tmp_path)
    monkeypatch.setattr(report, "TABLES", tuple(tables))
    monkeypatch.setattr(report.io, "read_table", read_table)
    monkeypatch.setattr(report, "_a_signal_function", lambda: lambda close, tickers, t: by_date[t])
    monkeypatch.setattr(report, "feature_table", lambda prices, bm, members, risk: kept.features)
    monkeypatch.setattr(report, "input_fingerprints", lambda t, root: {n: {"rows": len(df)} for n, df in t.items()})
    monkeypatch.setattr(report, "_git_commit", commit)
    monkeypatch.setattr(report, "LAST_SIGNAL_DATE", dates[-2])  # 합성 시장의 마지막 주는 주 중간에서 끝난다
    monkeypatch.setattr(report, "fold_split", split)
    _fake_training(monkeypatch, {0.0: 0.1, 0.5: 0.3, 1.0: 0.3, 2.0: 0.2})
    return config, dates, kept


def test_main_writes_the_run_folder_once(synthetic, cfg, tmp_path, monkeypatch):
    config, dates, _ = _main_setup(synthetic, cfg, tmp_path, monkeypatch)
    args = ["--run-id", "20261110_B_test", "--fold", "2016", "--main-seeds", "0", "--config", str(config)]
    out = report.main(args)
    assert sorted(p.name for p in out.iterdir()) == [
        "config.yaml", "git_commit.txt", "lambda_0", "lambda_0.5", "lambda_1", "lambda_2", "lambda_search.csv",
        "metrics.json", "seed_0"]
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["task"] == "B-06" and m["criteria"]["lambda_selection_ran_to_the_end"]
    assert m["lambda_search"]["chosen_lambda"] == 0.5 and m["main_training"]["seeds"] == [0]
    assert m["main_training"]["all_prereg_seeds"] is False and "baseline" in m["signals"]["source"]
    assert m["fold"]["train"] == {"n": 96, "first": dates[0], "last": dates[95], "purged": m["fold"]["train"]["purged"]}
    assert m["fold"]["purge"]["h"] == 20 and "명목" in m["fold"]["purge"]["end_dates"]
    assert "delay_purged" not in m["fold"]["purge"] and m["label_end_dates"] is None
    assert m["m9"]["applied"] is False and "M9" in m["m9"]["note"] and m["device"] == "cpu"
    assert set(m["versions"]) >= {"stable_baselines3", "torch", "gymnasium"}
    assert m["universe_size"]["min"] > 0
    files = m["checkpoints"]["files"]
    assert sorted(files) == sorted(f"{p.parent.name}/best.zip" for p in out.glob("*/best.zip"))
    assert len(files) == 5 and "드라이브" in m["checkpoints"]["storage"]
    for rel, f in files.items():
        assert f["sha256"] == hashlib.sha256((out / rel).read_bytes()).hexdigest() and len(f["policy_sha256"]) == 64

    with pytest.raises(SystemExit, match="R-11"):
        report.main(args)
    with pytest.raises(SystemExit, match="형식"):
        report.main(["--run-id", "20261110_A_test", "--fold", "2016", "--config", str(config)])
    with pytest.raises(SystemExit, match="사전등록 시드"):
        report.main(["--run-id", "20261110_B_test2", "--fold", "2016", "--main-seeds", "0,7", "--config", str(config)])


def _write_ml_signals(kept, train, val, run_id):
    """합성 신호를 A의 폴드별 ML 신호 파일 형식(학습 = 교차 적합, 검증 = 폴드 모델)으로 쓴다."""
    sig = kept.signals.assign(date=pd.to_datetime(kept.signals["date"]))
    day = sig["date"].dt.strftime("%Y-%m-%d")
    sig = sig[day.isin(set(train) | set(val))].copy()
    sig["model_version"] = np.where(day[sig.index].isin(set(val)), "x_fold", "x_cross")
    sig = sig.astype({"rank": "int32", "crash_flag": "int8"}).reset_index(drop=True)
    return report.io.write_table("signals", sig, run_id=run_id), sig


def _write_end_dates(path, dates):
    d = pd.DatetimeIndex(pd.to_datetime(dates[:150]))
    df = pd.DataFrame({"date": d, "ticker": "000001", "h": 20, "nominal_end": d + pd.Timedelta(days=28)})
    df["actual_end"] = df["nominal_end"]
    df.loc[94, "actual_end"] += pd.Timedelta(days=60)  # 거래정지로 늦어진 라벨
    df.to_parquet(path, index=False)


def _write_alerts(kept, day, run_id):
    """day 10:00 KST에 나온 경보 하나 (U_t 첫 종목)."""
    u = kept.universe
    ticker = sorted(u.loc[(pd.to_datetime(u["date"]) == pd.Timestamp(day)) & u["in_universe"].astype(bool), "ticker"])[0]
    at = pd.Series(pd.DatetimeIndex([pd.Timestamp(day) + pd.Timedelta(hours=10)]).tz_localize("Asia/Seoul"))
    at = at.astype(pd.DatetimeTZDtype("ns", "Asia/Seoul"))
    alerts = pd.DataFrame({"date": [pd.Timestamp(day)], "ticker": [ticker], "trigger_type": ["T1"],
                           "rcept_no": ["20110101000001"], "report_nm": ["테스트 공시"], "rcept_dt": at, "available_at": at})
    if "generated_at" in report.io.SCHEMAS["alerts"]:  # D #50부터 필수. 백테스트 파일은 available_at과 같다 (PM #51)
        alerts["generated_at"] = at
    report.io.write_table("alerts", alerts, run_id=run_id)
    return ticker


def test_main_b07_reads_ml_signals_actual_end_dates_and_m9(synthetic, cfg, tmp_path, monkeypatch):
    """B-07 입력: A 폴드별 ML 신호 파일, C 실제 라벨 종료일 purge, PM 백테스트 경보 (쿨다운 마스킹 + 엔진 청산)."""
    config, dates, kept = _main_setup(synthetic, cfg, tmp_path, monkeypatch, run_id="20261111_B_wf_y2016")
    tables, _ = synthetic
    path, sig = _write_ml_signals(kept, dates[:95], dates[100:150], "20261110_A_rlsig_y2016")
    _write_end_dates(tmp_path / "end_dates.parquet", dates)
    ticker = _write_alerts(kept, dates[30], "20261110_PM_bt")
    days = pd.DatetimeIndex(sorted(pd.to_datetime(tables["index"]["date"]).unique()))
    monkeypatch.setattr("src.events.alerts.load_trading_days", lambda: days)
    seen = {}
    real_snaps, real_engine = report.build_snapshots, report.build_engine
    monkeypatch.setattr(report, "build_snapshots", lambda *a, **k: seen.setdefault("snaps", real_snaps(*a, **k)))
    monkeypatch.setattr(report, "build_engine", lambda inputs, c, **kw: seen.setdefault("kw", kw) and real_engine(inputs, c, **kw))

    out = report.main(["--run-id", "20261111_B_wf_y2016", "--fold", "2016", "--main-seeds", "0", "--config", str(config),
                       "--signals-run-id", "20261110_A_rlsig", "--label-end-dates", str(tmp_path / "end_dates.parquet"),
                       "--m9-alerts-run-id", "20261110_PM_bt"])
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["task"] == "B-07" and m["criteria"]["lambda_selection_ran_to_the_end"]
    s = m["signals"]
    assert (s["run_id"], s["path"], s["rows"]) == ("20261110_A_rlsig_y2016", "signals/wf_20261110_A_rlsig_y2016.parquet", len(sig))
    assert s["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert s["model_version"] == {"train": {"x_cross": int((sig["model_version"] == "x_cross").sum())},
                                  "val": {"x_fold": int((sig["model_version"] == "x_fold").sum())}}
    assert m["label_end_dates"]["sha256"] == hashlib.sha256((tmp_path / "end_dates.parquet").read_bytes()).hexdigest()
    assert m["fold"]["train"]["n"] == 95 and m["fold"]["purge"]["delay_purged"] == {"train": 1, "val": 0}
    assert "실제 라벨 종료일" in m["fold"]["purge"]["end_dates"]

    # M9: 경보 다음 거래일부터 cooldown_trading_days 동안 신규 진입 금지, 그 첫날 이벤트 청산
    n_cool = int(load_prereg()["events"]["cooldown_trading_days"])  # load_book은 configs/prereg.yaml을 읽는다
    start = days[days.get_loc(pd.Timestamp(dates[30])) + 1].strftime("%Y-%m-%d")
    assert m["m9"] == {"applied": True, "alerts_run_id": "20261110_PM_bt", "path": "alerts/wf_20261110_PM_bt.parquet",
                       "sha256": m["m9"]["sha256"], "alerts": 1, "cooldown_days": n_cool, "event_liquidations": 1}
    assert seen["kw"]["event_liquidations"] == {start: frozenset({ticker})} and len(seen["kw"]["cooldowns"]) == n_cool
    snap = {sn.date: sn for sn in seen["snaps"]}
    assert snap[dates[31]].cooldown[snap[dates[31]].index[ticker]]  # 다음 신호일 체결도 쿨다운 안
    assert not snap[dates[60]].cooldown.any()


def test_main_b07_stops_when_a_signals_use_other_dates(synthetic, cfg, tmp_path, monkeypatch):
    """A 파일이 명목 purge 신호일(학습 96개)인데 B는 실제 종료일 purge(95개)면 결과 폴더를 만들기 전에 멈춘다."""
    config, dates, kept = _main_setup(synthetic, cfg, tmp_path, monkeypatch, run_id="20261111_B_wf_y2016")
    _write_ml_signals(kept, dates[:96], dates[100:150], "20261110_A_bad_y2016")
    _write_end_dates(tmp_path / "end_dates.parquet", dates)
    with pytest.raises(ValueError, match="학습·검증 신호일과 다르다"):
        report.main(["--run-id", "20261111_B_wf_y2016", "--fold", "2016", "--config", str(config),
                     "--signals-run-id", "20261110_A_bad", "--label-end-dates", str(tmp_path / "end_dates.parquet")])
    assert not (tmp_path / "reports" / "20261111_B_wf_y2016").exists()
