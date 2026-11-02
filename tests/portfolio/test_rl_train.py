"""PPO 학습 루프 (6.5절 λ 선택, 본 학습, 체크포인트, 테스트 행동).

작업 B-01 완료 기준 '임의 입력으로 PPO 학습 루프가 돈다'를 확인한다. 스텝 수는 테스트 전용 축소값이다.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO

from src.portfolio.params import LambdaSearchParams
from src.portfolio.risk_layer import CASH
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_env import PortfolioEnv
from src.portfolio.rl_train import rollout, train_fold, train_one
from tests.portfolio.fakes import FakeEngine, ScriptedEngine, fake_ir, make_snapshot


@pytest.fixture(scope="module")
def segments(market, risk_params, cfg):
    snaps = build_snapshots(
        market.signal_dates, market.signals, market.features, market.universe, market.sector, market.bm_close,
        lambda d: market.cooldown[d], risk_params,
    )
    train = [s for s in snaps if s.date < "2013-01-01"]
    val = [s for s in snaps if s.date >= "2013-01-01"]
    return train, val, FakeEngine(market.closes, market.bm_close, cfg["costs"])


def _small(rl):
    return dataclasses.replace(
        rl,
        total_timesteps_per_seed=2048,
        eval_every_steps=1024,
        seeds=(0, 1),
        lambda_search=LambdaSearchParams(grid=(0.0, 1.0), seed=0, total_timesteps=2048),
    )


def test_ppo_loop_runs_on_synthetic_input(segments, rl_params, risk_params, tmp_path):
    train, val, engine = segments
    rl = _small(rl_params)
    val_env = PortfolioEnv(val, engine, rl, risk_params, 0.0, "full")
    fold = train_fold(lambda lam: PortfolioEnv(train, engine, rl, risk_params, lam), val_env, rl, fake_ir, tmp_path)

    assert list(fold.lambda_table["lambda"]) == [0.0, 1.0] and fold.lam in (0.0, 1.0)
    assert (tmp_path / "lambda_search.csv").exists()
    assert fold.lam == fold.lambda_table.loc[fold.lambda_table["best_val_ir"].idxmax(), "lambda"]
    for res in fold.seeds:
        assert res.best_path.exists()
        assert list(res.eval_log["timesteps"]) == [1024, 2048]
        assert res.best_val_ir == res.eval_log["val_ir"].max()

    best = PPO.load(fold.seeds[0].best_path, device="cpu")
    out = rollout(val_env, lambda _: best)
    assert fake_ir(out.excess_returns) == pytest.approx(fold.seeds[0].best_val_ir, rel=1e-12)
    pos = out.positions
    assert list(pos.columns) == ["date", "ticker", "target_weight", "constructor", "exposure", "regime_mc"]
    assert pos["date"].nunique() == len(val) and len(out.excess_returns) == len(val) - 1
    assert np.allclose(pos.groupby("date")["target_weight"].sum(), 1.0, atol=1e-9)
    stocks = pos[pos["ticker"] != CASH]
    assert (stocks["target_weight"] <= risk_params.stock_cap + 1e-9).all()
    assert stocks.groupby("date").size().max() <= rl.max_holdings
    assert (pos["constructor"] == "rl").all()


def test_same_seed_gives_same_training(segments, rl_params, risk_params, tmp_path):
    train, val, engine = segments
    rl = _small(rl_params)
    val_env = PortfolioEnv(val, engine, rl, risk_params, 0.0, "full")
    runs = [
        train_one(lambda lam: PortfolioEnv(train, engine, rl, risk_params, lam), val_env, 1.0, 7, 2048, 1024,
                  fake_ir, tmp_path / name)
        for name in ("a", "b")
    ]
    pd.testing.assert_frame_equal(runs[0].eval_log, runs[1].eval_log)


class _Recorder:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, obs, deterministic):
        assert deterministic
        self.calls += 1
        return np.zeros(61, dtype=np.float32), None


def test_rollout_switches_model_by_date_and_keeps_holdings(rl_params, risk_params):
    dates = ["2015-12-18", "2015-12-24", "2016-01-08", "2016-01-15"]
    engine = ScriptedEngine()
    env = PortfolioEnv([make_snapshot(date=d) for d in dates], engine, rl_params, risk_params, 0.0, "full")
    y2015, y2016 = _Recorder(), _Recorder()
    out = rollout(env, lambda d: y2015 if d < "2016-01-01" else y2016)
    assert (y2015.calls, y2016.calls) == (2, 2)
    assert len(engine.calls) == 3  # 연도 경계에서도 같은 상태로 이어서 체결한다
    assert sorted(out.positions["date"].astype(str).unique()) == dates
