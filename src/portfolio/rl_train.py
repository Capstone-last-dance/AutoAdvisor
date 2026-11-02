"""M4 RL 학습 절차 (6.5절: λ 선택, 본 학습, 체크포인트, 테스트 행동). B 소유.

PPO 하이퍼파라미터는 SB3 기본값을 쓴다 (부록 1 portfolio.rl.other_hyperparameters).
폴드별 학습·검증 신호일과 입력 신호의 출처는 호출자가 정한다 (docs/design_B.md [결정 필요] D1, D5).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.portfolio.interfaces import IRFunction
from src.portfolio.params import RLParams
from src.portfolio.risk_layer import positions_frame
from src.portfolio.rl_env import PortfolioEnv

DEVICE = "cpu"  # 재현성(R-10)을 위해 고정한다. PPO 하이퍼파라미터가 아니라 계산 장치다

EnvFactory = Callable[[float], PortfolioEnv]  # λ → 학습 환경 (episode="random")


@dataclass(frozen=True)
class Rollout:
    positions: pd.DataFrame  # 신호일별 positions 행 (9.3절)
    excess_returns: np.ndarray  # 체결된 스텝의 주간 초과수익 a_k = R_net − R_BM


def rollout(env: PortfolioEnv, policy_for: Callable[[str], PPO]) -> Rollout:
    """결정적 행동(가우시안 평균)으로 env의 전 구간을 운용한다 (6.5절 테스트 행동).

    policy_for(신호일)이 그날 쓸 모델을 고른다. walk-forward 테스트에서는 연도 경계에서 보유를
    유지한 채 다음 폴드의 모델로 바꾼다 (4.7절).
    """
    if env.episode != "full":
        raise ValueError("rollout은 episode='full' 환경에서만 쓴다")
    obs, _ = env.reset()
    frames, excess = [], []
    while True:
        action, _ = policy_for(env.current_date).predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        frames.append(
            positions_frame(info["date"], info["targets"], info["cash"], "rl", info["exposure"], info["regime_mc"])
        )
        if info["executed"]:
            excess.append(info["excess_return"])
        if terminated or truncated:
            break
    return Rollout(pd.concat(frames, ignore_index=True), np.asarray(excess, dtype=np.float64))


class ValidationIRCallback(BaseCallback):
    """eval_every 스텝마다 검증 구간을 결정적 행동으로 운용해 IR(M5)을 계산하고 최고 체크포인트를 저장한다.

    동률이면 먼저 나온 체크포인트를 쓴다. IR이 유한하지 않은 평가는 기록만 하고 고르지 않는다.
    """

    def __init__(self, val_env: PortfolioEnv, eval_every: int, ir_fn: IRFunction, best_path: Path) -> None:
        super().__init__()
        self.val_env = val_env
        self.eval_every = eval_every
        self.ir_fn = ir_fn
        self.best_path = best_path
        self.records: list[dict[str, float]] = []
        self.best_ir = -np.inf
        self.best_timesteps: int | None = None
        self._next_eval = eval_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            ir = float(self.ir_fn(rollout(self.val_env, lambda _: self.model).excess_returns))
            self.records.append({"timesteps": self.num_timesteps, "val_ir": ir})
            if np.isfinite(ir) and ir > self.best_ir:
                self.best_ir, self.best_timesteps = ir, self.num_timesteps
                self.model.save(self.best_path)
        return True


@dataclass(frozen=True)
class TrainResult:
    lam: float
    seed: int
    best_val_ir: float
    best_timesteps: int
    best_path: Path
    eval_log: pd.DataFrame


def train_one(
    make_env: EnvFactory,
    val_env: PortfolioEnv,
    lam: float,
    seed: int,
    total_timesteps: int,
    eval_every: int,
    ir_fn: IRFunction,
    out_dir: Path,
) -> TrainResult:
    """λ와 시드 하나로 PPO(MlpPolicy, SB3 기본값)를 학습한다. out_dir에 best.zip과 eval_log.csv를 남긴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = PPO("MlpPolicy", make_env(lam), seed=seed, device=DEVICE)
    callback = ValidationIRCallback(val_env, eval_every, ir_fn, out_dir / "best.zip")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    log = pd.DataFrame(callback.records, columns=["timesteps", "val_ir"])
    log.to_csv(out_dir / "eval_log.csv", index=False)
    if callback.best_timesteps is None:
        raise RuntimeError(f"λ={lam}, seed={seed}: 유한한 검증 IR이 없다 ({out_dir / 'eval_log.csv'})")
    return TrainResult(lam, seed, callback.best_ir, callback.best_timesteps, out_dir / "best.zip", log)


def select_lambda(
    make_env: EnvFactory, val_env: PortfolioEnv, rl: RLParams, ir_fn: IRFunction, out_dir: Path
) -> tuple[float, pd.DataFrame]:
    """6.5절 λ 선택. 격자의 λ마다 lambda_search.seed, lambda_search.total_timesteps로 학습해 검증 IR이
    가장 높은 λ를 고른다. λ의 점수는 체크포인트 중 최고 검증 IR이고, 동률이면 격자에서 앞선 λ다 (D3)."""
    search = rl.lambda_search
    results = [
        train_one(make_env, val_env, lam, search.seed, search.total_timesteps, rl.eval_every_steps, ir_fn,
                  Path(out_dir) / f"lambda_{lam:g}")
        for lam in search.grid
    ]
    table = pd.DataFrame(
        {
            "lambda": [r.lam for r in results],
            "best_val_ir": [r.best_val_ir for r in results],
            "best_timesteps": [r.best_timesteps for r in results],
        }
    )
    best = max(results, key=lambda r: r.best_val_ir)  # max는 동률이면 먼저 나온 것을 돌려준다
    return best.lam, table


@dataclass(frozen=True)
class FoldResult:
    lam: float
    lambda_table: pd.DataFrame
    seeds: tuple[TrainResult, ...]


def train_fold(
    make_env: EnvFactory, val_env: PortfolioEnv, rl: RLParams, ir_fn: IRFunction, out_dir: Path
) -> FoldResult:
    """폴드 하나의 λ 선택과 본 학습 (시드마다 total_timesteps_per_seed 스텝, 6.5절)."""
    out_dir = Path(out_dir)
    lam, table = select_lambda(make_env, val_env, rl, ir_fn, out_dir)
    table.to_csv(out_dir / "lambda_search.csv", index=False)
    seeds = tuple(
        train_one(make_env, val_env, lam, seed, rl.total_timesteps_per_seed, rl.eval_every_steps, ir_fn,
                  out_dir / f"seed_{seed}")
        for seed in rl.seeds
    )
    return FoldResult(lam, table, seeds)
