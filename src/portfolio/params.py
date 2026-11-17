"""부록 1의 `portfolio`, `risk_layer` 설정 (B 소유).

값은 configs/prereg.yaml(부록 1)에서만 읽고 코드에 기본값을 두지 않는다 (R-01).
`FILL_`로 시작하는 값은 None으로 읽고, 그 값이 필요한 계산에서 예외를 낸다 (R-08).
필드 이름은 부록 1의 키 이름을 그대로 쓴다 (R-14).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FILL_PREFIX = "FILL_"
PREREG_PATH = Path(__file__).resolve().parents[2] / "configs" / "prereg.yaml"


class PreregValueMissing(RuntimeError):
    """부록 1의 값이 비어 있거나 `FILL_` 상태인데 실제 계산에 필요할 때 (R-08)."""


def load_prereg(path: str | Path = PREREG_PATH) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise PreregValueMissing(f"{path}가 비어 있다 (작업 PM-07에서 부록 1을 저장한다)")
    return cfg


def _unfilled(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(FILL_PREFIX)


def _expect(value: Any, expected: Any, key: str) -> None:
    """설정에 적힌 동작이 이 코드가 구현한 동작과 같은지 확인한다."""
    if value != expected:
        raise ValueError(f"{key} = {value!r}: 이 코드는 {expected!r}만 구현한다")


def vol_target_years(cfg: Mapping[str, Any]) -> tuple[int, int]:
    """vol_target_annual을 계산하는 연도 (6.9절: 2011년 ~ 2015년, 2016년 이후 데이터 금지).

    시작은 data.first_train_signal_year, 끝은 walk-forward 첫 테스트 연도의 전년이다.
    """
    first = int(cfg["data"]["first_train_signal_year"])
    last = min(int(y) for y in cfg["splits"]["walk_forward_test_years"]) - 1
    if last < first:
        raise ValueError("splits.walk_forward_test_years가 data.first_train_signal_year보다 이르다")
    return first, last


@dataclass(frozen=True)
class BufferRuleParams:
    """portfolio.buffer_rule (6.5절 버퍼 규칙)."""

    enter_rank_max: int
    exit_rank_greater_than: int
    max_holdings: int

    @classmethod
    def from_prereg(cls, cfg: Mapping[str, Any]) -> BufferRuleParams:
        b = cfg["portfolio"]["buffer_rule"]
        _expect(b["weighting"], "equal", "portfolio.buffer_rule.weighting")
        return cls(
            enter_rank_max=int(b["enter_rank_max"]),
            exit_rank_greater_than=int(b["exit_rank_greater_than"]),
            max_holdings=int(b["max_holdings"]),
        )


# portfolio.rl.env (DEC-21, design_B.md 8절 D3): rl_env.py·rl_train.py·rl_fold_report.py가 구현한 규칙.
# 사전등록 값이 이와 다르면 멈춘다. action_bound·out_of_universe_fill은 rl_env의 상수와 같아야 한다 (테스트가 확인)
RL_ENV_IMPLEMENTED: dict[str, Any] = {
    "action_bound": 10.0,  # (a) rl_env.ACTION_BOUND
    "out_of_universe_fill": 0.0,  # (b) rl_env.OUT_OF_UNIVERSE_FILL
    "candidate_overflow": "drop_nonheld_worst_rank",  # (c)
    "top_k_tie_break": "better_rank_first",  # (d)
    "episode_end": "truncated",  # (e)
    "lambda_score": "max_checkpoint_val_ir_then_smaller_lambda",  # (f) rl_train.select_lambda
    "signal_dates": "purged_h20_no_step_over_gaps",  # (g) PortfolioEnv(calendar=...), h = 20 purge
    "current_weight": "signal_date_close",  # (h)
}


@dataclass(frozen=True)
class LambdaSearchParams:
    """portfolio.rl.lambda_search (6.5절 λ 선택)."""

    grid: tuple[float, ...]
    seed: int
    total_timesteps: int


@dataclass(frozen=True)
class RLParams:
    """portfolio.rl (6.5절 RL 정책)."""

    candidate_top_k: int
    max_holdings: int
    min_weight: float
    slots: int
    episode_steps: int
    total_timesteps_per_seed: int
    lambda_search: LambdaSearchParams
    eval_every_steps: int
    seeds: tuple[int, ...]

    @classmethod
    def from_prereg(cls, cfg: Mapping[str, Any]) -> RLParams:
        p = cfg["portfolio"]["rl"]
        for key, expected in (
            ("library", "stable-baselines3"),
            ("algorithm", "PPO"),
            ("policy", "MlpPolicy"),
            ("checkpoint_selection", "val_ir"),
            ("test_action", "deterministic_mean"),
            ("other_hyperparameters", "sb3_defaults"),
        ):
            _expect(p[key], expected, f"portfolio.rl.{key}")
        env = p["env"]
        if set(env) != set(RL_ENV_IMPLEMENTED):
            raise ValueError(
                f"portfolio.rl.env 키가 구현과 다르다. 없는 키 {sorted(set(RL_ENV_IMPLEMENTED) - set(env))}, "
                f"모르는 키 {sorted(set(env) - set(RL_ENV_IMPLEMENTED))}"
            )
        for key, expected in RL_ENV_IMPLEMENTED.items():
            _expect(env[key], expected, f"portfolio.rl.env.{key}")
        ls = p["lambda_search"]
        grid = [float(x) for x in ls["grid"]]
        if grid != sorted(set(grid)):
            # select_lambda는 동률이면 격자에서 앞선 λ를 고른다. 그게 '작은 λ'(env.lambda_score)가 되려면 오름차순이어야 한다
            raise ValueError(f"portfolio.rl.lambda_search.grid = {grid}: 중복 없는 오름차순이어야 한다")
        params = cls(
            candidate_top_k=int(p["candidate_top_k"]),
            max_holdings=int(p["max_holdings"]),
            min_weight=float(p["min_weight"]),
            slots=int(p["slots"]),
            episode_steps=int(p["episode_steps"]),
            total_timesteps_per_seed=int(p["total_timesteps_per_seed"]),
            lambda_search=LambdaSearchParams(
                grid=tuple(float(x) for x in ls["grid"]),
                seed=int(ls["seed"]),
                total_timesteps=int(ls["total_timesteps"]),
            ),
            eval_every_steps=int(p["eval_every_steps"]),
            seeds=tuple(int(s) for s in p["seeds"]),
        )
        # 6.5절: 후보 슬롯 = (rank ≤ candidate_top_k) ∪ (보유 ≤ max_holdings)
        if params.slots < params.candidate_top_k + params.max_holdings:
            raise ValueError("portfolio.rl.slots < candidate_top_k + max_holdings")
        return params


@dataclass(frozen=True)
class RegimeParams:
    """risk_layer.momentum_crash_regime (6.9절 Step 5)."""

    drawdown_threshold: float
    drawdown_high_lookback_days: int
    drawdown_recent_window_days: int
    rebound_lookback_days: int
    rebound_threshold: float
    exposure_cap: float


@dataclass(frozen=True)
class RiskLayerParams:
    """risk_layer (6.9절). annualization_daily는 calendar.annualization.daily다."""

    stock_cap: float
    sector_cap: float
    unknown_sector_label: str
    vol_lookback_days: int
    vol_target_annual: float | None  # FILL_ 상태면 None (Q-12, 작업 B-02)
    exposure_min: float
    exposure_max: float
    momentum_crash_regime: RegimeParams
    invariant_tolerance: float
    annualization_daily: int

    def require_vol_target(self) -> float:
        if self.vol_target_annual is None:
            raise PreregValueMissing("risk_layer.vol_target_annual이 FILL_ 상태다 (Q-12, 작업 B-02)")
        return self.vol_target_annual

    @classmethod
    def from_prereg(cls, cfg: Mapping[str, Any]) -> RiskLayerParams:
        r = cfg["risk_layer"]
        _expect(r["excess_weight_to"], "cash", "risk_layer.excess_weight_to")
        m = r["momentum_crash_regime"]
        vt = r["vol_target_annual"]
        return cls(
            stock_cap=float(r["stock_cap"]),
            sector_cap=float(r["sector_cap"]),
            unknown_sector_label=str(r["unknown_sector_label"]),
            vol_lookback_days=int(r["vol_lookback_days"]),
            vol_target_annual=None if _unfilled(vt) else float(vt),
            exposure_min=float(r["exposure_min"]),
            exposure_max=float(r["exposure_max"]),
            momentum_crash_regime=RegimeParams(
                drawdown_threshold=float(m["drawdown_threshold"]),
                drawdown_high_lookback_days=int(m["drawdown_high_lookback_days"]),
                drawdown_recent_window_days=int(m["drawdown_recent_window_days"]),
                rebound_lookback_days=int(m["rebound_lookback_days"]),
                rebound_threshold=float(m["rebound_threshold"]),
                exposure_cap=float(m["exposure_cap"]),
            ),
            invariant_tolerance=float(r["invariant_tolerance"]),
            annualization_daily=int(cfg["calendar"]["annualization"]["daily"]),
        )
