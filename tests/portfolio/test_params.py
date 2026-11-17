"""부록 1 설정 읽기 (R-01, R-08, R-14)."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.portfolio.params import (
    BufferRuleParams,
    LambdaSearchParams,
    PreregValueMissing,
    RiskLayerParams,
    RLParams,
    load_prereg,
    vol_target_years,
)

ROOT = Path(__file__).resolve().parents[2]


def test_appendix1_values(cfg):
    rl = RLParams.from_prereg(cfg)
    assert (rl.candidate_top_k, rl.max_holdings, rl.slots) == (40, 20, 60)
    assert (rl.min_weight, rl.episode_steps, rl.total_timesteps_per_seed) == (0.005, 52, 300_000)
    assert rl.lambda_search == LambdaSearchParams(grid=(0.0, 0.5, 1.0, 2.0), seed=0, total_timesteps=100_000)
    assert (rl.eval_every_steps, rl.seeds) == (10_000, (0, 1, 2, 3, 4))

    risk = RiskLayerParams.from_prereg(cfg)
    assert (risk.stock_cap, risk.sector_cap, risk.unknown_sector_label) == (0.10, 0.30, "미분류")
    assert (risk.vol_lookback_days, risk.exposure_min, risk.exposure_max) == (20, 0.30, 1.00)
    m = risk.momentum_crash_regime
    assert (m.drawdown_threshold, m.drawdown_high_lookback_days, m.drawdown_recent_window_days) == (0.20, 252, 60)
    assert (m.rebound_lookback_days, m.rebound_threshold, m.exposure_cap) == (20, 0.10, 0.70)
    assert (risk.invariant_tolerance, risk.annualization_daily) == (1e-9, 252)

    buffer = BufferRuleParams.from_prereg(cfg)
    assert (buffer.enter_rank_max, buffer.exit_rank_greater_than, buffer.max_holdings) == (20, 40, 20)


def test_vol_target_years_from_appendix1(cfg):
    assert vol_target_years(cfg) == (2011, 2015)  # 6.9절: 2011~2015년, 2016년 이후 금지


def test_fill_value_stops_real_computation(cfg):
    risk = RiskLayerParams.from_prereg(cfg)
    assert risk.vol_target_annual is None
    with pytest.raises(PreregValueMissing):
        risk.require_vol_target()


@pytest.mark.parametrize(
    ("loader", "path", "value"),
    [
        (RiskLayerParams, ("risk_layer", "excess_weight_to"), "redistribute"),
        (RLParams, ("portfolio", "rl", "algorithm"), "SAC"),
        (BufferRuleParams, ("portfolio", "buffer_rule", "weighting"), "rank"),
    ],
)
def test_behaviour_not_implemented_is_rejected(cfg, loader, path, value):
    bad = copy.deepcopy(cfg)
    node = bad
    for name in path[:-1]:
        node = node[name]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        loader.from_prereg(bad)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("action_bound", 5.0),
        ("out_of_universe_fill", 0.5),
        ("candidate_overflow", "drop_worst_rank"),
        ("top_k_tie_break", "random"),
        ("episode_end", "terminated"),
        ("lambda_score", "mean_checkpoint_val_ir"),
        ("signal_dates", "all_signal_dates"),
        ("current_weight", "next_day_open"),
    ],
)
def test_rl_env_rules_other_than_dec21_are_rejected(cfg, key, value):
    """부록 1 portfolio.rl.env 8개 (DEC-21). 구현과 다른 값이면 학습하지 않는다."""
    bad = copy.deepcopy(cfg)
    bad["portfolio"]["rl"]["env"][key] = value
    with pytest.raises(ValueError, match=f"portfolio.rl.env.{key}"):
        RLParams.from_prereg(bad)


def test_rl_env_keys_must_match(cfg):
    missing = copy.deepcopy(cfg)
    del missing["portfolio"]["rl"]["env"]["episode_end"]
    with pytest.raises(ValueError, match="없는 키 \\['episode_end'\\]"):
        RLParams.from_prereg(missing)
    extra = copy.deepcopy(cfg)
    extra["portfolio"]["rl"]["env"]["reward_clip"] = 1.0
    with pytest.raises(ValueError, match="모르는 키 \\['reward_clip'\\]"):
        RLParams.from_prereg(extra)


def test_rl_env_constants_match_the_environment():
    from src.portfolio import rl_env
    from src.portfolio.params import RL_ENV_IMPLEMENTED

    assert RL_ENV_IMPLEMENTED["action_bound"] == rl_env.ACTION_BOUND
    assert RL_ENV_IMPLEMENTED["out_of_universe_fill"] == rl_env.OUT_OF_UNIVERSE_FILL


@pytest.mark.parametrize("grid", [[0, 1, 0.5, 2], [0, 0.5, 0.5, 2], [2, 1, 0.5, 0]])
def test_lambda_grid_must_be_ascending(cfg, grid):
    """select_lambda는 동률이면 격자에서 앞선 λ를 고른다. '동률이면 작은 λ'(DEC-21 (f))가 되려면 오름차순이어야 한다."""
    bad = copy.deepcopy(cfg)
    bad["portfolio"]["rl"]["lambda_search"]["grid"] = grid
    with pytest.raises(ValueError, match="오름차순"):
        RLParams.from_prereg(bad)


def test_subset_matches_configs_prereg(cfg):
    path = ROOT / "configs" / "prereg.yaml"
    if path.stat().st_size == 0:
        pytest.skip("configs/prereg.yaml이 아직 비어 있다 (작업 PM-07)")
    real = load_prereg(path)
    for section in ("calendar", "data", "labels", "costs", "execution", "splits", "portfolio", "risk_layer"):
        expected, actual = copy.deepcopy(cfg[section]), copy.deepcopy(real[section])
        if section == "risk_layer":  # B-02에서 채우는 값만 다를 수 있다
            expected.pop("vol_target_annual")
            actual.pop("vol_target_annual")
        assert actual == expected, f"부록 1 {section}이 테스트 입력과 다르다"
