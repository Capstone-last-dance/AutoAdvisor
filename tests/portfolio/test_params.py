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


def test_subset_matches_configs_prereg(cfg):
    path = ROOT / "configs" / "prereg.yaml"
    if path.stat().st_size == 0:
        pytest.skip("configs/prereg.yaml이 아직 비어 있다 (작업 PM-07)")
    real = load_prereg(path)
    for section in ("calendar", "costs", "portfolio", "risk_layer"):
        expected, actual = copy.deepcopy(cfg[section]), copy.deepcopy(real[section])
        if section == "risk_layer":  # B-02에서 채우는 값만 다를 수 있다
            expected.pop("vol_target_annual")
            actual.pop("vol_target_annual")
        assert actual == expected, f"부록 1 {section}이 테스트 입력과 다르다"
