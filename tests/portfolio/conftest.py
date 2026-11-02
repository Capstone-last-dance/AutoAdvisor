from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from src.portfolio.params import RiskLayerParams, RLParams
from tests.portfolio.fakes import make_market

HERE = Path(__file__).parent
VOL_TARGET_FOR_TESTS = 0.15  # 단위 테스트 전용 임의값 (R-08). 실제 값은 작업 B-02에서 계산한다


@pytest.fixture(scope="session")
def cfg() -> dict:
    with open(HERE / "prereg_subset.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def rl_params(cfg) -> RLParams:
    return RLParams.from_prereg(cfg)


@pytest.fixture(scope="session")
def risk_params(cfg) -> RiskLayerParams:
    return dataclasses.replace(RiskLayerParams.from_prereg(cfg), vol_target_annual=VOL_TARGET_FOR_TESTS)


@pytest.fixture(scope="session")
def market(risk_params):
    return make_market(risk_params)
