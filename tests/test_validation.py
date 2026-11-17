"""M8 목표 비중 G0 검사에 B의 업종 시점 규칙과 off 예외를 적용한다."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest.validation import validate_g0_positions
from src.portfolio.risk_layer import RiskLayerViolation


PARAMS = SimpleNamespace(stock_cap=0.10, sector_cap=0.30,
                         invariant_tolerance=1e-9, unknown_sector_label="미분류")
SECTOR = pd.DataFrame({"ticker": ["005930", "000660"], "sector": ["IT", "IT"],
                       "valid_from": ["2020-01-01"] * 2, "valid_to": [None] * 2})


def positions(weights, cash):
    return pd.DataFrame({"date": ["2025-01-06"] * (len(weights) + 1),
                         "ticker": list(weights) + ["CASH"],
                         "target_weight": list(weights.values()) + [cash]})


def test_g0_accepts_valid_targets_and_checks_cash():
    frame = positions({"005930": 0.1, "000660": 0.1}, 0.8)
    assert validate_g0_positions(frame, SECTOR, "2025-01-06", PARAMS) == pytest.approx(0.8)
    frame.loc[frame.ticker == "CASH", "target_weight"] = 0.9
    with pytest.raises(ValueError, match="CASH"):
        validate_g0_positions(frame, SECTOR, "2025-01-06", PARAMS)


def test_g0_blocks_stock_cap_but_off_mode_allows_declared_exception():
    frame = positions({"005930": 0.4}, 0.6)
    with pytest.raises(RiskLayerViolation, match="stock_cap"):
        validate_g0_positions(frame, SECTOR, "2025-01-06", PARAMS)
    assert validate_g0_positions(frame, SECTOR, "2025-01-06", PARAMS, m8_mode="off") == pytest.approx(0.6)


def test_unknown_sectors_share_one_cap_group():
    frame = positions({"005930": 0.1, "000660": 0.1, "035420": 0.1, "005380": 0.1}, 0.6)
    with pytest.raises(RiskLayerViolation, match="sector_cap"):
        validate_g0_positions(frame, SECTOR.iloc[:0], "2025-01-06", PARAMS)
