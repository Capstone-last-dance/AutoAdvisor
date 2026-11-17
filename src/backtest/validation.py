"""M5 G0 목표 비중 자동 검사. B의 M8 업종 선택·불변식을 그대로 사용한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.params import RiskLayerParams
from src.portfolio.risk_layer import check_invariants, sectors_at


def validate_g0_positions(
    positions: pd.DataFrame, sector: pd.DataFrame, date: str,
    params: RiskLayerParams, *, m8_mode: str = "full",
) -> float:
    """신호일 ``positions``의 목표 비중을 검사하고 현금 비중을 돌려준다.

    G7의 M8 미적용 실행만 ``m8_mode='off'``를 전달한다. 실행 설정의
    mode를 호출자가 확인하고 전달해야 하며, 파일에 mode가 없다고 자동으로
    한도 검사를 끄지 않는다.
    """

    if m8_mode not in {"full", "no_exposure", "off"}:
        raise ValueError("알 수 없는 M8 모드입니다")
    required = {"date", "ticker", "target_weight"}
    if not required.issubset(positions):
        raise ValueError(f"positions에 필수 열이 없습니다: {sorted(required - set(positions))}")
    rows = positions.loc[pd.to_datetime(positions["date"]) == pd.Timestamp(date)]
    if rows.empty or rows["ticker"].duplicated().any():
        raise ValueError("해당 신호일 positions가 없거나 중복 종목이 있습니다")
    cash_rows = rows.loc[rows["ticker"] == "CASH"]
    if len(cash_rows) != 1:
        raise ValueError("CASH 행은 정확히 하나여야 합니다")
    stocks = rows.loc[rows["ticker"] != "CASH"]
    if (stocks["target_weight"] <= 0).any():
        raise ValueError("종목 행은 양의 목표 비중만 기록합니다")
    sectors = sectors_at(sector, date)
    labels = [sectors.get(ticker, params.unknown_sector_label) for ticker in stocks["ticker"]]
    codes = pd.factorize(pd.Index(labels), sort=False)[0].astype(np.int64)
    implied_cash = check_invariants(
        stocks["target_weight"].to_numpy(dtype=np.float64), codes,
        params, mode=m8_mode,
    )
    recorded_cash = float(cash_rows.iloc[0]["target_weight"])
    if not np.isfinite(recorded_cash) or abs(implied_cash - recorded_cash) > params.invariant_tolerance:
        raise ValueError("CASH 목표 비중과 종목 목표 비중의 합이 1이 아닙니다")
    return implied_cash
