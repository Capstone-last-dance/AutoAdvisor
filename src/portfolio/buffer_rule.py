"""M4 버퍼 규칙 (6.5절, B 소유). 작업 B-03에서 구현한다."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.portfolio.params import BufferRuleParams


def buffer_rule_weights(
    holdings: Iterable[str],
    signals: pd.DataFrame,
    universe: Iterable[str],
    cooldown: Iterable[str],
    params: BufferRuleParams,
) -> pd.Series:
    """6.5절 버퍼 규칙 단계 1~4의 원시 목표 비중 w_raw (종목코드 → 비중, 나머지는 현금).

    단계 1: holdings는 직전 체결 후 보유 종목 집합 H다 (M5 상태에서 읽는다).
    단계 2: H 중 rank ≤ exit_rank_greater_than, crash_flag = 0, U_t 소속, 쿨다운 아님을 모두 만족하면 유지한다.
    단계 3: 유지 종목이 max_holdings보다 적으면 보유하지 않은 종목 중 rank ≤ enter_rank_max,
            crash_flag = 0, 쿨다운 아님을 만족하는 종목을 rank 오름차순으로 추가한다.
    단계 4: 보유 예정 종목에 1 ÷ 종목 수를 준다. 종목이 없으면 빈 Series(현금 100%)다.
    단계 5는 호출자가 결과를 apply_risk_layer(..., constructor="buffer")에 넘기는 것이다 (R-06).
    """
    raise NotImplementedError("작업 B-03 (기한 11/13)")
