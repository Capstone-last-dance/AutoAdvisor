"""B가 다른 역할의 산출물에 기대하는 호출 형태 (R-17).

실제 구현은 각 담당 역할이 만든다. B 코드는 이 형태로만 호출하고, 실제 함수가 나오면 어댑터로
연결한다 (작업 B-05). 테스트에서는 tests/portfolio/fakes.py의 가짜 구현을 쓴다.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class PeriodResult:
    """신호일 t의 목표 비중을 t+1일 종가에 체결하고, 다음 신호일 t′의 체결일(t′+1) 체결 직전까지
    운용한 결과. M5(C)가 계산한다 (4.3절, 4.8절, 6.6절 이벤트 청산)."""

    nav_start: float  # t+1일 종가, 체결 직전 NAV
    nav_end: float  # t′+1일 종가, 체결 직전 NAV
    nav_max: float  # t+1일 ~ t′+1일 일간 NAV의 최댓값 (보상의 DD 고점)
    r_net: float  # nav_end / nav_start − 1. t+1일 체결 비용을 포함한다
    r_bm: float  # BM(t′+1) / BM(t+1) − 1
    turnover: float  # t+1일 체결의 회전율 (6.6절)
    weights_obs: Mapping[str, float]  # t′일 종가 기준 보유 종목 비중 (관측의 '현재 비중')
    cash_obs: float  # t′일 종가 기준 현금 비중
    state: Any  # 다음 run_period에 넘길 엔진 상태 (t′+1일 체결 직전)


class ExecutionEngine(Protocol):
    """M5 체결 엔진 (C 소유). RL 환경은 체결, 비용, NAV를 다시 구현하지 않고 이것만 호출한다 (6.5절)."""

    def initial_state(self) -> Any:
        """현금 100%, NAV 1의 상태."""
        ...

    def run_period(
        self, state: Any, targets: Mapping[str, float], signal_date: str, next_signal_date: str
    ) -> PeriodResult:
        """targets(M8을 거친 종목 비중, 나머지는 현금)를 signal_date 다음 거래일 종가에 체결하고
        next_signal_date 다음 거래일 종가(체결 직전)까지 운용한다."""
        ...


# 6.6절 IR (M5, C): 주간 초과수익 a_k 배열 → mean(a) ÷ std(a, ddof=1) × √52
IRFunction = Callable[[np.ndarray], float]

# M9 쿨다운 (PM): 신호일 t → 체결일 t+1에 신규 진입이 금지된 종목코드.
# available_at < cutoff_at(t)인 경보만 쓴다 (6.10절, R-03). 컷오프 뒤 경보는 엔진의 체결일 cooldowns가 막는다
CooldownFunction = Callable[[str], Iterable[str]]
