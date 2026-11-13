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
# available_at < cutoff_at(t)인 경보만 쓴다 (6.10절, R-03). 컷오프 뒤 경보는 엔진의 체결일 cooldowns가 막는다.
# 구현은 PM의 src.events.alerts:cooldown_at (PR #34). 날짜별 경보 파일이 하나도 없으면 AlertsNotStarted,
# 필요한 날짜(t 포함)의 파일이 없으면 AlertsNotCollected (PR #51, 둘 다 FileNotFoundError 하위)
CooldownFunction = Callable[[str], Iterable[str]]

# M9 체결 지도 (PM): (첫 신호일, 마지막 날) → (cooldowns, event_liquidations). 둘 다 체결일(YYYY-MM-DD) → 종목코드
# 집합이고, 첫 신호일의 다음 거래일부터 마지막 날까지 그날 걸린 모든 경보를 담는다 (design_B.md 5.2절 '체결' 행).
# forward의 지난 보유 재생에서 C 엔진에 넘긴다. 구현은 PM의 src.events.alerts:engine_alerts (PR #34). 예외는 위와 같다
EngineAlertsFunction = Callable[[str, str], tuple[Mapping[str, frozenset[str]], Mapping[str, frozenset[str]]]]

# 지난 forward 기록 (D, M7): 신호일 t → t보다 앞선 forward 기록의 positions (신호일 YYYY-MM-DD → 9.3절 positions 행).
# 태그 ft-{YYYYMMDD}에 커밋된 파일을 읽는다 (PM 11/7 의견). positions가 없는 기록(11/6, 11/13)은 넣지 않는다.
# 구현은 D의 src.pipeline.forward_test:past_positions (PR #31)
RecordedPositionsFunction = Callable[[str], Mapping[str, Any]]


class ForwardEngine(Protocol):
    """forward 지난 보유 재생용 M5 호출 (C, PR #22). run_period와 같은 일별 체결 로직이지만 next_signal_date
    종가에서 멈추므로 그 다음 거래일 가격이 없어도 된다 (forward 실행은 t+1 장 전, DEC-04)."""

    def initial_state(self) -> Any:
        ...

    def run_to_signal_close(
        self, state: Any, targets: Mapping[str, float], signal_date: str, next_signal_date: str
    ) -> Any:
        """결과의 weights_obs는 next_signal_date 종가의 보유 비중이고, state는 다음 호출에 그대로 넘긴다."""
        ...
