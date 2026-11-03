"""M5·M4·M7이 공유할 체결 가능 여부, 비용, 현금·NAV 계산의 순수 함수.

목표 비중을 주문 수량으로 바꾸는 규칙은 비용 포함 시 매수 가능 현금의
처리 방식이 아직 미합의라 이 모듈에 넣지 않는다. 호출자는 주문 수량을
명시하고, 이 모듈은 같은 주문에 항상 같은 체결 결과를 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Mapping


@dataclass(frozen=True)
class CostRates:
    sell_tax: float
    commission_per_side: float
    slippage_per_side: float
    stress_multiplier: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.sell_tax,
            self.commission_per_side,
            self.slippage_per_side,
            self.stress_multiplier,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("비용률은 유한해야 합니다")
        if any(value < 0 for value in values[:3]) or self.stress_multiplier < 0:
            raise ValueError("비용률과 배율은 음수일 수 없습니다")


@dataclass(frozen=True)
class Fill:
    ticker: str
    signed_quantity: float
    price: float
    notional: float
    cost: float


@dataclass(frozen=True)
class ExecutionResult:
    holdings: dict[str, float]
    cash: float
    total_cost: float
    fills: tuple[Fill, ...]
    rejected: dict[str, str]


def trade_cost(*, buy_notional: float, sell_notional: float, rates: CostRates) -> float:
    """4.8절의 매수·매도 비용. 두 금액은 양수 또는 0이다."""

    if not all(
        isfinite(value) and value >= 0 for value in (buy_notional, sell_notional)
    ):
        raise ValueError("매매 금액은 유한한 0 이상이어야 합니다")
    buy_rate = rates.commission_per_side + rates.slippage_per_side
    sell_rate = rates.sell_tax + buy_rate
    return rates.stress_multiplier * (
        buy_notional * buy_rate + sell_notional * sell_rate
    )


def blocked_reason(
    *,
    side: str,
    previous_close: float,
    execution_close: float,
    is_halted: bool,
    limit_move_abs_return: float,
) -> str | None:
    """4.3절의 거래정지·상한가 근접·하한가 근접 미체결 이유."""

    if side not in {"buy", "sell"}:
        raise ValueError("side는 buy 또는 sell이어야 합니다")
    if not isfinite(previous_close) or previous_close <= 0:
        raise ValueError("전일 종가는 양수여야 합니다")
    if not isfinite(execution_close) or execution_close <= 0:
        raise ValueError("체결일 종가는 양수여야 합니다")
    if not isfinite(limit_move_abs_return) or limit_move_abs_return <= 0:
        raise ValueError("가격 제한 기준은 양수여야 합니다")
    if is_halted:
        return "halted"
    price_return = execution_close / previous_close - 1.0
    if side == "buy" and (
        price_return >= limit_move_abs_return
        or isclose(price_return, limit_move_abs_return, rel_tol=0.0, abs_tol=1e-12)
    ):
        return "near_upper_limit"
    if side == "sell" and (
        price_return <= -limit_move_abs_return
        or isclose(price_return, -limit_move_abs_return, rel_tol=0.0, abs_tol=1e-12)
    ):
        return "near_lower_limit"
    return None


def portfolio_nav(
    *, holdings: Mapping[str, float], closes: Mapping[str, float], cash: float
) -> float:
    """6.6절: 일별 NAV = 보유 수량 × 수정종가의 합 + 현금."""

    if not isfinite(cash) or cash < 0:
        raise ValueError("현금은 유한한 0 이상이어야 합니다")
    nav = float(cash)
    for ticker, quantity in holdings.items():
        if not isfinite(quantity) or quantity < 0:
            raise ValueError(f"{ticker} 보유 수량은 유한한 0 이상이어야 합니다")
        if ticker not in closes:
            raise ValueError(f"{ticker} 평가 가격이 없습니다")
        price = closes[ticker]
        # 상장폐지 정리매매 가격이 없는 경우에만 0 평가 가격을 전달할 수 있다.
        if not isfinite(price) or price < 0:
            raise ValueError(f"{ticker} 평가 가격은 유한한 0 이상이어야 합니다")
        nav += quantity * price
    return nav


def execute_orders(
    *,
    orders: Mapping[str, float],
    holdings: Mapping[str, float],
    cash: float,
    execution_closes: Mapping[str, float],
    previous_closes: Mapping[str, float],
    halted_tickers: frozenset[str],
    rates: CostRates,
    limit_move_abs_return: float,
) -> ExecutionResult:
    """명시된 수량을 매도 후 매수하고, 불가능한 주문은 이유와 함께 남긴다.

    양수 주문은 매수, 음수는 매도다. 비용까지 낼 현금이 부족하면 임의로
    수량을 줄이지 않고 오류를 낸다. 그 조정 규칙은 B·C 합의 대상이다.
    """

    positions = dict(holdings)
    portfolio_nav(holdings=positions, closes=execution_closes, cash=cash)
    current_cash = float(cash)
    fills: list[Fill] = []
    rejected: dict[str, str] = {}

    for ticker, quantity in orders.items():
        if not isfinite(quantity):
            raise ValueError(f"{ticker} 주문 수량이 유한하지 않습니다")
        if quantity < 0 and -quantity > positions.get(ticker, 0.0) + 1e-12:
            raise ValueError(f"{ticker} 보유 수량보다 많이 매도할 수 없습니다")

    # 매도 대금은 같은 체결 시점의 매수에 사용할 수 있다.
    for ticker, quantity in sorted(orders.items(), key=lambda item: item[1]):
        if quantity == 0:
            continue
        if ticker not in execution_closes or ticker not in previous_closes:
            raise ValueError(f"{ticker} 체결일 또는 전일 가격이 없습니다")
        side = "buy" if quantity > 0 else "sell"
        reason = blocked_reason(
            side=side,
            previous_close=previous_closes[ticker],
            execution_close=execution_closes[ticker],
            is_halted=ticker in halted_tickers,
            limit_move_abs_return=limit_move_abs_return,
        )
        if reason:
            rejected[ticker] = reason
            continue
        notional = abs(quantity) * execution_closes[ticker]
        cost = trade_cost(
            buy_notional=notional if quantity > 0 else 0.0,
            sell_notional=notional if quantity < 0 else 0.0,
            rates=rates,
        )
        if quantity > 0:
            if notional + cost > current_cash + 1e-9:
                raise ValueError(f"{ticker} 매수 금액과 비용이 가용 현금을 넘습니다")
            current_cash -= notional + cost
        else:
            current_cash += notional - cost
            if current_cash < -1e-9:
                raise ValueError(f"{ticker} 매도 대금과 현금으로 비용을 낼 수 없습니다")
        positions[ticker] = positions.get(ticker, 0.0) + quantity
        if abs(positions[ticker]) < 1e-12:
            del positions[ticker]
        fills.append(Fill(ticker, quantity, execution_closes[ticker], notional, cost))

    return ExecutionResult(
        holdings=positions,
        cash=max(current_cash, 0.0),
        total_cost=sum(fill.cost for fill in fills),
        fills=tuple(fills),
        rejected=rejected,
    )
