"""B의 RL 환경과 C의 백테스트가 공유하는 일별 체결·평가 엔진.

입력 시계열은 KRX 거래일 순서의 수정종가다. 상장 전·폐지 후 가격은
결측이어도 되지만 보유·목표 종목의 평가 가격은 양수여야 한다.
거래정지일의 평가 가격은 호출자가 직전 유효 가격으로 제공해야 한다.
``delistings``의 날짜는 정리매매 마지막 가격이 있는 거래일이다.
A의 ``delisted.parquet``에 있는 ``delist_date``가 마지막 가격일의 다음
거래일이면 어댑터가 직전 거래일로 바꿔 넘겨야 한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from src.backtest.accounting import CostRates, Fill, blocked_reason, execute_orders, portfolio_nav, trade_cost
from src.portfolio.interfaces import PeriodResult


@dataclass(frozen=True)
class EngineState:
    holdings: dict[str, float]
    cash: float
    asof_date: str | None = None
    pending_liquidations: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SignalCloseResult:
    """다음 신호일 종가까지의 forward 재생 결과 (다음 체결일 가격 미사용)."""

    nav_start: float
    nav_end: float
    weights_obs: Mapping[str, float]
    cash_obs: float
    state: EngineState


class ExecutionEngine:
    """``src.portfolio.interfaces.ExecutionEngine``의 실제 구현.

    ``event_liquidations``와 ``cooldowns``는 적용할 *거래일*별 종목 집합이다.
    목표 비중은 체결 직전 NAV 기준이며, 현금이 모자라면 매도 후 모든
    가능한 매수를 동일 비율로 줄인다(DEC-18). 리밸런싱 배율은
    원장(ledger)에 남긴다.
    """

    def __init__(
        self,
        closes: pd.DataFrame,
        bm_close: pd.Series,
        rates: CostRates,
        *,
        halted: pd.DataFrame | None = None,
        event_liquidations: Mapping[str, frozenset[str]] | None = None,
        cooldowns: Mapping[str, frozenset[str]] | None = None,
        delistings: Mapping[str, tuple[str, float | None]] | None = None,
        limit_move_abs_return: float = 0.295,
        record_ledger: bool = False,
    ) -> None:
        if closes.empty or closes.index.has_duplicates or closes.columns.has_duplicates:
            raise ValueError("종가 표는 비어 있지 않고 날짜·종목이 고유해야 합니다")
        if not closes.index.is_monotonic_increasing:
            raise ValueError("거래일은 오름차순이어야 합니다")
        if not isfinite(limit_move_abs_return) or limit_move_abs_return <= 0:
            raise ValueError("가격 제한 기준은 양수여야 합니다")
        self.days = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in closes.index)
        self.day_index = {day: i for i, day in enumerate(self.days)}
        if len(self.day_index) != len(self.days):
            raise ValueError("거래일은 중복될 수 없습니다")
        self.tickers = tuple(str(ticker) for ticker in closes.columns)
        self.col_index = {ticker: i for i, ticker in enumerate(self.tickers)}
        if len(self.col_index) != len(self.tickers):
            raise ValueError("종목코드는 고유해야 합니다")
        self.prices = closes.to_numpy(dtype=np.float64, copy=True)
        if np.isinf(self.prices).any() or ((self.prices <= 0) & ~np.isnan(self.prices)).any():
            raise ValueError("존재하는 수정종가는 유한한 양수여야 합니다")
        bm = bm_close.reindex(closes.index).to_numpy(dtype=np.float64)
        if not np.isfinite(bm).all() or (bm <= 0).any():
            raise ValueError("벤치마크 종가는 모든 거래일에 양수여야 합니다")
        self.bm = bm
        if halted is None:
            self.halted = np.zeros_like(self.prices, dtype=bool)
            self.halted_known = np.ones_like(self.prices, dtype=bool)
        else:
            if not halted.index.equals(closes.index) or not halted.columns.equals(closes.columns):
                raise ValueError("거래정지 표의 날짜·종목 순서가 종가 표와 다릅니다")
            self.halted_known = halted.notna().to_numpy(dtype=bool, copy=True)
            self.halted = halted.fillna(False).to_numpy(dtype=bool, copy=True)
        self.events = {day: frozenset(names) for day, names in (event_liquidations or {}).items()}
        self.cooldowns = {day: frozenset(names) for day, names in (cooldowns or {}).items()}
        for names in (*self.events.values(), *self.cooldowns.values()):
            if not names.issubset(self.col_index):
                raise ValueError("이벤트·쿨다운에 알 수 없는 종목코드가 있습니다")
        self.delistings = dict(delistings or {})
        self.relisting_index: dict[str, int] = {}
        for ticker, (day, price) in self.delistings.items():
            if ticker not in self.col_index or day not in self.day_index:
                raise ValueError("상장폐지 종목과 날짜는 입력 시계열에 있어야 합니다")
            if price is not None and (not isfinite(price) or price < 0):
                raise ValueError("정리매매 최종 가격은 0 이상이어야 합니다")
            # v1 종목코드 재사용: 폐지 뒤 가격 공백 다음에 나타나는 첫 가격은
            # 새 회사의 상장 구간이다. 구회사 폐지 제한을 그 날부터 해제한다.
            after = self.prices[self.day_index[day] + 1:, self.col_index[ticker]]
            if ticker in {"101970", "036220"} and len(after) and np.isnan(after[0]):
                resumed = np.flatnonzero(np.isfinite(after))
                if len(resumed):
                    self.relisting_index[ticker] = self.day_index[day] + 1 + int(resumed[0])
        self.rates = rates
        self.limit_move_abs_return = limit_move_abs_return
        self.record_ledger = record_ledger
        self.ledger: list[dict] = []
        self.rebalance_count = 0
        self.scaled_rebalance_count = 0

    def initial_state(self) -> EngineState:
        return EngineState({}, 1.0)

    def with_alert_schedule(
        self,
        cooldowns: Mapping[str, frozenset[str]],
        event_liquidations: Mapping[str, frozenset[str]],
    ) -> ExecutionEngine:
        """같은 가격·비용 엔진에 M9 일정을 붙인 독립 실행 인스턴스."""
        schedules = (cooldowns, event_liquidations)
        for schedule in schedules:
            if any(day not in self.day_index for day in schedule):
                raise ValueError("M9 일정에 거래일이 아닌 날짜가 있습니다")
            if any(not set(names).issubset(self.col_index) for names in schedule.values()):
                raise ValueError("M9 일정에 알 수 없는 종목코드가 있습니다")
        result = copy(self)
        result.cooldowns = {day: frozenset(names) for day, names in cooldowns.items()}
        result.events = {day: frozenset(names) for day, names in event_liquidations.items()}
        result.ledger = []
        result.rebalance_count = 0
        result.scaled_rebalance_count = 0
        return result

    def _prices(self, day_index: int, holdings: Mapping[str, float]) -> dict[str, float]:
        out = {}
        for ticker in holdings:
            column = self.col_index[ticker]
            price = float(self.prices[day_index, column])
            if self.halted[day_index, column]:
                # DEC-25: 정지일 원종가가 변해도 마지막 거래일 수정종가로 평가.
                previous = day_index - 1
                while previous >= 0 and self.halted[previous, column]:
                    previous -= 1
                price = float(self.prices[previous, column]) if previous >= 0 else float("nan")
            if not isfinite(price) or price <= 0:
                raise ValueError(f"{self.days[day_index]} {ticker} 평가 가격이 없습니다")
            if not self.halted_known[day_index, column]:
                raise ValueError(f"{self.days[day_index]} {ticker} 거래정지 여부가 없습니다")
            out[ticker] = price
        return out

    def _orders(
        self, holdings: Mapping[str, float], targets: Mapping[str, float], cash: float,
        prices: Mapping[str, float], *, forbidden_buys: frozenset[str],
    ) -> tuple[dict[str, float], float, float]:
        nav = portfolio_nav(holdings=holdings, closes=prices, cash=cash)
        names = set(holdings) | set(targets)
        orders = {
            ticker: targets.get(ticker, 0.0) * nav / prices[ticker] - holdings.get(ticker, 0.0)
            for ticker in names
        }
        orders = {ticker: qty for ticker, qty in orders.items() if abs(qty) > 1e-12}
        # 체결 가능 매도를 먼저 고려한다. 매수 제한 주문에는 현금을 배정하지 않는다.
        sell_cash = cash
        for ticker, qty in orders.items():
            if qty >= 0:
                continue
            # 거래정지·하한가 매도는 execute_orders가 기록한다. 여기서는
            # 예상 현금에 포함하지 않아 매수 배율을 과대평가하지 않는다.
            if ticker in self._blocked_sells:
                continue
            notional = -qty * prices[ticker]
            sell_cash += notional * (1.0 - self._sell_rate)
        allowed = {ticker: qty for ticker, qty in orders.items() if qty > 0 and ticker not in forbidden_buys and ticker not in self._blocked_buys}
        needed = sum(qty * prices[ticker] * (1.0 + self._buy_rate) for ticker, qty in allowed.items())
        scale = min(1.0, max(0.0, sell_cash) / needed) if needed > 0 else 1.0
        for ticker, qty in tuple(orders.items()):
            if qty > 0:
                if ticker in allowed:
                    orders[ticker] *= scale
                elif ticker not in self._blocked_buys:
                    del orders[ticker]
        return orders, nav, scale

    def run_period(
        self, state: EngineState, targets: Mapping[str, float], signal_date: str,
        next_signal_date: str,
    ) -> PeriodResult:
        """다음 신호일의 다음 체결일 종가까지 평가한다 (M4·M5)."""
        return self._run_interval(state, targets, signal_date, next_signal_date,
                                  stop_at_signal_close=False)

    def run_to_signal_close(
        self, state: EngineState, targets: Mapping[str, float], signal_date: str,
        next_signal_date: str,
    ) -> SignalCloseResult:
        """t+1 종가에 체결하고 t′ 종가의 실제 보유와 상태를 반환한다.

        forward test에서는 첫 신호일에 현금 100%로 시작하고, 이후 이 함수가
        반환한 상태를 다음 호출에 넘긴다. t′+1 가격은 필요하지 않다.
        """
        result = self._run_interval(state, targets, signal_date, next_signal_date,
                                    stop_at_signal_close=True)
        return SignalCloseResult(result.nav_start, result.nav_end, result.weights_obs,
                                 result.cash_obs, result.state)

    def _run_interval(
        self, state: EngineState, targets: Mapping[str, float], signal_date: str,
        next_signal_date: str, *, stop_at_signal_close: bool,
    ) -> PeriodResult:
        if signal_date not in self.day_index or next_signal_date not in self.day_index:
            raise ValueError("신호일은 거래일 목록에 있어야 합니다")
        signal_i, next_i = self.day_index[signal_date], self.day_index[next_signal_date]
        if next_i <= signal_i or (not stop_at_signal_close and next_i + 1 >= len(self.days)):
            raise ValueError("평가 종료일이 거래일 목록에 있어야 합니다")
        expected_asof = signal_date if stop_at_signal_close else self.days[signal_i + 1]
        if state.asof_date is not None and state.asof_date != expected_asof:
            raise ValueError(f"상태의 평가일은 {expected_asof}이어야 합니다")
        if any(ticker not in self.col_index for ticker in set(targets) | set(state.holdings)):
            raise ValueError("알 수 없는 종목코드가 있습니다")
        if any(not isfinite(weight) or weight < 0 for weight in targets.values()) or sum(targets.values()) > 1 + 1e-9:
            raise ValueError("목표 비중은 음수가 아니고 합이 1 이하여야 합니다")
        first_i = signal_i + 1
        end_i = next_i if stop_at_signal_close else next_i + 1
        holdings, cash = dict(state.holdings), float(state.cash)
        # 마지막 정리매매일이 신호일이면 첫 체결일에는 가격이 없다.
        # 보유분은 마지막 가격일에 이미 청산됐어야 하므로, 미보유 목표만
        # 가격 조회 전에 제외한다. 재상장 첫날부터는 새 회사 매수를 허용한다.
        expired_targets = {
            ticker: weight for ticker, weight in targets.items()
            if ticker not in holdings and ticker in self.delistings
            and self.day_index[self.delistings[ticker][0]] < first_i
            < self.relisting_index.get(ticker, len(self.days))
        }
        pending = set(state.pending_liquidations)
        pending_rebalance_sells: dict[str, float] = {}
        nav_start = portfolio_nav(holdings=holdings, closes=self._prices(first_i, holdings), cash=cash)
        if nav_start <= 0:
            raise ValueError("시작 NAV가 0 이하이면 다음 구간을 운용할 수 없습니다")
        nav_max = 0.0
        turnover = 0.0
        weights_obs: dict[str, float] = {}
        cash_obs = 1.0
        for i in range(first_i, end_i + 1):
            day = self.days[i]
            active = set(holdings) | (set(targets) - set(expired_targets) if i == first_i else set())
            px = self._prices(i, active)
            nav_before = portfolio_nav(holdings=holdings, closes=px, cash=cash)
            pending.update(self.events.get(day, frozenset()))
            # 다음 체결일은 이 호출에서 평가만 한다. 새 목표와 같은 날의
            # 청산은 다음 호출의 리밸런싱보다 먼저 처리된다.
            if i == end_i and not stop_at_signal_close:
                nav_max = max(nav_max, nav_before)
                if self.record_ledger:
                    self.ledger.append({"date": day, "nav_before": nav_before, "nav_after": nav_before,
                                        "cash": cash, "holdings": dict(holdings), "cost": 0.0,
                                        "fills": (), "rejected": {}, "scale": 1.0})
                break
            day_targets = dict(targets) if i == first_i else {}
            if i == first_i:
                for ticker in expired_targets:
                    day_targets.pop(ticker)
            if i != first_i:
                day_targets = {ticker: qty * px[ticker] / nav_before for ticker, qty in holdings.items()}
                for ticker, target_weight in pending_rebalance_sells.items():
                    if ticker in holdings:
                        day_targets[ticker] = min(target_weight, day_targets[ticker])
            # 주문·보유 대상만 검사한다. 다른 종목의 과거 폐지는 이 구간의
            # 체결에 영향이 없고, 전체 폐지 목록을 매 스텝 순회할 필요도 없다.
            delisted_today = {
                ticker for ticker in set(holdings) | set(day_targets)
                if ticker in self.delistings and self.day_index[self.delistings[ticker][0]] <= i
                and i < self.relisting_index.get(ticker, len(self.days))
            }
            for ticker in pending & px.keys():
                day_targets[ticker] = 0.0
            for ticker in delisted_today & holdings.keys():
                # 정리매매 가격으로 별도 청산하므로 보통 매도 주문을 내지 않는다.
                day_targets[ticker] = holdings.get(ticker, 0.0) * px[ticker] / nav_before
            forbidden = self.cooldowns.get(day, frozenset()) | frozenset(pending) | frozenset(delisted_today)
            self._buy_rate = self.rates.stress_multiplier * (self.rates.commission_per_side + self.rates.slippage_per_side)
            self._sell_rate = self.rates.stress_multiplier * (self.rates.sell_tax + self.rates.commission_per_side + self.rates.slippage_per_side)
            previous = {
                ticker: (px[ticker] if self.relisting_index.get(ticker) == i
                         else self._prices(i - 1, {ticker: 0.0})[ticker])
                for ticker in px
            }
            requested = {ticker: day_targets.get(ticker, 0.0) - holdings.get(ticker, 0.0) * px[ticker] / nav_before
                         for ticker in set(holdings) | set(day_targets)}
            self._blocked_sells = {ticker for ticker, difference in requested.items() if difference < 0 and blocked_reason(
                side="sell", previous_close=previous[ticker], execution_close=px[ticker],
                is_halted=bool(self.halted[i, self.col_index[ticker]]),
                limit_move_abs_return=self.limit_move_abs_return)}
            self._blocked_buys = {ticker for ticker, difference in requested.items() if difference > 0 and blocked_reason(
                side="buy", previous_close=previous[ticker], execution_close=px[ticker],
                is_halted=bool(self.halted[i, self.col_index[ticker]]),
                limit_move_abs_return=self.limit_move_abs_return)}
            orders, _, scale = self._orders(holdings, day_targets, cash, px, forbidden_buys=forbidden)
            if i == first_i:
                self.rebalance_count += 1
                self.scaled_rebalance_count += scale < 1.0 - 1e-12
                # 체결 거부된 폐지 종목도 원래 체결일 목표의 회전율에 포함한다.
                turnover = 0.5 * (sum(abs(diff) for diff in requested.values())
                                  + sum(expired_targets.values()))
            fill = execute_orders(
                orders=orders, holdings=holdings, cash=cash,
                execution_closes=px, previous_closes=previous,
                halted_tickers=frozenset(ticker for ticker in px if self.halted[i, self.col_index[ticker]]),
                rates=self.rates, limit_move_abs_return=self.limit_move_abs_return,
            )
            holdings, cash = fill.holdings, fill.cash
            if i == first_i:
                pending_rebalance_sells = {
                    ticker: day_targets.get(ticker, 0.0) for ticker, difference in requested.items()
                    if difference < 0 and fill.rejected.get(ticker) == "halted" and ticker in holdings
                }
            else:
                for ticker in tuple(pending_rebalance_sells):
                    if ticker not in holdings or not self.halted[i, self.col_index[ticker]]:
                        pending_rebalance_sells.pop(ticker)
            for ticker in tuple(pending):
                if ticker not in holdings:
                    pending.remove(ticker)
            # 마지막 정리매매 종가로 비용을 내고 청산한다. 가격이 없다면
            # 0으로 평가하고 보유를 제거한다.
            delist_fills: list[Fill] = []
            delist_cost = 0.0
            for ticker, (delist_day, last_price) in self.delistings.items():
                if delist_day == day and ticker in holdings:
                    if last_price is not None:
                        notional = holdings[ticker] * last_price
                        cost = trade_cost(buy_notional=0, sell_notional=notional, rates=self.rates)
                        cash += notional - cost
                        delist_cost += cost
                        delist_fills.append(Fill(ticker, -holdings[ticker], last_price, notional, cost))
                    del holdings[ticker]
                    pending.discard(ticker)
            nav_after = portfolio_nav(holdings=holdings, closes=self._prices(i, holdings), cash=cash)
            nav_max = max(nav_max, nav_after)
            if i == next_i:
                weights_obs = ({ticker: qty * px[ticker] / nav_after for ticker, qty in holdings.items() if qty > 0}
                               if nav_after > 0 else {})
                cash_obs = cash / nav_after if nav_after > 0 else 1.0
            if self.record_ledger:
                skipped = {ticker: "cooldown_or_delisted" for ticker in set(day_targets) & forbidden
                           if requested.get(ticker, 0.0) > 0}
                if i == first_i:
                    skipped.update({ticker: "cooldown_or_delisted" for ticker, weight in expired_targets.items()
                                    if weight > 0})
                self.ledger.append({"date": day, "nav_before": nav_before, "nav_after": nav_after,
                                    "cash": cash, "holdings": dict(holdings), "cost": fill.total_cost + delist_cost,
                                    "fills": fill.fills + tuple(delist_fills), "rejected": {**fill.rejected, **skipped},
                                    "scale": scale})
        nav_end = portfolio_nav(holdings=holdings, closes=self._prices(end_i, holdings), cash=cash)
        if nav_start <= 0 or nav_end < 0:
            raise ValueError("시작 NAV는 양수이고 종료 NAV는 0 이상이어야 합니다")
        return PeriodResult(
            nav_start=nav_start, nav_end=nav_end, nav_max=nav_max,
            r_net=nav_end / nav_start - 1.0,
            r_bm=self.bm[end_i] / self.bm[first_i] - 1.0,
            turnover=turnover, weights_obs=weights_obs, cash_obs=cash_obs,
            state=EngineState(holdings, cash, self.days[end_i], frozenset(pending)),
        )
