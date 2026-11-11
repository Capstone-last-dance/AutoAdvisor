"""DEC-22 개인 행동 프록시의 코호트와 G2의 신호일별 검정 통계."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from math import isfinite

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.backtest.accounting import blocked_reason, execute_orders, portfolio_nav, trade_cost
from src.backtest.engine import ExecutionEngine
from src.backtest.metrics import cagr, max_drawdown
from src.common.guards import DataAccessPolicy, assert_outcome_dates_allowed


@dataclass(frozen=True)
class CohortSelection:
    cohort: pd.DataFrame  # date, ticker, proxy_score (내림차순, 동률은 ticker 오름차순)
    missing_history: int
    zero_baseline: int


def select_retail_cohort(
    values: pd.DataFrame,
    eligible_tickers: Iterable[str],
    signal_date: str,
    *,
    top_n: int = 30,
) -> CohortSelection:
    """U_t에서 5일 평균 거래대금 / 직전 60일 평균 거래대금 상위 종목을 고른다.

    입력은 KRX 거래일 × 종목 거래대금(`prices.value`) 표다. 분자는 t−4~t, 분모는
    t−64~t−5이며, 후행 날짜를 읽지 않는다. 65일 중 결측이 있거나
    분모가 0인 종목은 순위를 만들지 않고 건수를 따로 돌려준다.
    """
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n은 양의 정수여야 합니다")
    if (values.empty or not isinstance(values.index, pd.DatetimeIndex)
            or values.index.has_duplicates or not values.index.is_monotonic_increasing):
        raise ValueError("거래대금 표는 중복 없는 오름차순 거래일이어야 합니다")
    if values.columns.has_duplicates:
        raise ValueError("거래대금 표의 종목코드는 고유해야 합니다")
    day = pd.Timestamp(signal_date)
    if day not in values.index:
        raise ValueError("신호일이 거래대금 표에 없습니다")
    end = values.index.get_loc(day)
    if end < 64:
        raise ValueError("신호일 이전 거래대금 이력 64일이 필요합니다")
    tickers = sorted(set(str(ticker) for ticker in eligible_tickers))
    if not tickers:
        raise ValueError("U_t가 비어 있습니다")
    history = values.reindex(columns=tickers).iloc[end - 64 : end + 1].apply(
        pd.to_numeric, errors="coerce"
    )
    if (history < 0).any().any():
        raise ValueError("거래대금은 음수일 수 없습니다")
    valid = history.notna().all(axis=0) & np.isfinite(history.to_numpy()).all(axis=0)
    baseline = history.iloc[:60].mean(axis=0)
    recent = history.iloc[60:].mean(axis=0)
    zero_baseline = valid & (baseline <= 0)
    scores = (recent / baseline).loc[valid & ~zero_baseline]
    ranked = (scores.rename("proxy_score").rename_axis("ticker").reset_index()
              .sort_values(["proxy_score", "ticker"], ascending=[False, True])
              .head(top_n).reset_index(drop=True))
    ranked.insert(0, "date", day)
    return CohortSelection(ranked, int((~valid).sum()), int(zero_baseline.sum()))


@dataclass(frozen=True)
class CohortEffect:
    daily_d: pd.Series  # 신호일별 mean(ER | flagged) - mean(ER | unflagged)
    mean_d: float | None
    hac_t_statistic: float | None
    signal_days_undefined: int
    missing_pairs: int
    signal_days_no_flags: int = 0
    signal_days_missing_group: int = 0


def cohort_effect_hac(
    cohorts: pd.DataFrame,
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    horizon: int = 20,
    maxlags: int = 4,
    policy: DataAccessPolicy,
    run_date: date,
    allow_holdout: bool = False,
) -> CohortEffect:
    """코호트의 G2 d_t와 상수항 OLS·HAC t값을 계산한다.

    C 홀드아웃 가드를 직접 적용한다. 결측 ER이나 crash_flag를 0으로
    채우지 않고 제외 건수에 기록한다.
    """
    if type(horizon) is not int or horizon < 1 or type(maxlags) is not int or maxlags < 0:
        raise ValueError("horizon은 양의 정수, maxlags는 0 이상의 정수여야 합니다")
    for name, frame, required in (
        ("cohorts", cohorts, {"date", "ticker"}),
        ("signals", signals, {"date", "ticker", "crash_flag"}),
        ("labels", labels, {"date", "ticker", "h", "er"}),
    ):
        if required - set(frame):
            raise ValueError(f"{name} 필수 열이 없습니다: {sorted(required - set(frame))}")
    assert_outcome_dates_allowed(
        pd.to_datetime(cohorts["date"]).dt.date,
        policy=policy, allow_holdout=allow_holdout, run_date=run_date,
    )
    selected = labels.loc[labels["h"] == horizon, ["date", "ticker", "er"]]
    for name, frame in (("cohorts", cohorts), ("signals", signals), ("labels", selected)):
        if frame.duplicated(["date", "ticker"]).any():
            raise ValueError(f"{name}에 중복 (date, ticker)가 있습니다")
    joined = (cohorts[["date", "ticker"]]
              .merge(signals[["date", "ticker", "crash_flag"]], on=["date", "ticker"], how="left", validate="one_to_one")
              .merge(selected, on=["date", "ticker"], how="left", validate="one_to_one"))
    joined["er"] = pd.to_numeric(joined["er"], errors="coerce")
    joined["crash_flag"] = pd.to_numeric(joined["crash_flag"], errors="coerce")
    if not joined["crash_flag"].dropna().isin([0, 1]).all():
        raise ValueError("crash_flag는 0/1이어야 합니다")
    valid = joined["crash_flag"].notna() & joined["er"].notna() & np.isfinite(joined["er"])
    missing_pairs = int((~valid).sum())
    values: dict[object, float] = {}
    undefined = 0
    no_flags = 0
    missing_group = 0
    for day, group in joined.groupby("date", sort=True):
        has_flagged = group["crash_flag"].eq(1).any()
        has_unknown_flag = group["crash_flag"].isna().any()
        group = group.loc[valid.loc[group.index]]
        flagged = group.loc[group["crash_flag"] == 1, "er"]
        unflagged = group.loc[group["crash_flag"] == 0, "er"]
        if flagged.empty or unflagged.empty:
            undefined += 1
            if not has_flagged and not has_unknown_flag:
                no_flags += 1
            else:
                missing_group += 1
            continue
        values[day] = float(flagged.mean() - unflagged.mean())
    daily_d = pd.Series(values, dtype="float64", name="d_t")
    if daily_d.empty:
        return CohortEffect(daily_d, None, None, undefined, missing_pairs, no_flags, missing_group)
    t_stat = None
    if len(daily_d) >= 2:
        fit = sm.OLS(daily_d.to_numpy(), np.ones((len(daily_d), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        candidate = float(fit.tvalues[0])
        if isfinite(candidate):
            t_stat = candidate
    return CohortEffect(daily_d, float(daily_d.mean()), t_stat, undefined, missing_pairs,
                        no_flags, missing_group)


@dataclass(frozen=True)
class ProxyRun:
    daily_nav: pd.Series  # 첫 코호트 체결일부터 종료일까지 체결 후 NAV
    cagr: float | None
    mdd: float
    ledger: pd.DataFrame  # date, nav_before, nav_after, cash, cost, rejected


@dataclass(frozen=True)
class G2Decision:
    mdd_reduction_relative: float | None
    cagr_gate_not_lower: bool | None
    hac_t_statistic: float | None
    passed: bool | None


def evaluate_g2(
    baseline: ProxyRun, guarded: ProxyRun, effect: CohortEffect,
    *, min_mdd_reduction_rel: float = 0.20, max_t_stat: float = -3.0,
) -> G2Decision:
    """사전등록 G2 세 조건을 같은 판정 기간에서 함께 검사한다."""
    if not baseline.daily_nav.index.equals(guarded.daily_nav.index):
        raise ValueError("게이트 전후 NAV 날짜가 다릅니다")
    reduction = ((baseline.mdd - guarded.mdd) / baseline.mdd
                 if baseline.mdd > 0 else None)
    cagr_ok = (guarded.cagr >= baseline.cagr
               if guarded.cagr is not None and baseline.cagr is not None else None)
    t_stat = effect.hac_t_statistic
    passed = (reduction >= min_mdd_reduction_rel and cagr_ok and t_stat <= max_t_stat
              if reduction is not None and cagr_ok is not None and t_stat is not None else None)
    return G2Decision(reduction, cagr_ok, t_stat, passed)


def simulate_retail_proxy(
    engine: ExecutionEngine,
    cohorts: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    end_date: str,
    gate: bool,
    holding_days: int = 20,
    cohort_fraction: float = 0.25,
    policy: DataAccessPolicy,
    run_date: date,
    allow_holdout: bool = False,
) -> ProxyRun:
    """겹치는 코호트를 20거래일 보유하며 공통 C 체결·비용 함수로 운용한다.

    각 코호트는 t+1 종가에 체결 직전 NAV의 1/4을 균등 매수하고
    t+1+20 종가에 매도한다. 청산일에 거래정지·하한가로 막힌 매도는
    다음 거래일에 다시 시도한다. 미체결 매수는 재시도하지 않는다.
    M8·M9를 적용하지 않는 G2 전용 실행이다.
    """
    if type(gate) is not bool or type(holding_days) is not int or holding_days < 1:
        raise ValueError("gate는 bool, holding_days는 양의 정수여야 합니다")
    if not isfinite(cohort_fraction) or not 0 < cohort_fraction <= 1:
        raise ValueError("코호트 투자 비율은 0 초과 1 이하여야 합니다")
    if engine.events or engine.cooldowns:
        raise ValueError("G2 프록시에는 M9 청산·쿨다운을 적용하지 않습니다")
    if not {"date", "ticker"}.issubset(cohorts) or not {"date", "ticker", "crash_flag"}.issubset(signals):
        raise ValueError("코호트·신호 필수 열이 없습니다")
    selected = cohorts[["date", "ticker"]].copy()
    selected["date"] = pd.to_datetime(selected["date"]).dt.strftime("%Y-%m-%d")
    assert_outcome_dates_allowed(
        [*selected["date"], end_date], policy=policy,
        allow_holdout=allow_holdout, run_date=run_date,
    )
    flags = signals[["date", "ticker", "crash_flag"]].copy()
    flags["date"] = pd.to_datetime(flags["date"]).dt.strftime("%Y-%m-%d")
    if selected.empty or selected.duplicated(["date", "ticker"]).any() or flags.duplicated(["date", "ticker"]).any():
        raise ValueError("코호트가 비어 있거나 종목·신호가 중복됐습니다")
    selected = selected.merge(flags, on=["date", "ticker"], how="left", validate="one_to_one")
    if not selected["crash_flag"].isin([0, 1]).all():
        raise ValueError("코호트의 crash_flag가 누락됐거나 0/1이 아닙니다")
    if end_date not in engine.day_index:
        raise ValueError("종료일이 거래일 목록에 없습니다")
    entries: dict[int, list[tuple[str, tuple[str, ...]]]] = {}
    for day, group in selected.groupby("date", sort=True):
        if day not in engine.day_index or engine.day_index[day] + 1 >= len(engine.days):
            raise ValueError(f"{day} 다음 체결일이 없습니다")
        names = tuple(group.loc[(group["crash_flag"] == 0) if gate else
                                pd.Series(True, index=group.index), "ticker"].astype(str))
        entries.setdefault(engine.day_index[day] + 1, []).append((day, names))
    first_i = min(entries)
    last_i = engine.day_index[end_date]
    if last_i <= first_i:
        raise ValueError("종료일은 첫 코호트 체결일 뒤여야 합니다")
    if max(entries) > last_i:
        raise ValueError("종료일 뒤에 체결되는 코호트가 포함돼 있습니다")

    lots: dict[tuple[str, str], float] = {}
    maturities: dict[str, int] = {}
    cash = 1.0
    navs: dict[str, float] = {}
    ledger: list[dict] = []
    for i in range(first_i, last_i + 1):
        day = engine.days[i]
        holdings: dict[str, float] = {}
        for (_, ticker), qty in lots.items():
            holdings[ticker] = holdings.get(ticker, 0.0) + qty
        buying = {ticker for _, names in entries.get(i, []) for ticker in names}
        prices = engine._prices(i, {ticker: 0.0 for ticker in set(holdings) | buying})
        nav_before = portfolio_nav(holdings=holdings, closes=prices, cash=cash)
        cost_today = 0.0
        rejected: dict[str, str] = {}

        # 폐지 청산은 일반 가격제한폭을 거치지 않는다 (M5와 동일).
        for ticker in tuple(holdings):
            if ticker not in engine.delistings or engine.delistings[ticker][0] != day:
                continue
            last_price = engine.delistings[ticker][1]
            if last_price is not None:
                notional = holdings[ticker] * last_price
                cost = trade_cost(buy_notional=0, sell_notional=notional, rates=engine.rates)
                cash += notional - cost
                cost_today += cost
            for key in tuple(lots):
                if key[1] == ticker:
                    del lots[key]
            del holdings[ticker]

        due = {ticker: 0.0 for (cohort_day, ticker) in lots
               if maturities[cohort_day] <= i}
        for (cohort_day, ticker), qty in lots.items():
            if maturities[cohort_day] <= i:
                due[ticker] += qty
        if due:
            previous = {ticker: engine._prices(i - 1, {ticker: 0.0})[ticker] for ticker in due}
            sold = execute_orders(
                orders={ticker: -qty for ticker, qty in due.items()}, holdings=holdings, cash=cash,
                execution_closes=prices, previous_closes=previous,
                halted_tickers=frozenset(ticker for ticker in due if engine.halted[i, engine.col_index[ticker]]),
                rates=engine.rates, limit_move_abs_return=engine.limit_move_abs_return,
            )
            holdings, cash = sold.holdings, sold.cash
            cost_today += sold.total_cost
            rejected.update(sold.rejected)
            for fill in sold.fills:
                for key in tuple(lots):
                    if key[1] == fill.ticker and maturities[key[0]] <= i:
                        del lots[key]

        for cohort_day, names in entries.get(i, []):
            maturities[cohort_day] = i + holding_days
            if not names:
                continue
            allocation = min(cohort_fraction * nav_before, cash)
            desired = {ticker: allocation / len(names) / prices[ticker] for ticker in names}
            previous = {
                ticker: (prices[ticker] if engine.relisting_index.get(ticker) == i
                         else engine._prices(i - 1, {ticker: 0.0})[ticker]) for ticker in names
            }
            allowed: dict[str, float] = {}
            for ticker, qty in desired.items():
                delisted = (ticker in engine.delistings
                            and engine.day_index[engine.delistings[ticker][0]] <= i
                            and i < engine.relisting_index.get(ticker, len(engine.days)))
                reason = "cooldown_or_delisted" if delisted else blocked_reason(
                    side="buy", previous_close=previous[ticker], execution_close=prices[ticker],
                    is_halted=bool(engine.halted[i, engine.col_index[ticker]]),
                    limit_move_abs_return=engine.limit_move_abs_return,
                )
                if reason:
                    rejected[ticker] = reason
                else:
                    allowed[ticker] = qty
            buy_rate = engine.rates.stress_multiplier * (engine.rates.commission_per_side + engine.rates.slippage_per_side)
            needed = sum(qty * prices[ticker] * (1 + buy_rate) for ticker, qty in allowed.items())
            scale = min(1.0, cash / needed) if needed > 0 else 1.0
            bought = execute_orders(
                orders={ticker: qty * scale for ticker, qty in allowed.items()},
                holdings=holdings, cash=cash, execution_closes=prices, previous_closes=previous,
                halted_tickers=frozenset(), rates=engine.rates,
                limit_move_abs_return=engine.limit_move_abs_return,
            )
            holdings, cash = bought.holdings, bought.cash
            cost_today += bought.total_cost
            for fill in bought.fills:
                lots[(cohort_day, fill.ticker)] = lots.get((cohort_day, fill.ticker), 0.0) + fill.signed_quantity
        nav_after = portfolio_nav(holdings=holdings, closes=engine._prices(i, holdings), cash=cash)
        navs[day] = nav_after
        ledger.append({"date": day, "nav_before": nav_before, "nav_after": nav_after,
                       "cash": cash, "cost": cost_today, "rejected": rejected})
    daily_nav = pd.Series(navs, name="nav", dtype="float64")
    return ProxyRun(daily_nav, cagr([1.0, daily_nav.iloc[-1]], trading_days=last_i - first_i),
                    max_drawdown([1.0, *daily_nav]), pd.DataFrame(ledger))
