"""C 평가용 CD 91일물 현금 수익 보정.

엔진의 현금 0% 원장을 보존하고 B의 ``t7rl_report.cd91_adjust``와 같은
일별 수익률 규칙을 적용한다. 보정 NAV는 성과 보고용이며 체결을 재실행하지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def cash_fraction(ledger: Sequence[Mapping[str, object]]) -> pd.Series:
    """일별 마지막 장부의 체결 후 현금 / NAV를 반환한다."""
    if not ledger:
        raise ValueError("현금 비중을 계산할 원장이 없습니다")
    frame = pd.DataFrame(
        {"date": [row["date"] for row in ledger],
         "cash": [float(row["cash"]) for row in ledger],
         "nav": [float(row["nav_after"]) for row in ledger]}
    )
    frame["date"] = pd.to_datetime(frame["date"])
    last = frame.groupby("date", sort=False).last()
    if (not last.index.is_monotonic_increasing or last.index.has_duplicates
            or not np.isfinite(last[["cash", "nav"]].to_numpy()).all()
            or (last["nav"] <= 0).any() or (last["cash"] < 0).any()
            or (last["cash"] > last["nav"] + 1e-10).any()):
        raise ValueError("원장의 날짜·현금·NAV가 올바르지 않습니다")
    out = last["cash"] / last["nav"]
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    return out


def cd91_adjust(nav: pd.Series, cash_frac: pd.Series, monthly: pd.Series) -> pd.Series:
    """엔진 일간 수익에 전날 현금 비중 × CD 91일물 일간 수익을 더한다.

    ``monthly``는 FRED IR3TIB01KRM156N의 월별 연율을 소수로 나타낸
    Series다. 거래일 간 달력일 수를 쓰며, 금리는 전 거래일의 월을 따른다.
    첫 NAV는 체결일 기준값 그대로 유지한다.
    """
    if not isinstance(nav, pd.Series) or not isinstance(cash_frac, pd.Series):
        raise TypeError("NAV와 현금 비중은 Series여야 합니다")
    days = pd.DatetimeIndex(pd.to_datetime(nav.index)).as_unit("ns")
    weights = cash_frac.copy()
    weights.index = pd.DatetimeIndex(pd.to_datetime(weights.index)).as_unit("ns")
    if (len(days) < 2 or days.hasnans or not days.is_monotonic_increasing
            or days.has_duplicates or weights.index.has_duplicates
            or not weights.index.equals(days)):
        raise ValueError("NAV와 현금 비중은 같은 중복 없는 일별 날짜여야 합니다")
    values = nav.to_numpy(dtype=float)
    fractions = weights.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values <= 0).any()
            or not np.isfinite(fractions).all() or (fractions < 0).any()
            or (fractions > 1 + 1e-10).any()):
        raise ValueError("NAV는 양수, 현금 비중은 0~1의 유한값이어야 합니다")
    if not isinstance(monthly.index, pd.PeriodIndex) or monthly.index.freqstr != "M":
        raise ValueError("CD 금리는 월별 PeriodIndex가 필요합니다")
    cash_daily = np.zeros(len(days))
    for i in range(1, len(days)):
        month = days[i - 1].to_period("M")
        if month not in monthly.index:
            raise KeyError(f"CD 91일물 금리가 없는 달: {month}")
        rate = float(monthly.loc[month])
        if not np.isfinite(rate) or rate <= -1:
            raise ValueError(f"CD 91일물 금리가 올바르지 않습니다: {month}")
        cash_daily[i] = (1 + rate) ** ((days[i] - days[i - 1]).days / 365) - 1
    engine_daily = values[1:] / values[:-1] - 1
    daily = engine_daily + fractions[:-1] * cash_daily[1:]
    adjusted = np.r_[values[0], values[0] * np.cumprod(1 + daily)]
    return pd.Series(adjusted, index=days, name=nav.name)
