"""현금 수익 = 안정자산(CD 91일물) (11/20 사용자 결정: "현금은 안정자산을 수익으로 책정").

- 자료: `cd91_monthly.csv` = FRED IR3TIB01KRM156N (OECD MEI, 한국 3개월 금리, 월평균, 연 %). 11/20 B가 받았다.
  한국은 이 시계열이 CD 91일물이다: 같은 출처의 CD 시계열 IR3TCD01KRM156N(2023-12까지)과 겹치는 396개월이 모두 같다.
- 세전, 보수 없음. 이자는 달력일로 쌓인다: 거래일 k−1 → k의 현금 수익 = (1 + 연금리)^(달력일 수 / 365) − 1,
  연금리는 k−1이 속한 달의 값이다.
- C 엔진(M5)은 현금에 수익을 주지 않는다 (engine_adapter가 costs.cash_return = 0을 요구). 그래서 엔진 결과는
  일별 현금 비중으로 보정한다: 보정 일간 수익 = 엔진 일간 수익 + 전날 현금 비중 × 그날 현금 수익 (첨부 자료 식 ①).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CD91_PATH = HERE / "cd91_monthly.csv"


class CashYield:
    def __init__(self, path: Path = CD91_PATH) -> None:
        m = pd.read_csv(path)
        self.monthly = pd.Series(m["cd91_pct"].to_numpy(dtype=float) / 100.0, index=pd.PeriodIndex(m["month"], freq="M"))

    def rate(self, day: pd.Timestamp) -> float:
        p = pd.Timestamp(day).to_period("M")
        if p not in self.monthly.index:
            raise KeyError(f"CD 91일물 금리가 없는 달: {p}")
        return float(self.monthly[p])

    def daily(self, days: Sequence[Any]) -> np.ndarray:
        """거래일 배열 → r[k] = 거래일 k−1 → k의 현금 수익 (r[0] = 0)."""
        d = pd.DatetimeIndex(pd.to_datetime(list(days)))
        r = np.zeros(len(d))
        for k in range(1, len(d)):
            r[k] = (1.0 + self.rate(d[k - 1])) ** ((d[k] - d[k - 1]).days / 365.0) - 1.0
        return r

    def adjust(self, nav: pd.Series, cash_frac: pd.Series) -> pd.Series:
        """엔진 일별 NAV에 현금 수익을 더한다. cash_frac = 그날 마지막 기록 뒤 현금 ÷ NAV."""
        days = nav.index
        r_eng = nav.pct_change().fillna(0.0).to_numpy()
        c_prev = cash_frac.reindex(days).ffill().shift(1).fillna(0.0).to_numpy()
        r_adj = r_eng + c_prev * self.daily(days)
        out = float(nav.iloc[0]) * np.cumprod(1.0 + r_adj)
        return pd.Series(out, index=days)

    def mix(self, index_close: pd.Series, w: float) -> pd.Series:
        """지수 w + 현금(CD 91일물) 1 − w를 매일 같은 비중으로 맞춘 잔고 (첫 날 1, 비용 없음)."""
        r = index_close.pct_change().fillna(0.0).to_numpy()
        rc = self.daily(index_close.index)
        return pd.Series(np.cumprod(1.0 + w * r + (1.0 - w) * rc), index=index_close.index)


def cash_fraction(ledger: Sequence[Mapping[str, Any]]) -> pd.Series:
    """엔진 원장 → 날마다 마지막 기록의 현금 ÷ 체결 뒤 NAV (daily_nav와 같은 기록을 쓴다)."""
    f = pd.DataFrame({"date": [r["date"] for r in ledger], "cash": [float(r["cash"]) for r in ledger],
                      "nav": [float(r["nav_after"]) for r in ledger]})
    last = f.groupby("date", sort=False).last()
    out = (last["cash"] / last["nav"].where(last["nav"] > 0)).fillna(1.0)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).as_unit("ns")
    return out
