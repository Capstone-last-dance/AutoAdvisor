"""DEC-22 개인 행동 프록시의 코호트와 G2의 신호일별 검정 통계."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class CohortSelection:
    cohort: pd.DataFrame  # date, ticker, proxy_score (내림차순, 동률은 ticker 오름차순)
    missing_history: int
    zero_baseline: int


def select_retail_cohort(
    volumes: pd.DataFrame,
    eligible_tickers: Iterable[str],
    signal_date: str,
    *,
    top_n: int = 30,
) -> CohortSelection:
    """U_t에서 5일 평균 거래량 / 직전 60일 평균 거래량 상위 종목을 고른다.

    입력은 KRX 거래일 × 종목 거래량 표다. 분자는 t−4~t, 분모는
    t−64~t−5이며, 후행 날짜를 읽지 않는다. 65일 중 결측이 있거나
    분모가 0인 종목은 순위를 만들지 않고 건수를 따로 돌려준다.
    """
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n은 양의 정수여야 합니다")
    if (volumes.empty or not isinstance(volumes.index, pd.DatetimeIndex)
            or volumes.index.has_duplicates or not volumes.index.is_monotonic_increasing):
        raise ValueError("거래량 표는 중복 없는 오름차순 거래일이어야 합니다")
    if volumes.columns.has_duplicates:
        raise ValueError("거래량 표의 종목코드는 고유해야 합니다")
    day = pd.Timestamp(signal_date)
    if day not in volumes.index:
        raise ValueError("신호일이 거래량 표에 없습니다")
    end = volumes.index.get_loc(day)
    if end < 64:
        raise ValueError("신호일 이전 거래량 이력 64일이 필요합니다")
    tickers = sorted(set(str(ticker) for ticker in eligible_tickers))
    if not tickers:
        raise ValueError("U_t가 비어 있습니다")
    history = volumes.reindex(columns=tickers).iloc[end - 64 : end + 1].apply(
        pd.to_numeric, errors="coerce"
    )
    if (history < 0).any().any():
        raise ValueError("거래량은 음수일 수 없습니다")
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


def cohort_effect_hac(
    cohorts: pd.DataFrame,
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    horizon: int = 20,
    maxlags: int = 4,
) -> CohortEffect:
    """코호트의 G2 d_t와 상수항 OLS·HAC t값을 계산한다.

    호출자는 C 홀드아웃 가드를 통과한 라벨만 전달해야 한다. 결측 ER이나
    crash_flag를 0으로 채우지 않고 제외 건수에 기록한다.
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
    for day, group in joined.groupby("date", sort=True):
        group = group.loc[valid.loc[group.index]]
        flagged = group.loc[group["crash_flag"] == 1, "er"]
        unflagged = group.loc[group["crash_flag"] == 0, "er"]
        if flagged.empty or unflagged.empty:
            undefined += 1
            continue
        values[day] = float(flagged.mean() - unflagged.mean())
    daily_d = pd.Series(values, dtype="float64", name="d_t")
    if daily_d.empty:
        return CohortEffect(daily_d, None, None, undefined, missing_pairs)
    t_stat = None
    if len(daily_d) >= 2:
        fit = sm.OLS(daily_d.to_numpy(), np.ones((len(daily_d), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        candidate = float(fit.tvalues[0])
        if isfinite(candidate):
            t_stat = candidate
    return CohortEffect(daily_d, float(daily_d.mean()), t_stat, undefined, missing_pairs)
