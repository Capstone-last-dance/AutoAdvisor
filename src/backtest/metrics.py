"""6.6절 M5 공통 지표 함수. KPI·보고서·대시보드는 이 계산만 사용한다."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _numeric(values: Sequence[float], name: str, *, nonnegative: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name}은 유한한 1차원 비어 있지 않은 수열이어야 합니다")
    if nonnegative and (array < 0).any():
        raise ValueError(f"{name}에 음수가 있습니다")
    return array


def cagr(nav: Sequence[float], *, trading_days: int, annualization: int = 252) -> float:
    """(종료 NAV / 시작 NAV)^(252 / 수익률 구간 수) - 1.

    trading_days는 NAV 관측 수가 아니라 연속 관측 사이의 일간 수익률 개수다.
    """

    values = _numeric(nav, "NAV", nonnegative=True)
    if values[0] <= 0 or trading_days < 1 or annualization < 1:
        raise ValueError("시작 NAV와 거래일 수·연율화 기준은 양수여야 합니다")
    return float((values[-1] / values[0]) ** (annualization / trading_days) - 1)


def annual_excess_return(
    portfolio_nav: Sequence[float],
    benchmark_nav: Sequence[float],
    *,
    trading_days: int,
    annualization: int = 252,
) -> float:
    """같은 기간 포트폴리오 CAGR에서 BM CAGR을 뺀다."""

    if len(portfolio_nav) != len(benchmark_nav):
        raise ValueError("포트폴리오와 BM의 날짜 수가 다릅니다")
    return cagr(
        portfolio_nav, trading_days=trading_days, annualization=annualization
    ) - cagr(benchmark_nav, trading_days=trading_days, annualization=annualization)


def weekly_active_returns(
    portfolio_nav_at_fills: Sequence[float], benchmark_close_at_fills: Sequence[float]
) -> np.ndarray:
    """연속한 체결일의 체결 직전 NAV와 BM 종가로 계산한 a_k.

    체결 후 NAV를 넣으면 해당 주 첫 체결 비용이 빠진다.
    """

    portfolio = _numeric(portfolio_nav_at_fills, "체결일 NAV", nonnegative=True)
    benchmark = _numeric(benchmark_close_at_fills, "체결일 BM", nonnegative=True)
    if len(portfolio) != len(benchmark) or len(portfolio) < 2:
        raise ValueError("체결일 NAV와 BM에는 같은 날짜가 2개 이상 필요합니다")
    if (portfolio[:-1] <= 0).any() or (benchmark <= 0).any():
        raise ValueError("수익률 계산 기준 가격은 양수여야 합니다")
    return np.diff(portfolio) / portfolio[:-1] - np.diff(benchmark) / benchmark[:-1]


def information_ratio(
    active_returns: Sequence[float], *, weeks_per_year: int = 52
) -> float | None:
    """mean(a) / std(a, ddof=1) × √52. 정의되지 않으면 None."""

    active = _numeric(active_returns, "주간 초과수익", nonnegative=False)
    if weeks_per_year < 1:
        raise ValueError("주간 연율화 기준은 양수여야 합니다")
    if len(active) < 2:
        return None
    deviation = float(active.std(ddof=1))
    if np.isclose(deviation, 0.0):
        return None
    return float(active.mean() / deviation * sqrt(weeks_per_year))


def validation_ir(active_returns: Sequence[float]) -> float:
    """B의 IRFunction용 어댑터. 정의되지 않은 IR은 비유한 값으로 표시한다.

    B-01 검증 콜백은 비유한 값을 체크포인트 선택에서 제외한다. 보고서용
    information_ratio는 미산출을 나타내도록 None을 유지한다.
    """

    value = information_ratio(active_returns)
    return float("nan") if value is None else value


def max_drawdown(daily_nav: Sequence[float]) -> float:
    """일별 NAV의 최대 고점 대비 낙폭. 손실 크기를 양수로 반환한다."""

    nav = _numeric(daily_nav, "일간 NAV", nonnegative=True)
    if nav[0] <= 0:
        raise ValueError("시작 NAV는 양수여야 합니다")
    peaks = np.maximum.accumulate(nav)
    return float(np.max(1.0 - nav / peaks))


def turnover(
    target_weights: Mapping[str, float], pretrade_weights: Mapping[str, float]
) -> float:
    """현금 제외 0.5 × Σ|목표 비중 - 체결 직전 비중|."""

    tickers = (set(target_weights) | set(pretrade_weights)) - {"CASH"}
    for weights in (target_weights, pretrade_weights):
        if any(not isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("비중은 유한한 0 이상이어야 합니다")
    return 0.5 * sum(
        abs(target_weights.get(ticker, 0.0) - pretrade_weights.get(ticker, 0.0))
        for ticker in tickers
    )


def annual_turnover(
    turnovers: Sequence[float], *, trading_days: int, annualization: int = 252
) -> float:
    """판정 기간 회전율 합 / 연수."""

    values = _numeric(turnovers, "회전율", nonnegative=True)
    if trading_days < 1 or annualization < 1:
        raise ValueError("거래일 수·연율화 기준은 양수여야 합니다")
    return float(values.sum() / (trading_days / annualization))


@dataclass(frozen=True)
class RankIcResult:
    mean_ic: float | None
    hac_t_statistic: float | None
    signal_days_used: int
    signal_days_undefined: int
    missing_pairs: int


def rank_ic_with_hac(frame: pd.DataFrame, *, maxlags: int = 4) -> RankIcResult:
    """신호일별 Spearman IC의 평균과 상수항 OLS의 HAC t값."""

    required = {"date", "rank_score", "er"}
    if required - set(frame):
        raise ValueError(f"rank IC 필수 열이 없습니다: {sorted(required - set(frame))}")
    if maxlags < 0:
        raise ValueError("HAC maxlags는 0 이상이어야 합니다")
    clean = frame[["date", "rank_score", "er"]].copy()
    if clean["date"].isna().any():
        raise ValueError("신호일에 결측이 있습니다")
    clean["rank_score"] = pd.to_numeric(clean["rank_score"], errors="coerce")
    clean["er"] = pd.to_numeric(clean["er"], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    missing_pairs = int(clean[["rank_score", "er"]].isna().any(axis=1).sum())
    values: list[float] = []
    undefined = 0
    for _, group in clean.groupby("date", sort=True):
        group = group.dropna(subset=["rank_score", "er"])
        if (
            len(group) < 2
            or group["rank_score"].nunique() < 2
            or group["er"].nunique() < 2
        ):
            undefined += 1
            continue
        ic = group["rank_score"].corr(group["er"], method="spearman")
        if pd.isna(ic):
            undefined += 1
            continue
        values.append(float(ic))
    if not values:
        return RankIcResult(None, None, 0, undefined, missing_pairs)
    t_stat = None
    if len(values) >= 2:
        fitted = sm.OLS(np.asarray(values), np.ones((len(values), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        candidate = float(fitted.tvalues[0])
        if isfinite(candidate):
            t_stat = candidate
    return RankIcResult(
        float(np.mean(values)), t_stat, len(values), undefined, missing_pairs
    )


def roc_auc(y_crash: Sequence[int], crash_prob: Sequence[float]) -> float | None:
    """모든 유효한 종목·신호일 쌍의 sklearn ROC-AUC."""

    from sklearn.metrics import roc_auc_score

    labels = _numeric(y_crash, "급락 라벨", nonnegative=True)
    scores = _numeric(crash_prob, "급락 확률", nonnegative=True)
    if len(labels) != len(scores) or not np.isin(labels, [0, 1]).all():
        raise ValueError("급락 라벨과 확률의 길이·값이 잘못됐습니다")
    if (scores > 1).any():
        raise ValueError("급락 확률은 0~1이어야 합니다")
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def lift_at_10pct(crash_flag: Sequence[int], y_crash: Sequence[int]) -> float | None:
    """플래그 집단 급락 비율 / 전체 급락 비율."""

    flags = _numeric(crash_flag, "급락 플래그", nonnegative=True)
    labels = _numeric(y_crash, "급락 라벨", nonnegative=True)
    if (
        len(flags) != len(labels)
        or not np.isin(flags, [0, 1]).all()
        or not np.isin(labels, [0, 1]).all()
    ):
        raise ValueError("급락 플래그와 라벨은 같은 길이의 0/1 값이어야 합니다")
    if flags.sum() == 0 or labels.mean() == 0:
        return None
    return float(labels[flags == 1].mean() / labels.mean())


def capture_ratio(portfolio_return: float, benchmark_return: float) -> float | None:
    """같은 스트레스 기간의 포트폴리오 수익률 / BM 수익률."""

    if not isfinite(portfolio_return) or not isfinite(benchmark_return):
        raise ValueError("기간 수익률은 유한해야 합니다")
    if np.isclose(benchmark_return, 0.0):
        return None
    return portfolio_return / benchmark_return


def rebound_capture(
    portfolio_daily_returns: Sequence[float],
    benchmark_daily_returns: Sequence[float],
    regime_mc: Sequence[int],
) -> float | None:
    """전일 국면이 1이었던 일간 수익률만 이어 붙인 누적수익률의 비율.

    호출자는 d-1→d 수익률에 d-1 종가 기준 regime_mc를 맞춰 전달한다.
    이 함수는 입력 시계열을 추가로 이동하지 않는다.
    """

    portfolio = _numeric(
        portfolio_daily_returns, "포트폴리오 일간 수익률", nonnegative=False
    )
    benchmark = _numeric(benchmark_daily_returns, "BM 일간 수익률", nonnegative=False)
    regime = _numeric(regime_mc, "regime_mc", nonnegative=True)
    if (
        not (len(portfolio) == len(benchmark) == len(regime))
        or not np.isin(regime, [0, 1]).all()
    ):
        raise ValueError("일간 수익률과 국면은 같은 길이여야 하며 국면은 0/1입니다")
    if (portfolio < -1).any() or (benchmark < -1).any():
        raise ValueError("일간 단순수익률은 -100%보다 작을 수 없습니다")
    selected = regime == 1
    if not selected.any():
        return None
    portfolio_compound = float(np.prod(1 + portfolio[selected]) - 1)
    benchmark_compound = float(np.prod(1 + benchmark[selected]) - 1)
    return capture_ratio(portfolio_compound, benchmark_compound)
