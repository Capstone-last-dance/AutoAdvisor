"""6.13절 M12 보조 ANOVA. 결과의 p값은 KPI 판정에 사용하지 않는다."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


PERIODS = tuple(f"P{i}" for i in range(1, 8))
SEEDS = tuple(range(5))
PERIOD_BOUNDS = {
    "P1": ("2016-01-01", "2017-12-31"),
    "P2": ("2018-01-01", "2019-12-31"),
    "P3": ("2020-01-01", "2020-12-31"),
    "P4": ("2021-01-01", "2022-12-31"),
    "P5": ("2023-01-01", "2023-11-05"),
    "P6": ("2023-11-06", "2025-03-30"),
    "P7": ("2025-03-31", "2025-12-31"),
}


def _finite_outcome(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or frame.isna().any().any():
        raise ValueError("ANOVA 입력에는 빈 표나 결측이 없어야 합니다")
    result = frame.copy()
    result[column] = pd.to_numeric(result[column], errors="raise")
    if not np.isfinite(result[column].to_numpy(dtype=float)).all():
        raise ValueError(f"{column}은 유한한 숫자여야 합니다")
    return result


def _anova(frame: pd.DataFrame, formula: str) -> pd.DataFrame:
    fitted = ols(formula, data=frame).fit()
    if fitted.df_resid <= 0:
        raise ValueError("잔차 자유도가 없어 ANOVA를 계산할 수 없습니다")
    result = anova_lm(fitted, typ=2)
    residual = float(result.loc["Residual", "sum_sq"])
    result["partial_eta_sq"] = result["sum_sq"] / (result["sum_sq"] + residual)
    result.loc["Residual", "partial_eta_sq"] = np.nan
    if not np.isfinite(result.loc[result.index != "Residual", "F"]).all():
        raise ValueError("F 통계량이 정의되지 않습니다. 입력 변동을 확인하세요")
    return result


def period_risk_anova(weekly: pd.DataFrame) -> pd.DataFrame:
    """ANOVA 1: 같은 체결 주의 a_k를 P1~P7 × M8 적용 여부로 분석.

    ``week``는 주간 a_k의 시작 체결일이다. 두 실행의 같은 주를 한 쌍으로
    넣고, period는 그 시작일이 속한 P1~P7로 지정한다.
    """
    required = {"week", "period", "risk_layer_on", "a_k"}
    if not required.issubset(weekly):
        raise ValueError(f"ANOVA 1 필수 열이 없습니다: {sorted(required - set(weekly))}")
    frame = _finite_outcome(weekly[list(required)], "a_k")
    frame["week"] = pd.to_datetime(frame["week"])
    if set(frame["period"]) != set(PERIODS):
        raise ValueError("ANOVA 1에는 홀드아웃을 제외한 P1~P7이 모두 필요합니다")
    for period, (start, end) in PERIOD_BOUNDS.items():
        dates = frame.loc[frame["period"] == period, "week"]
        if not dates.between(pd.Timestamp(start), pd.Timestamp(end)).all():
            raise ValueError(f"{period}에 범위 밖 체결일이 있습니다")
    if set(frame["risk_layer_on"]) != {False, True}:
        raise ValueError("M8 적용·미적용 실행이 모두 필요합니다")
    if frame.duplicated(["week", "risk_layer_on"]).any():
        raise ValueError("같은 주·M8 실행에 a_k가 중복됐습니다")
    paired = frame.pivot(index="week", columns="risk_layer_on", values="a_k")
    if paired.isna().any().any():
        raise ValueError("M8 적용·미적용 실행의 주간 표본이 일치해야 합니다")
    if frame.groupby("week")["period"].nunique().gt(1).any():
        raise ValueError("같은 주의 분석 구간이 두 실행에서 다릅니다")
    return _anova(frame, "a_k ~ C(period) * C(risk_layer_on)")


def fold_seed_ir_anova(scores: pd.DataFrame) -> pd.DataFrame:
    """ANOVA 2: 10폴드 × 5시드 IR의 폴드·시드 주효과(교호작용 없음)."""
    required = {"fold", "seed", "ir"}
    if not required.issubset(scores):
        raise ValueError(f"ANOVA 2 필수 열이 없습니다: {sorted(required - set(scores))}")
    frame = _finite_outcome(scores[list(required)], "ir")
    if frame["fold"].nunique() != 10 or set(frame["seed"]) != set(SEEDS):
        raise ValueError("사전등록된 10폴드 × 시드 0~4가 필요합니다")
    if len(frame) != 50 or frame.duplicated(["fold", "seed"]).any():
        raise ValueError("각 폴드·시드 조합에는 IR이 정확히 한 개 필요합니다")
    return _anova(frame, "ir ~ C(fold) + C(seed)")
