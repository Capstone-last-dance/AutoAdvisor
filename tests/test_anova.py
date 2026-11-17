"""M12의 사전등록 표본 단위와 교호작용 규칙."""

import pandas as pd
import pytest

from src.analysis.anova import fold_seed_ir_anova, period_risk_anova


def test_period_risk_anova_uses_paired_weeks_and_interaction():
    rows = []
    starts = ["2016-01", "2018-01", "2020-01", "2021-01", "2023-01", "2024-01", "2025-04"]
    for period, start in enumerate(starts, start=1):
        for week in range(4):
            for on in (False, True):
                rows.append({
                    "week": f"{start}-{week + 1:02d}",
                    "period": f"P{period}", "risk_layer_on": on,
                    "a_k": period * 0.001 + week * week * 0.0001 + on * period * 0.0002,
                })
    frame = pd.DataFrame(rows)
    result = period_risk_anova(frame)
    assert "C(period):C(risk_layer_on)" in result.index
    assert result.loc["C(period)", "F"] > 0
    assert 0 <= result.loc["C(period)", "partial_eta_sq"] <= 1
    with pytest.raises(ValueError, match="주간 표본"):
        period_risk_anova(frame.iloc[:-1])
    holdout = frame.copy()
    holdout.loc[0, "week"] = "2026-01-02"
    with pytest.raises(ValueError, match="범위 밖"):
        period_risk_anova(holdout)


def test_fold_seed_anova_has_no_interaction_and_rejects_missing_cell():
    frame = pd.DataFrame([
        {"fold": fold, "seed": seed,
         "ir": 0.1 * fold + 0.02 * seed + 0.001 * fold * seed}
        for fold in range(10) for seed in range(5)
    ])
    result = fold_seed_ir_anova(frame)
    assert "C(fold)" in result.index and "C(seed)" in result.index
    assert not any(":" in name for name in result.index)
    assert result.loc["Residual", "df"] == 36
    with pytest.raises(ValueError, match="정확히 한 개"):
        fold_seed_ir_anova(frame.iloc[:-1])
