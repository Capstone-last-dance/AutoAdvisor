"""C-05 판정은 신호·진단의 동일 키와 시험 연도 분리를 지켜야 한다."""

import pandas as pd
import pytest

from src.backtest.kpi_report import evaluate_g1_g3, evaluate_g7
from src.portfolio.params import load_prereg


def inputs():
    signals, diagnostics, labels = [], [], []
    for year in (2023, 2024):
        for week in range(3):
            day = pd.Timestamp(year=year, month=1, day=6 + 7 * week)
            for n in range(4):
                ticker = f"{n + 1:06d}"
                crash = int(n >= 2)
                signals.append({
                    "date": day, "ticker": ticker, "crash_prob": 0.1 + n * 0.25,
                    "crash_flag": crash, "rank_score": float(n + (week % 2) * (3 - 2 * n)),
                    "model_version": f"run_y{year}",
                })
                diagnostics.append({
                    "date": day, "ticker": ticker, "test_year": year,
                    "base_prob": 0.9 - n * 0.25,
                    "b1_score": float(3 - n), "b2_score": float((n + week) % 4),
                    "b3_score": float((n + 2 * week) % 4),
                })
                labels.append({
                    "date": day, "ticker": ticker, "h": 20, "y_crash": crash,
                    "y_rank": n + 1, "er": float(n * 0.02 + week * 0.001),
                })
    return pd.DataFrame(signals), pd.DataFrame(diagnostics), pd.DataFrame(labels)


def evaluate(s, d, l):
    return evaluate_g1_g3(s, d, l, load_prereg(), test_years=(2023, 2024))


def test_g1_g3_use_same_rows_for_models_baselines_and_report_missing_labels():
    s, d, l = inputs()
    l = l.loc[~((l["date"].dt.year == 2024) & (l["ticker"] == "000004"))]
    result = evaluate(s, d, l)
    assert list(result["by_year"]) == ["2023", "2024"]
    assert result["overall"]["signal_rows"] == 24
    assert result["overall"]["gate_rows_missing"] == 3
    assert result["overall"]["rank_rows_missing"] == 3
    assert result["overall"]["gate_rows_used"] == result["overall"]["rank_rows_used"] == 21
    assert result["overall"]["auc"] == pytest.approx(1.0)
    assert result["overall"]["baseline_auc"] == pytest.approx(0.0)
    assert result["G1"]["auc_gain"] == pytest.approx(1.0)
    assert result["G1"]["passed"] is True


def test_missing_diagnostics_and_future_test_year_fail_closed():
    s, d, l = inputs()
    with pytest.raises(ValueError, match="집합이 다릅니다"):
        evaluate(s, d.iloc[:-1], l)
    s.loc[0, "date"] = pd.Timestamp("2026-01-06")
    with pytest.raises(ValueError, match="시험 연도"):
        evaluate(s, d, l)
    s, d, l = inputs()
    future_label = l.iloc[[0]].copy()
    future_label["date"] = pd.Timestamp("2026-01-06")
    with pytest.raises(ValueError, match="홀드아웃 라벨"):
        evaluate(s, d, pd.concat([l, future_label]))


def test_rank_label_missing_keeps_gate_row_but_excludes_rank_pair():
    s, d, l = inputs()
    l.loc[0, "y_rank"] = pd.NA
    result = evaluate(s, d, l)
    assert result["overall"]["gate_rows_used"] == 24
    assert result["overall"]["rank_rows_used"] == 23


def test_wrong_fold_model_and_duplicate_keys_fail_closed():
    s, d, l = inputs()
    s.loc[0, "model_version"] = "run_y2024"
    with pytest.raises(ValueError, match="폴드 연도"):
        evaluate(s, d, l)
    s, d, l = inputs()
    with pytest.raises(ValueError, match="중복"):
        evaluate(pd.concat([s, s.iloc[[0]]]), d, l)


def test_g7_uses_same_daily_interval_and_returns_undefined_when_no_baseline_drawdown():
    days = pd.bdate_range("2023-01-02", periods=4)
    on = pd.Series([1.0, 1.1, 1.0, 1.12], index=days)
    off = pd.Series([1.0, 1.1, 0.8, 1.15], index=days)
    result = evaluate_g7(on, off, load_prereg(), initial_nav=1.0)
    assert result["daily_return_count"] == 3
    assert result["mdd_applied"] == pytest.approx(1 - 1.0 / 1.1)
    assert result["mdd_without_m8_m9"] == pytest.approx(1 - 0.8 / 1.1)
    assert result["mdd_reduction_relative"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="같은 일별 날짜"):
        evaluate_g7(on, off.iloc[1:], load_prereg(), initial_nav=1.0)
    flat = pd.Series([1.0, 1.1, 1.2, 1.3], index=days)
    assert evaluate_g7(flat, flat, load_prereg(), initial_nav=1.0)["passed"] is None


def test_g7_uses_post_fill_nav_for_mdd_and_pre_fill_nav_for_cagr():
    days = pd.bdate_range("2023-01-02", periods=3)
    on = pd.Series([0.99, 1.02, 0.90], index=days)
    off = pd.Series([0.98, 1.01, 0.95], index=days)
    result = evaluate_g7(on, off, load_prereg(), initial_nav=1.0)
    assert result["cagr_applied"] == pytest.approx((0.90 / 1.0) ** 126 - 1)
    assert result["mdd_applied"] == pytest.approx(1 - 0.90 / 1.02)
    rising_after_fill = pd.Series([0.98, 0.99, 1.01], index=days)
    assert evaluate_g7(rising_after_fill, rising_after_fill, load_prereg(), initial_nav=1.0)[
        "mdd_applied"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="체결 직전 시작 NAV"):
        evaluate_g7(on, off, load_prereg(), initial_nav=0.0)
