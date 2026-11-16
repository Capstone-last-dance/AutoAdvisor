"""C-05: 고정된 A 시험 신호로 G1·G3을 독립 재계산한다.

모델 선택이나 임계값 조정은 하지 않는다. 모든 수치는 M5 공통 지표 함수를
사용하고, 라벨·진단 열이 빠진 행의 개수를 결과에 남긴다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.backtest import metrics as m5
from src.portfolio.params import load_prereg

KEY = ["date", "ticker"]
BASELINES = {"B1": "b1_score", "B2": "b2_score", "B3": "b3_score"}


def _keyed(frame: pd.DataFrame, name: str, required: set[str]) -> pd.DataFrame:
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{name} 필수 열이 없습니다: {sorted(missing)}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    if out["date"].isna().any() or out["ticker"].isna().any():
        raise ValueError(f"{name}의 date 또는 ticker가 결측입니다")
    out["ticker"] = out["ticker"].astype(str)
    if out.duplicated(KEY).any():
        raise ValueError(f"{name}에 중복 (date, ticker)가 있습니다")
    return out


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    numbers = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(numbers.to_numpy()).all(axis=1), index=frame.index)


def _scores(rows: pd.DataFrame, *, maxlags: int) -> dict[str, Any]:
    gate_ok = rows["y_crash"].isin([0, 1]) & _finite(rows, ["crash_prob", "base_prob"])
    rank_ok = _finite(rows, ["y_rank", "er", "rank_score", *BASELINES.values()])
    gate, rank = rows.loc[gate_ok], rows.loc[rank_ok]
    result: dict[str, Any] = {
        "signal_rows": len(rows),
        "gate_rows_used": len(gate),
        "gate_rows_missing": int((~gate_ok).sum()),
        "rank_rows_used": len(rank),
        "rank_rows_missing": int((~rank_ok).sum()),
        "auc": m5.roc_auc(gate["y_crash"], gate["crash_prob"]) if len(gate) else None,
        "baseline_auc": m5.roc_auc(gate["y_crash"], gate["base_prob"]) if len(gate) else None,
        "lift_at_10pct": m5.lift_at_10pct(gate["crash_flag"], gate["y_crash"]) if len(gate) else None,
    }
    rank_values = {}
    for name, column in {"model": "rank_score", **BASELINES}.items():
        ic = m5.rank_ic_with_hac(
            rank[["date", column, "er"]].rename(columns={column: "rank_score"}),
            maxlags=maxlags,
        ) if len(rank) else None
        rank_values[name] = {
            "mean_ic": ic.mean_ic if ic else None,
            "hac_t": ic.hac_t_statistic if ic else None,
            "signal_days_used": ic.signal_days_used if ic else 0,
            "signal_days_undefined": ic.signal_days_undefined if ic else 0,
        }
    result["rank_ic"] = rank_values
    return result


def evaluate_g1_g3(
    signals: pd.DataFrame,
    diagnostics: pd.DataFrame,
    labels: pd.DataFrame,
    prereg: Mapping[str, Any],
    *,
    test_years: Sequence[int] | None = None,
    horizon: int = 20,
) -> dict[str, Any]:
    """시험 연도별·전체 G1/G3 판정과 결측 건수를 반환한다.

    A의 ``wf_diagnostics.parquet``과 신호는 키가 정확히 같아야 한다. 각 지표의
    모델·기준선은 같은 유효 행에서 계산한다. 2026 홀드아웃은 이 함수의
    입력으로 허용하지 않는다.
    """
    if horizon != 20:
        raise ValueError("G1·G3 본 판정은 h=20만 사용합니다")
    years = tuple(int(y) for y in (test_years or prereg["splits"]["walk_forward_test_years"]))
    if not years or len(years) != len(set(years)) or max(years) > 2025:
        raise ValueError("시험 연도는 중복 없는 2025년 이하 연도여야 합니다")
    s = _keyed(signals, "signals", set(KEY) | {"crash_prob", "crash_flag", "rank_score", "model_version"})
    d = _keyed(diagnostics, "diagnostics", set(KEY) | {"test_year", "base_prob", *BASELINES.values()})
    if "date" not in labels or pd.to_datetime(labels["date"], errors="raise").ge(pd.Timestamp("2026-01-01")).any():
        raise ValueError("C-05는 2026 홀드아웃 라벨을 입력으로 읽지 않습니다")
    l = labels.loc[labels["h"].eq(horizon)].copy() if "h" in labels else labels
    l = _keyed(l, "labels", set(KEY) | {"y_crash", "y_rank", "er"})
    if s.empty or set(s["date"].dt.year) != set(years):
        raise ValueError("signals의 시험 연도가 사전등록 구간과 다릅니다")
    model_year = s["model_version"].str.extract(r"_y(\d{4})$")[0].astype("Int64")
    if model_year.isna().any() or not s["date"].dt.year.eq(model_year).all():
        raise ValueError("model_version의 폴드 연도가 신호일 연도와 다릅니다")
    diagnostic_year = pd.to_numeric(d["test_year"], errors="coerce")
    if diagnostic_year.isna().any() or not d["date"].dt.year.eq(diagnostic_year).all():
        raise ValueError("diagnostics의 test_year가 신호일 연도와 다릅니다")
    joined = s.merge(d[KEY + ["test_year", "base_prob", *BASELINES.values()]],
                     on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("signals와 diagnostics의 (date, ticker) 집합이 다릅니다")
    joined = joined.drop(columns=["_merge", "test_year"])
    joined = joined.merge(l[KEY + ["y_crash", "y_rank", "er"]], on=KEY, how="left", validate="one_to_one")
    if not joined["crash_flag"].isin([0, 1]).all():
        raise ValueError("crash_flag는 0/1이어야 합니다")
    prob = pd.to_numeric(joined["crash_prob"], errors="coerce")
    if not prob.dropna().between(0, 1).all():
        raise ValueError("crash_prob는 0~1이어야 합니다")
    maxlags = int(prereg["kpi"]["G3"]["nw_maxlags"])
    by_year = {str(y): _scores(joined.loc[joined["date"].dt.year.eq(y)], maxlags=maxlags) for y in years}
    overall = _scores(joined, maxlags=maxlags)
    g1, g3 = prereg["kpi"]["G1"], prereg["kpi"]["G3"]
    baseline_ics = [overall["rank_ic"][name]["mean_ic"] for name in BASELINES]
    best_ic = max(baseline_ics) if all(v is not None for v in baseline_ics) else None
    auc_gain = (overall["auc"] - overall["baseline_auc"]
                if overall["auc"] is not None and overall["baseline_auc"] is not None else None)
    model_ic = overall["rank_ic"]["model"]["mean_ic"]
    ic_gain = model_ic - best_ic if model_ic is not None and best_ic is not None else None
    g1_values = (auc_gain, overall["lift_at_10pct"])
    g3_values = (model_ic, overall["rank_ic"]["model"]["hac_t"], ic_gain)
    return {
        "test_years": list(years), "horizon": horizon,
        "overall": overall, "by_year": by_year,
        "G1": {"auc_gain": auc_gain, "lift_at_10pct": overall["lift_at_10pct"],
               "passed": (auc_gain >= float(g1["min_auc_gain_vs_baseline"]) and
                          overall["lift_at_10pct"] >= float(g1["min_lift_at_10pct"]))
               if all(v is not None for v in g1_values) else None},
        "G3": {"best_baseline_mean_ic": best_ic, "ic_gain": ic_gain,
               "passed": (model_ic >= float(g3["min_mean_ic"]) and
                          overall["rank_ic"]["model"]["hac_t"] >= float(g3["min_t_stat"]) and
                          ic_gain >= float(g3["min_ic_gain_vs_best_baseline"]))
               if all(v is not None for v in g3_values) else None},
    }


def evaluate_g7(applied_nav: pd.Series, without_m8_m9_nav: pd.Series,
                prereg: Mapping[str, Any], *, initial_nav: float) -> dict[str, Any]:
    """C-07: 같은 일별 날짜·시작 NAV의 G7 적용/미적용 경로를 비교한다.

    NAV는 같은 C 엔진과 비용 규칙으로 만든 체결 후 일별 값이다. 첫 체결일의
    체결 직전 NAV를 별도로 받아 첫날 비용을 MDD·CAGR에 반영한다. 이 함수는
    B의 비교용 positions를 생성하지 않으며, 판정 임계값을 재설정하지 않는다.
    """
    if (not isinstance(applied_nav, pd.Series) or not isinstance(without_m8_m9_nav, pd.Series)
            or not applied_nav.index.equals(without_m8_m9_nav.index)
            or len(applied_nav) < 2 or applied_nav.index.has_duplicates):
        raise ValueError("G7 NAV는 중복 없는 같은 일별 날짜를 2개 이상 가져야 합니다")
    days = pd.DatetimeIndex(pd.to_datetime(applied_nav.index))
    if days.hasnans or not days.is_monotonic_increasing:
        raise ValueError("G7 NAV 날짜는 중복 없는 오름차순이어야 합니다")
    on = pd.to_numeric(applied_nav, errors="coerce").to_numpy(dtype=float)
    off = pd.to_numeric(without_m8_m9_nav, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(on).all() or not np.isfinite(off).all() or (on <= 0).any() or (off <= 0).any():
        raise ValueError("G7 NAV는 유한한 양수여야 합니다")
    if not np.isfinite(initial_nav) or initial_nav <= 0:
        raise ValueError("G7 체결 직전 시작 NAV는 유한한 양수여야 합니다")
    count = len(days) - 1
    mdd_on = m5.max_drawdown(np.r_[initial_nav, on])
    mdd_off = m5.max_drawdown(np.r_[initial_nav, off])
    cagr_on = m5.cagr(np.r_[initial_nav, on[1:]], trading_days=count)
    cagr_off = m5.cagr(np.r_[initial_nav, off[1:]], trading_days=count)
    reduction = (mdd_off - mdd_on) / mdd_off if mdd_off > 0 else None
    threshold = prereg["kpi"]["G7"]
    return {
        "daily_return_count": count, "mdd_applied": mdd_on, "mdd_without_m8_m9": mdd_off,
        "mdd_reduction_relative": reduction, "cagr_applied": cagr_on,
        "cagr_without_m8_m9": cagr_off,
        "passed": (reduction >= float(threshold["min_mdd_reduction_rel"]) and
                   cagr_on >= cagr_off - float(threshold["max_cagr_giveup"]))
        if reduction is not None else None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="C-05 G1·G3 시험 연도 판정")
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, default=Path("configs/prereg.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True, help="새 reports/{run_id} 디렉터리")
    args = parser.parse_args(argv)
    report = evaluate_g1_g3(
        pd.read_parquet(args.signals), pd.read_parquet(args.diagnostics), pd.read_parquet(args.labels),
        load_prereg(args.prereg),
    )
    inputs = {name: path for name, path in (("signals", args.signals), ("diagnostics", args.diagnostics),
                                            ("labels", args.labels), ("prereg", args.prereg))}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    out = args.out_dir
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "inputs_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    shutil.copyfile(args.prereg, out / "config.yaml")
    (out / "git_commit.txt").write_text(f"{commit}\ndirty: {dirty}\n", encoding="utf-8")
    print(f"G1={report['G1']['passed']} G3={report['G3']['passed']} -> {out}")


if __name__ == "__main__":
    main()
