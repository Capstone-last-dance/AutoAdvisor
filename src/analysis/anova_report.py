"""Run the preregistered supplementary ANOVAs on pinned B walk-forward outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.analysis.anova import PERIOD_BOUNDS, fold_seed_ir_anova, period_risk_anova


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _period(week: pd.Timestamp) -> str:
    for name, (start, end) in PERIOD_BOUNDS.items():
        if pd.Timestamp(start) <= week <= pd.Timestamp(end):
            return name
    raise ValueError(f"사전등록 기간 밖 체결 주: {week.date()}")


def run(weekly_path: Path, fold_seed_path: Path, output_dir: Path) -> dict:
    weekly = pd.read_csv(weekly_path)
    required = {"portfolio", "week", "a_k"}
    if not required.issubset(weekly):
        raise ValueError(f"주간 장부 필수 열 누락: {sorted(required - set(weekly))}")
    selected = weekly.loc[weekly["portfolio"].isin(["buffer_full", "buffer_off"])].copy()
    selected["week"] = pd.to_datetime(selected["week"], errors="raise")
    selected["period"] = selected["week"].map(_period)
    selected["risk_layer_on"] = selected["portfolio"].eq("buffer_full")
    if selected.groupby("portfolio").size().to_dict() != {"buffer_full": 519, "buffer_off": 519}:
        raise ValueError("B 인계의 각 M8 실행에 519개 주간 관측이 필요합니다")
    paired = selected.pivot(index="week", columns="portfolio", values="a_k")
    if paired.isna().any().any() or len(paired) != 519:
        raise ValueError("M8 적용·미적용의 체결 주 집합이 다릅니다")
    table1 = period_risk_anova(selected[["week", "period", "risk_layer_on", "a_k"]])

    scores = pd.read_csv(fold_seed_path)
    table2 = fold_seed_ir_anova(scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    table1.to_csv(output_dir / "anova_period_risk.csv", index_label="effect")
    table2.to_csv(output_dir / "anova_fold_seed.csv", index_label="effect")
    selected.sort_values(["week", "risk_layer_on"]).to_csv(output_dir / "weekly_input.csv", index=False)
    summary = {
        "analysis": "M12 supplementary ANOVA; p-values are not KPI pass/fail criteria",
        "input_sha256": {str(weekly_path): _sha256(weekly_path), str(fold_seed_path): _sha256(fold_seed_path)},
        "anova_1": {"paired_weeks": len(paired), "rows": len(selected), "effects": table1.drop(index="Residual")[["F", "PR(>F)", "partial_eta_sq"]].to_dict(orient="index")},
        "anova_2": {"fold_seed_cells": len(scores), "effects": table2.drop(index="Residual")[["F", "PR(>F)", "partial_eta_sq"]].to_dict(orient="index")},
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly", type=Path, default=Path("reports/20261118_B_buffer_wf/weekly_excess.csv"))
    parser.add_argument("--fold-seed", type=Path, default=Path("reports/20261117_B_rl_wf/fold_seed_ir.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.weekly, args.fold_seed, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
