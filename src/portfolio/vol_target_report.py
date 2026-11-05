"""작업 B-02 결과 리포트: risk_layer.vol_target_annual(Q-12)을 계산해 reports/{run_id}/에 남긴다 (R-10).

    python -m src.portfolio.vol_target_report --run-id 20261103_B_vol_target [--config configs/prereg.yaml]

reports/{run_id}/에 config.yaml(사본), git_commit.txt, metrics.json을 쓴다 (9.3절). 코스피 200은 io.py로만
읽는다 (R-18). 같은 run_id 폴더가 있으면 덮어쓰지 않고 멈춘다 (R-11).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.portfolio.params import PREREG_PATH, RiskLayerParams, load_prereg, vol_target_years
from src.portfolio.risk_layer import (
    _as_bm,
    compute_vol_target_annual,
    load_kospi200,
    market_state_table,
    sigma_series,
)

ROOT = Path(__file__).resolve().parents[2]
RUN_ID_RE = re.compile(r"^\d{8}_B_[A-Za-z0-9_\-]+$")  # 9.3절 run_id 형식 (역할 B)


def vol_target_metrics(bm_close: pd.Series, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """metrics.json의 계산 결과. 계산에 쓴 종가 구간의 지문(sha256)을 남겨 데이터가 바뀌면 알 수 있게 한다."""
    params = RiskLayerParams.from_prereg(cfg)
    first, last = vol_target_years(cfg)
    value = compute_vol_target_annual(bm_close, params, (first, last))

    bm = _as_bm(bm_close).loc[: pd.Timestamp(f"{last}-12-31")]
    sigma = sigma_series(bm, params).loc[pd.Timestamp(f"{first}-01-01") :]
    used = bm.iloc[-(len(sigma) + params.vol_lookback_days) :]  # 첫 sigma_t 창의 앞 종가 20개 포함
    bear = market_state_table(bm_close, params, mode="no_exposure")["bear"].dropna()
    return {
        "vol_target_annual": value,
        "years": [first, last],
        "n_sigma_days": len(sigma),
        "sigma_mean_by_year": {str(y): float(s.mean()) for y, s in sigma.groupby(sigma.index.year)},
        "kospi200_used": {
            "first_date": used.index[0].date().isoformat(),
            "last_date": used.index[-1].date().isoformat(),
            "n_closes": len(used),
            "sha256_float64_closes": hashlib.sha256(used.to_numpy().tobytes()).hexdigest(),
        },
        # bear_t의 창이 처음 차는 날. 그 전에는 rebound_t = 0인 날만 regime_mc가 정해진다 (D4)
        "bear_first_date": bear.index[0].date().isoformat(),
    }


def _git_commit() -> str:
    """HEAD 커밋 해시와, 커밋되지 않은 추적 파일 목록 (R-10)."""
    git = ["git", "-C", str(ROOT)]
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(
        [*git, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    lines = [head] + [f"uncommitted: {d[3:]}" for d in dirty]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="B-02 vol_target_annual(Q-12) 리포트")
    ap.add_argument("--run-id", required=True, help="{YYYYMMDD}_B_{짧은설명}")
    ap.add_argument("--config", default=str(PREREG_PATH), help="부록 1 설정 (기본 configs/prereg.yaml)")
    args = ap.parse_args(argv)
    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit(f"run_id는 '{{YYYYMMDD}}_B_{{짧은설명}}' 형식이어야 한다: {args.run_id!r}")

    out = ROOT / "reports" / args.run_id
    if out.exists():
        raise SystemExit(f"{out}가 이미 있다. 결과를 덮어쓰지 않는다 (R-11)")
    cfg_path = Path(args.config)
    metrics = {"run_id": args.run_id, "task": "B-02", **vol_target_metrics(load_kospi200(), load_prereg(cfg_path))}

    out.mkdir(parents=True)
    shutil.copyfile(cfg_path, out / "config.yaml")
    (out / "git_commit.txt").write_text(_git_commit(), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"vol_target_annual = {metrics['vol_target_annual']:.6f} → {out}")
    return out


if __name__ == "__main__":
    main()
