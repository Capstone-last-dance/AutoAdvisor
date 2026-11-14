"""작업 B-07: 10폴드 RL 정책의 시험 연도 운용 (walk-forward 4.7절, 6.5절 테스트 행동). R-10 기록을 남긴다.

    python -m src.portfolio.rl_wf_report --run-id 20261112_B_rl_wf --fold-runs 20261111_B_rlwf_y{Y} \
        --test-signals-run-id 20261110_A_wfpm

- 폴드 결과: --fold-runs의 {Y}를 시험 연도로 바꾼 reports/{run_id}/ 10개 (rl_fold_report, task B-07).
  폴드마다 완료 기준 통과, 사전등록 시드 전부, 시험 연도가 맞는지 확인한다. 설정 사본·라벨 종료일·M9 경보·
  신호 run_id 앞부분은 폴드끼리, 그리고 지금 설정과 같아야 한다. 하나라도 다르면 멈춘다.
  폴드 실행 커밋이 서로 다르면 멈추지 않고 완료 기준에 적는다.
- 체크포인트: 폴드 폴더의 seed_{s}/best.zip (팀 드라이브에서 받은 파일). 폴드 metrics.json의 `policy_sha256`과
  다르면 다른 가중치이므로 멈춘다. 파일 `sha256`만 다르면(다시 저장한 파일) 기록만 한다.
- 시험 신호: A의 walk-forward 신호 `signals/wf_{--test-signals-run-id}.parquet`. 시험 연도 Y의 행은 폴드 Y 모델의
  예측이다. 신호일이 시험 연도(splits.walk_forward_test_years)의 신호일과 같아야 한다.
- M9: 폴드 학습과 같은 백테스트 경보(폴드 metrics.json의 m9)를 쓴다. 폴드가 M9 없이 학습됐으면 시험도 M9 없이
  운용하고 '설계 그대로' 기준을 통과하지 않은 것으로 적는다.
- 운용: 시드마다 첫 시험 신호일에 현금 100%로 시작해 결정적 행동으로 운용한다. 신호일 t의 정책은 t가 속한 해의
  폴드 정책이고, 연도 경계에서는 보유를 유지한 채 정책만 바꾼다 (rl_train.rollout). 마지막 시험 신호일은
  다음 신호일이 홀드아웃이라 비중만 기록하고 체결하지 않는다.
- 출력: positions/wf_{run_id}_s{seed}.parquet (9.3절, constructor = rl). reports/{run_id}/에 config.yaml,
  git_commit.txt, metrics.json, fold_seed_ir.csv (ANOVA 2 입력: fold, seed, ir), weekly_excess.csv
  (seed, fold, signal_date, week = 체결 시작일, a_k). IR은 C의 validation_ir를 시험 주간 초과수익에 쓴 B의 확인값이다.
  G5 판정 지표는 C-07이 positions로 계산한다.
- 같은 run_id의 결과 폴더나 positions 파일이 하나라도 있으면 덮어쓰지 않고 멈춘다 (R-11).
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.metrics import validation_ir
from src.common import io
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.env_check_report import (
    FIRST_SIGNAL_DATE,
    LAST_SIGNAL_DATE,
    TABLES,
    content_sha256,
    feature_table,
    input_fingerprints,
    kospi200,
    members_by_date,
    signal_dates,
)
from src.portfolio.params import PREREG_PATH, RiskLayerParams, RLParams, load_prereg
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_env import PortfolioEnv
from src.portfolio.rl_fold_report import (
    CHECKPOINT,
    ENGINE_TABLES,
    _file_sha256,
    _iso,
    _versions,
    m9_schedule,
    policy_sha256,
)
from src.portfolio.rl_train import rollout
from src.portfolio.vol_target_report import ROOT, RUN_ID_RE, _git_commit

FOLD_PLACEHOLDER = "{Y}"


# ---------------------------------------------------------------------------
# 폴드 결과
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldRun:
    year: int
    run_id: str
    folder: Path
    metrics: dict[str, Any]
    config: bytes
    commit: str  # git_commit.txt 첫 줄


def fold_run_ids(pattern: str, years: Sequence[int]) -> dict[int, str]:
    if FOLD_PLACEHOLDER not in pattern:
        raise SystemExit(f"--fold-runs에는 시험 연도 자리 {FOLD_PLACEHOLDER}가 있어야 한다: {pattern!r}")
    ids = {int(y): pattern.replace(FOLD_PLACEHOLDER, str(y)) for y in years}
    bad = [r for r in ids.values() if not RUN_ID_RE.match(r)]
    if bad:
        raise SystemExit(f"폴드 run_id 형식이 아니다: {bad[:3]}")
    return ids


def load_fold_runs(root: Path, ids: Mapping[int, str]) -> list[FoldRun]:
    runs = []
    for year, run_id in sorted(ids.items()):
        folder = root / "reports" / run_id
        if not (folder / "metrics.json").exists():
            raise SystemExit(f"폴드 {year}의 결과가 없다: {folder}/metrics.json (rl_fold_report를 먼저 돌린다)")
        runs.append(FoldRun(
            year, run_id, folder, json.loads((folder / "metrics.json").read_text(encoding="utf-8")),
            (folder / "config.yaml").read_bytes(),
            (folder / "git_commit.txt").read_text(encoding="utf-8").splitlines()[0].strip(),
        ))
    return runs


def _same(values: Sequence[Any], what: str) -> Any:
    first = values[0]
    if any(v != first for v in values[1:]):
        raise ValueError(f"폴드마다 {what}이(가) 다르다: {values}")
    return first


def check_fold_runs(runs: Sequence[FoldRun], rl: RLParams, config: bytes) -> dict[str, Any]:
    """폴드 결과가 같은 B-07 설계로 끝까지 돌았는지 본다. 구조가 다르면 멈추고, 실행 커밋 차이는 기록한다."""
    for r in runs:
        m = r.metrics
        if m.get("task") != "B-07":
            raise ValueError(f"{r.run_id}: task = {m.get('task')!r}. B-07 폴드 학습(--signals-run-id)이 아니다")
        if int(m["fold"]["test_year"]) != r.year:
            raise ValueError(f"{r.run_id}: 시험 연도가 {m['fold']['test_year']}다 ({r.year} 폴드 자리)")
        if not m["criteria"]["lambda_selection_ran_to_the_end"]:
            raise ValueError(f"{r.run_id}: λ 선택 완료 기준을 통과하지 않았다")
        if [int(s) for s in m["main_training"]["seeds"]] != list(rl.seeds):
            raise ValueError(f"{r.run_id}: 본 학습 시드 {m['main_training']['seeds']}가 사전등록 시드 {list(rl.seeds)}가 아니다")
        if r.config != config:
            raise ValueError(f"{r.run_id}: config.yaml이 지금 설정과 다르다")
        expected = f"_y{r.year}"
        if not str(m["signals"]["run_id"]).endswith(expected):
            raise ValueError(f"{r.run_id}: 신호 run_id {m['signals']['run_id']}가 폴드 {r.year} 파일이 아니다")
    signal_base = _same([str(r.metrics["signals"]["run_id"])[: -len(f"_y{r.year}")] for r in runs], "학습 신호 run_id")
    labels = _same([r.metrics["label_end_dates"] for r in runs], "라벨 종료일 파일")
    m9 = _same([r.metrics["m9"] for r in runs], "M9 경보")
    commits = sorted({r.commit for r in runs})
    return {
        "signals_run_id": signal_base, "label_end_dates": labels, "m9": m9, "commits": commits,
        "folds": [
            {"fold": r.year, "run_id": r.run_id, "commit": r.commit,
             "lambda": r.metrics["lambda_search"]["chosen_lambda"],
             "delay_purged": r.metrics["fold"]["purge"].get("delay_purged"),
             "best_val_ir": {str(s["seed"]): s["best_val_ir"] for s in r.metrics["main_training"]["per_seed"]}}
            for r in runs
        ],
    }


def load_policies(runs: Sequence[FoldRun], seeds: Sequence[int]) -> tuple[dict[tuple[int, int], Any], list[dict]]:
    """(폴드, 시드) → PPO 정책. 체크포인트의 정책 파라미터 지문이 폴드 기록과 같아야 한다."""
    from stable_baselines3 import PPO

    from src.portfolio.rl_train import DEVICE

    policies, record = {}, []
    for r in runs:
        files = r.metrics["checkpoints"]["files"]
        for seed in seeds:
            rel = f"seed_{seed}/{CHECKPOINT}"
            path = r.folder / rel
            if not path.exists():
                raise SystemExit(f"{path}가 없다. 팀 드라이브의 {r.run_id}/{rel}을 받아 같은 자리에 둔다")
            want = files[rel]
            got_policy, got_file = policy_sha256(path), _file_sha256(path)
            if got_policy != want["policy_sha256"]:
                raise ValueError(f"{path}: 정책 파라미터 지문이 {r.run_id} 기록과 다르다 (다른 가중치)")
            policies[(r.year, int(seed))] = PPO.load(path, device=DEVICE)
            record.append({"fold": r.year, "seed": int(seed), "path": f"reports/{r.run_id}/{rel}",
                           "policy_sha256": got_policy, "file_matches_record": got_file == want["sha256"]})
    return policies, record


# ---------------------------------------------------------------------------
# 시험 연도 운용
# ---------------------------------------------------------------------------


def test_signal_table(run_id: str, test_dates: Sequence[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    signals = io.read_table("signals", run_id=run_id)
    got, want = set(_iso(pd.unique(signals["date"]))), set(test_dates)
    if got != want:
        missing, extra = sorted(want - got), sorted(got - want)
        raise ValueError(f"signals/wf_{run_id}: 시험 신호일과 다르다 (없는 신호일 {len(missing)}개 {missing[:3]}, "
                         f"남는 신호일 {len(extra)}개 {extra[:3]})")
    year = pd.to_datetime(signals["date"]).dt.year
    versions = signals.groupby([year, signals["model_version"].astype(str)]).size()
    path = io.ROOT / "signals" / f"wf_{run_id}.parquet"
    return signals, {
        "run_id": run_id, "path": path.relative_to(io.ROOT).as_posix(), "rows": len(signals),
        "sha256": _file_sha256(path),
        "model_version": {str(y): {v: int(n) for v, n in versions[y].items()} for y in sorted(set(year))},
    }


def _next_trading_day(days: pd.DatetimeIndex, date: str) -> str:
    i = int(days.searchsorted(pd.Timestamp(date), side="right"))
    return days[i].strftime("%Y-%m-%d")


def weekly_table(seed: int, executed_dates: Sequence[str], excess: np.ndarray, days: pd.DatetimeIndex) -> pd.DataFrame:
    """체결된 주의 a_k. fold는 그 주의 정책을 고른 신호일의 해, week는 체결 시작일(신호일 다음 거래일)이다."""
    return pd.DataFrame({
        "seed": seed,
        "fold": [int(d[:4]) for d in executed_dates],
        "signal_date": list(executed_dates),
        "week": [_next_trading_day(days, d) for d in executed_dates],
        "a_k": np.asarray(excess, dtype=np.float64),
    })


def fold_seed_ir(weekly: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """ANOVA 2 입력 (fold, seed, ir). 폴드마다 그 해 신호일로 고른 주의 a_k로 IR을 계산한다."""
    rows = []
    for (seed, fold), g in weekly.groupby(["seed", "fold"], sort=True):
        rows.append({"fold": int(fold), "seed": int(seed), "ir": float(validation_ir(g["a_k"].to_numpy()))})
    out = pd.DataFrame(rows, columns=["fold", "seed", "ir"])
    missing = sorted(set(int(y) for y in years) - set(out["fold"]))
    if missing:
        raise ValueError(f"체결된 주가 없는 시험 연도: {missing}")
    return out


def main(argv: Sequence[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="B-07 10폴드 RL 정책의 시험 연도 운용 (walk-forward)")
    ap.add_argument("--run-id", required=True, help="{YYYYMMDD}_B_{짧은설명}")
    ap.add_argument("--fold-runs", required=True, help="폴드 학습 run_id, 시험 연도 자리는 {Y} (예: 20261111_B_rlwf_y{Y})")
    ap.add_argument("--test-signals-run-id", required=True, help="A walk-forward 시험 신호 run_id (signals/wf_{값}.parquet)")
    ap.add_argument("--config", default=str(PREREG_PATH), help="부록 1 설정 (기본 configs/prereg.yaml)")
    args = ap.parse_args(argv)
    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit(f"run_id는 '{{YYYYMMDD}}_B_{{짧은설명}}' 형식이어야 한다: {args.run_id!r}")
    cfg_path = Path(args.config)
    cfg = load_prereg(cfg_path)
    rl, risk = RLParams.from_prereg(cfg), RiskLayerParams.from_prereg(cfg)
    years = [int(y) for y in cfg["splits"]["walk_forward_test_years"]]
    out = ROOT / "reports" / args.run_id
    position_ids = {int(s): f"{args.run_id}_s{s}" for s in rl.seeds}
    existing = [out] if out.exists() else []
    existing += [p for p in (io.ROOT / "positions" / f"wf_{r}.parquet" for r in position_ids.values()) if p.exists()]
    if existing:
        raise SystemExit(f"{existing[0]}가 이미 있다. 결과를 덮어쓰지 않는다 (R-11)")

    start = perf_counter()
    runs = load_fold_runs(ROOT, fold_run_ids(args.fold_runs, years))
    folds = check_fold_runs(runs, rl, cfg_path.read_bytes())
    policies, checkpoints = load_policies(runs, rl.seeds)

    tables = {name: io.read_table(name) for name in TABLES}
    bm = kospi200(tables["index"])
    dates = signal_dates(tables["universe"], bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)
    test_dates = [d for d in dates if int(d[:4]) in years]
    signals, signal_record = test_signal_table(args.test_signals_run_id, test_dates)
    members = members_by_date(tables["universe"], test_dates)
    features = feature_table(tables["prices"], bm, members, risk)
    inputs = engine_inputs(*(tables[n] for n in ENGINE_TABLES))
    m9 = m9_schedule(folds["m9"].get("alerts_run_id") if folds["m9"]["applied"] else None, dates[0], dates[-1])
    keys = ("applied", "alerts_run_id", "sha256")
    if [m9.record.get(k) for k in keys] != [folds["m9"].get(k) for k in keys]:
        raise ValueError(f"M9 경보 파일이 폴드 학습 때와 다르다: 지금 {m9.record}, 폴드 {folds['m9']}")
    snaps = build_snapshots(test_dates, signals, features, tables["universe"], tables["sector"], inputs.bm_close,
                            m9.cooldown_at, risk)
    prepare_seconds = perf_counter() - start

    commit = _git_commit()  # 결과 폴더·positions를 만들기 전에 (R-10)
    out.mkdir(parents=True)
    shutil.copyfile(cfg_path, out / "config.yaml")
    (out / "git_commit.txt").write_text(commit, encoding="utf-8")

    start = perf_counter()
    weekly, positions = [], {}
    for seed in rl.seeds:
        engine = build_engine(inputs, cfg, cooldowns=m9.cooldowns, event_liquidations=m9.liquidations)
        env = PortfolioEnv(snaps, engine, rl, risk, 0.0, "full", calendar=dates)
        run = rollout(env, lambda d, s=int(seed): policies[(int(d[:4]), s)])
        path = io.write_table("positions", run.positions, run_id=position_ids[int(seed)])
        positions[str(seed)] = {"path": path.relative_to(io.ROOT).as_posix(), "rows": len(run.positions),
                                "signal_dates": int(run.positions["date"].nunique()), "sha256": _file_sha256(path)}
        weekly.append(weekly_table(int(seed), run.executed_dates, run.excess_returns, bm.index))
    rollout_seconds = perf_counter() - start
    weekly_df = pd.concat(weekly, ignore_index=True)
    weekly_df.to_csv(out / "weekly_excess.csv", index=False)
    scores = fold_seed_ir(weekly_df, years)
    scores.to_csv(out / "fold_seed_ir.csv", index=False)
    per_seed = {str(s): float(validation_ir(g["a_k"].to_numpy())) for s, g in weekly_df.groupby("seed")}

    criteria = {
        "every_fold_and_seed_trained": len(runs) == len(years) and len(checkpoints) == len(years) * len(rl.seeds),
        "fold_runs_same_commit": len(folds["commits"]) == 1,
        "m9_applied": bool(folds["m9"]["applied"]),
        "label_end_dates_applied": folds["label_end_dates"] is not None,
        "every_test_date_in_positions": all(p["signal_dates"] == len(test_dates) for p in positions.values()),
    }
    criteria["b07_complete"] = all(criteria.values())  # 10폴드 × 5시드 완료, 설계 변경 없음
    metrics = {
        "run_id": args.run_id,
        "task": "B-07",
        "criteria": criteria,
        "fold_runs": folds,
        "checkpoints": checkpoints,
        "test_signals": signal_record,
        "m9": m9.record,
        "test": {"years": years, "signal_dates": len(test_dates), "first": test_dates[0], "last": test_dates[-1],
                 "executed_weeks_per_seed": int(len(weekly_df) / len(rl.seeds)),
                 "start": "첫 시험 신호일에 현금 100%, 연도 경계에서 보유 유지·정책 교체"},
        "positions": positions,
        "ir": {"function": "src.backtest.metrics.validation_ir (B 확인값, 판정은 C-07)",
               "per_seed": per_seed, "mean_over_seeds": float(np.mean(list(per_seed.values()))),
               "fold_seed": "fold_seed_ir.csv"},
        "universe_size": {"mean": float(np.mean([len(s.tickers) for s in snaps])),
                          "min": int(min(len(s.tickers) for s in snaps)), "max": int(max(len(s.tickers) for s in snaps))},
        "inputs": input_fingerprints(tables, io.ROOT),
        "generated": {"features": {"rows": len(features), "sha256": content_sha256(features)}},
        "seconds": {"prepare": prepare_seconds, "rollout": rollout_seconds},
        "versions": _versions(),
        "device": "cpu",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    print(f"B-07 시험 운용: 시드 평균 IR {metrics['ir']['mean_over_seeds']:.3f}, 완료 기준 {criteria['b07_complete']} → {out}")
    return out


if __name__ == "__main__":
    main()
