"""작업 B-06: 폴드 하나의 λ 탐색과 PPO 학습(6.5절)을 돌려 reports/{run_id}/에 남긴다 (R-10).

    python -m src.portfolio.rl_fold_report --run-id 20261110_B_rl_y2016 --fold 2016 --main-seeds 0

- 폴드 분할: A의 `src.models.splits.make_folds` (4.7절). A-06·A-07과 같은 학습·검증 신호일을 쓴다.
  purge는 h = labels.main_horizon_days(20)다 (DEC-21 (g)). C의 실제 라벨 종료일(DEC-09 (3))은 아직 공개 표가
  없어 명목 t+1+h로 자른다. 그렇게 했다고 metrics.json에 적는다.
- 신호: ML 신호(A-06·A-07, DEC-08 폴드별 wf 파일)가 나오기 전에는 A의 베이스라인 신호로 리허설한다 (B-06).
  베이스라인은 규칙 기반이라 학습 구간에서도 교차 적합이 필요 없다.
- 피처·엔진·M9는 B-05 리포트(env_check_report)와 같다. M9 쿨다운·이벤트 청산은 비운다 (M9 백테스트 경보 전).
- 학습 환경: 폴드 학습 신호일, episode="random". 검증 환경: 폴드 검증 신호일, episode="full".
  둘 다 2011~2025 신호일 달력을 넘겨 purge로 빠진 신호일을 건너뛰는 스텝을 만들지 않는다 (DEC-21 (g)).
  검증 IR은 C의 `validation_ir`(M5)다.
- λ 탐색: `rl_train.select_lambda` (격자 전부, 시드 0, 100,000스텝, 10,000스텝마다 검증 IR).
  본 학습: 고른 λ로 --main-seeds의 시드마다 300,000스텝. 기본은 사전등록 시드 0~4 전부(B-07)이고,
  리허설에서 일부만 돌리면 metrics.json에 적는다.
- 완료 기준(B-06): λ 선택 절차가 끝까지 돈다. 격자의 λ마다 평가 기록이 빠짐없고 체크포인트가 있으며,
  고른 λ가 '최고 검증 IR, 동률이면 작은 λ'(DEC-21 (f))와 같다.

reports/{run_id}/에 config.yaml(사본), git_commit.txt, metrics.json, lambda_search.csv,
lambda_{λ}/eval_log.csv·best.zip, seed_{s}/eval_log.csv·best.zip을 쓴다.
best.zip(가중치)은 git에 넣지 않고 팀 드라이브 run_id 폴더에 올린다 (PM 11/10). 드라이브 파일이 정본이고,
metrics.json의 checkpoints에 파일마다 SHA-256(드라이브 파일 대조)과 정책 파라미터 지문(다시 학습한 가중치 비교)을 남긴다.
같은 run_id 폴더가 있으면 덮어쓰지 않고 멈춘다 (R-11). 학습이 중간에 실패하면 그때까지의 파일이 남는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.metrics import validation_ir
from src.common import io
from src.models.splits import Fold, make_folds
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.env_check_report import (
    FIRST_SIGNAL_DATE,
    LAST_SIGNAL_DATE,
    TABLES,
    _a_signal_function,
    _pivot,
    baseline_signal_table,
    content_sha256,
    feature_table,
    input_fingerprints,
    kospi200,
    members_by_date,
    signal_dates,
)
from src.portfolio.interfaces import IRFunction
from src.portfolio.params import PREREG_PATH, RiskLayerParams, RLParams, load_prereg
from src.portfolio.rl_data import SignalSnapshot, build_snapshots
from src.portfolio.rl_env import PortfolioEnv
from src.portfolio.rl_train import TrainResult, select_lambda, train_one
from src.portfolio.vol_target_report import ROOT, RUN_ID_RE, _git_commit

PURGE_H = 20  # portfolio.rl.env.signal_dates = "purged_h20_no_step_over_gaps" (DEC-21 (g))
ENGINE_TABLES = ("prices", "status", "index", "delisted", "delisted_reused")


# ---------------------------------------------------------------------------
# 폴드
# ---------------------------------------------------------------------------


def _iso(dates: Sequence[Any]) -> list[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]


def fold_split(
    dates: Sequence[str], trading_days: Sequence[Any], test_year: int, cfg: Mapping[str, Any],
    actual_end_dates: Mapping[Any, Any] | None = None,
) -> Fold:
    """폴드 Y의 학습·검증·테스트 신호일 (4.7절). A의 make_folds를 그대로 부른다."""
    years = [int(y) for y in cfg["splits"]["walk_forward_test_years"]]
    if test_year not in years:
        raise ValueError(f"폴드 {test_year}는 splits.walk_forward_test_years {years}에 없다")
    if int(cfg["splits"]["validation_years"]) != 2:
        raise ValueError("splits.validation_years: A의 make_folds는 검증 2년((Y−2)년, (Y−1)년)만 구현한다")
    h = int(cfg["labels"]["main_horizon_days"])
    if h != PURGE_H:
        raise ValueError(f"labels.main_horizon_days = {h}: portfolio.rl.env.signal_dates는 h = {PURGE_H} purge다")
    (fold,) = make_folds(
        pd.DatetimeIndex(pd.to_datetime(list(dates))), pd.DatetimeIndex(pd.to_datetime(list(trading_days))),
        h=h, test_years=(test_year,), first_train_year=int(cfg["data"]["first_train_signal_year"]),
        actual_end_dates=actual_end_dates,
    )
    return fold


@dataclass(frozen=True)
class FoldEnvs:
    make_env: Callable[[float], PortfolioEnv]  # λ → 학습 환경 (episode="random")
    val_env: PortfolioEnv  # 검증 구간 전체 (episode="full")
    train_dates: list[str]
    val_dates: list[str]


def fold_envs(
    fold: Fold, snapshots: Sequence[SignalSnapshot], engine: Any, rl: RLParams, risk: RiskLayerParams,
    calendar: Sequence[str],
) -> FoldEnvs:
    """폴드의 학습·검증 환경. calendar는 purge 전 신호일 달력이다 (DEC-21 (g))."""
    by_date = {s.date: s for s in snapshots}
    train, val = _iso(fold.train), _iso(fold.val)
    missing = [d for d in train + val if d not in by_date]
    if missing:
        raise ValueError(f"스냅샷이 없는 폴드 신호일이 있다: {missing[:3]}")
    cal = list(calendar)
    train_snaps = [by_date[d] for d in train]
    val_env = PortfolioEnv([by_date[d] for d in val], engine, rl, risk, 0.0, "full", calendar=cal)

    def make_env(lam: float) -> PortfolioEnv:
        return PortfolioEnv(train_snaps, engine, rl, risk, lam, "random", calendar=cal)

    make_env(0.0)  # 학습 전에 52스텝 연속 구간이 있는지 확인한다
    return FoldEnvs(make_env, val_env, train, val)


# ---------------------------------------------------------------------------
# 학습과 완료 기준
# ---------------------------------------------------------------------------


def lambda_choice(table: pd.DataFrame) -> float:
    """최고 검증 IR, 동률이면 작은 λ (DEC-21 (f)). select_lambda와 따로 계산해 대조한다."""
    best = table["best_val_ir"].max()
    return float(table.loc[table["best_val_ir"] == best, "lambda"].min())


def _eval_steps(total: int, every: int) -> list[int]:
    return [every * k for k in range(1, total // every + 1)]


def _log_check(log: pd.DataFrame, best_path: Path, total: int, every: int) -> dict[str, Any]:
    want = _eval_steps(total, every)
    return {
        "eval_steps_complete": [int(x) for x in log["timesteps"]] == want,
        "n_evals": len(log),
        "finite_evals": int(np.isfinite(log["val_ir"].to_numpy(dtype=float)).sum()),
        "checkpoint": best_path.exists(),
    }


def run_fold(envs: FoldEnvs, rl: RLParams, ir_fn: IRFunction, out: Path, main_seeds: Sequence[int]) -> dict[str, Any]:
    """λ 탐색과 본 학습. 결과 파일은 out 아래에 쓴다."""
    start = perf_counter()
    lam, table = select_lambda(envs.make_env, envs.val_env, rl, ir_fn, out)
    table.to_csv(out / "lambda_search.csv", index=False)
    search_seconds = perf_counter() - start

    search = rl.lambda_search
    per_lambda = []
    for row in table.to_dict("records"):
        folder = out / f"lambda_{row['lambda']:g}"  # rl_train.select_lambda가 쓰는 폴더
        log = pd.read_csv(folder / "eval_log.csv")
        per_lambda.append({
            "lambda": float(row["lambda"]), "best_val_ir": float(row["best_val_ir"]),
            "best_timesteps": int(row["best_timesteps"]),
            **_log_check(log, folder / "best.zip", search.total_timesteps, rl.eval_every_steps),
        })

    start = perf_counter()
    main: list[TrainResult] = [
        train_one(envs.make_env, envs.val_env, lam, seed, rl.total_timesteps_per_seed, rl.eval_every_steps, ir_fn,
                  out / f"seed_{seed}")
        for seed in main_seeds
    ]
    main_seconds = perf_counter() - start
    per_seed = [
        {"seed": r.seed, "best_val_ir": float(r.best_val_ir), "best_timesteps": int(r.best_timesteps),
         **_log_check(r.eval_log, r.best_path, rl.total_timesteps_per_seed, rl.eval_every_steps)}
        for r in main
    ]

    grid_done = [p["lambda"] for p in per_lambda] == list(search.grid)
    logs_done = all(p["eval_steps_complete"] and p["checkpoint"] and p["finite_evals"] > 0 for p in per_lambda + per_seed)
    rule = lambda_choice(table)
    return {
        "criteria": {
            "lambda_selection_ran_to_the_end": bool(grid_done and logs_done and lam == rule),
            "every_lambda_trained": grid_done,
            "eval_logs_and_checkpoints_complete": logs_done,
            "chosen_lambda_follows_dec21_f": lam == rule,
        },
        "lambda_search": {
            "grid": list(search.grid), "seed": search.seed, "total_timesteps": search.total_timesteps,
            "eval_every_steps": rl.eval_every_steps, "rule": "max_checkpoint_val_ir_then_smaller_lambda",
            "chosen_lambda": lam, "per_lambda": per_lambda,
        },
        "main_training": {
            "lambda": lam, "seeds": list(main_seeds), "prereg_seeds": list(rl.seeds),
            "all_prereg_seeds": list(main_seeds) == list(rl.seeds),
            "total_timesteps_per_seed": rl.total_timesteps_per_seed, "per_seed": per_seed,
        },
        "seconds": {"lambda_search": search_seconds, "main_training": main_seconds},
    }


CHECKPOINT = "best.zip"  # rl_train이 λ·시드 폴더마다 쓰는 최고 검증 IR 체크포인트


def policy_sha256(path: Path) -> str:
    """체크포인트의 정책 파라미터 값 지문. zip 파일에는 저장 시각과 시스템 정보가 들어가 같은 가중치라도
    파일 SHA-256이 저장마다 달라진다. 이 지문은 파라미터 이름·모양·dtype·값만 본다."""
    from stable_baselines3.common.save_util import load_from_zip_file

    _, params, _ = load_from_zip_file(path, device="cpu")
    h = hashlib.sha256()
    for name, tensor in sorted(params["policy"].items()):
        t = tensor.detach().cpu().contiguous()
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}|".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def checkpoint_fingerprints(out: Path) -> dict[str, dict[str, Any]]:
    """out 아래 λ·시드 폴더의 체크포인트마다 크기, 파일 SHA-256, 정책 파라미터 지문."""
    paths = sorted(out.glob(f"*/{CHECKPOINT}"), key=lambda p: p.relative_to(out).as_posix())
    return {
        p.relative_to(out).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "policy_sha256": policy_sha256(p),
        }
        for p in paths
    }


def _span(dates: Sequence[str]) -> dict[str, Any]:
    return {"n": len(dates), "first": dates[0], "last": dates[-1]}


def fold_summary(fold: Fold, dates: Sequence[str], purge_end_dates: str) -> dict[str, Any]:
    """폴드 신호일과 purge로 빠진 수."""
    sig = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    y = int(fold.test_year)
    train_years = sig[(sig.year >= sig.year.min()) & (sig.year <= y - 3)]
    val_years = sig[(sig.year >= y - 2) & (sig.year <= y - 1)]
    return {
        "test_year": y,
        "train": {**_span(_iso(fold.train)), "purged": len(train_years) - len(fold.train)},
        "val": {**_span(_iso(fold.val)), "purged": len(val_years) - len(fold.val)},
        "test": _span(_iso(fold.test)),
        "purge": {"h": PURGE_H, "end_dates": purge_end_dates},
    }


def _versions() -> dict[str, str]:
    import gymnasium
    import stable_baselines3
    import torch

    return {"python": platform.python_version(), "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__, "gymnasium": gymnasium.__version__, "numpy": np.__version__,
            "pandas": pd.__version__}


def _seeds(text: str | None, rl: RLParams) -> list[int]:
    if text is None:
        return list(rl.seeds)
    seeds = [int(x) for x in text.split(",") if x.strip()]
    unknown = [s for s in seeds if s not in rl.seeds]
    if unknown or len(set(seeds)) != len(seeds):
        raise SystemExit(f"--main-seeds {text!r}: 사전등록 시드 {list(rl.seeds)} 안에서 중복 없이 고른다")
    return seeds


def main(argv: Sequence[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="B-06 폴드 하나의 λ 탐색과 PPO 학습 (베이스라인 신호 리허설)")
    ap.add_argument("--run-id", required=True, help="{YYYYMMDD}_B_{짧은설명}")
    ap.add_argument("--fold", type=int, required=True, help="폴드 Y = 시험 연도 (splits.walk_forward_test_years)")
    ap.add_argument("--main-seeds", default=None,
                    help="본 학습 시드 (쉼표 구분, 빈 문자열이면 본 학습 안 함). 기본은 portfolio.rl.seeds 전부")
    ap.add_argument("--config", default=str(PREREG_PATH), help="부록 1 설정 (기본 configs/prereg.yaml)")
    args = ap.parse_args(argv)
    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit(f"run_id는 '{{YYYYMMDD}}_B_{{짧은설명}}' 형식이어야 한다: {args.run_id!r}")
    out = ROOT / "reports" / args.run_id
    if out.exists():
        raise SystemExit(f"{out}가 이미 있다. 결과를 덮어쓰지 않는다 (R-11)")
    cfg_path = Path(args.config)
    cfg = load_prereg(cfg_path)
    rl, risk = RLParams.from_prereg(cfg), RiskLayerParams.from_prereg(cfg)
    seeds = _seeds(args.main_seeds, rl)
    signal_fn = _a_signal_function()

    start = perf_counter()
    tables = {name: io.read_table(name) for name in TABLES}
    bm = kospi200(tables["index"])
    dates = signal_dates(tables["universe"], bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)
    fold = fold_split(dates, bm.index, args.fold, cfg)
    needed = sorted(set(_iso(fold.train)) | set(_iso(fold.val)))
    members = members_by_date(tables["universe"], needed)
    union = sorted({tk for tickers in members.values() for tk in tickers})
    signals = baseline_signal_table(_pivot(tables["prices"], "close", union), members, signal_fn)
    features = feature_table(tables["prices"], bm, members, risk)
    inputs = engine_inputs(*(tables[n] for n in ENGINE_TABLES))
    snaps = build_snapshots(needed, signals, features, tables["universe"], tables["sector"], inputs.bm_close,
                            lambda d: (), risk)
    envs = fold_envs(fold, snaps, build_engine(inputs, cfg), rl, risk, dates)
    prepare_seconds = perf_counter() - start

    commit = _git_commit()  # 결과 폴더를 만들기 전에 (R-10)
    out.mkdir(parents=True)
    shutil.copyfile(cfg_path, out / "config.yaml")
    (out / "git_commit.txt").write_text(commit, encoding="utf-8")

    result = run_fold(envs, rl, validation_ir, out, seeds)
    result["seconds"]["prepare"] = prepare_seconds
    metrics = {
        "run_id": args.run_id,
        "task": "B-06",
        "rehearsal": {"signals": "A baseline_signals (ML 신호 전, B-06)", "all_prereg_seeds": seeds == list(rl.seeds)},
        **result,
        "checkpoints": {
            "storage": f"best.zip은 git에 넣지 않고 팀 드라이브 {args.run_id}/ 폴더에 둔다 (PM 11/10). "
                       "드라이브 파일이 정본이고 sha256으로 대조한다",
            "policy_sha256": "정책 파라미터 값만의 지문. 파일 sha256은 저장 시각이 들어가 다시 학습하면 달라지므로 "
                             "같은 가중치인지는 이것으로 비교한다",
            "files": checkpoint_fingerprints(out),
        },
        "fold": fold_summary(fold, dates, "명목 t+1+h 거래일 (C 실제 라벨 종료일 미반영)"),
        "m9": "쿨다운·이벤트 청산 없음 (M9 백테스트 경보 전)",
        "ir_function": "src.backtest.metrics.validation_ir",
        "universe_size": {
            "mean": float(np.mean([len(s.tickers) for s in snaps])),
            "min": int(min(len(s.tickers) for s in snaps)), "max": int(max(len(s.tickers) for s in snaps)),
        },
        "inputs": input_fingerprints(tables, io.ROOT),
        "generated": {
            "signals": {"source": "src.models.baseline.baseline_signals", "rows": len(signals),
                        "sha256": content_sha256(signals)},
            "features": {"source": "src.features.price_features + B market_state_table (regime_mc)",
                         "rows": len(features), "sha256": content_sha256(features)},
        },
        "versions": _versions(),
        "device": "cpu",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    c = metrics["criteria"]
    print(f"B-06 폴드 {args.fold}: λ = {result['lambda_search']['chosen_lambda']:g}, "
          f"λ 선택 절차 끝까지: {c['lambda_selection_ran_to_the_end']} → {out}")
    return out


if __name__ == "__main__":
    main()
