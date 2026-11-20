"""작업 B-08: G5·G7 리포트 (7장). R-10 기록을 남긴다.

    python -m src.portfolio.g5_g7_report --run-id 20261118_B_g5g7 --buffer-run-id 20261117_B_buffer_wf

- 입력: --buffer-run-id의 reports/{run_id}/ (buffer_wf_report, B-07 G5 비교 표)와 그 기록이 가리키는 RL 시험 운용
  (rl_wf_report), 폴드 학습(rl_fold_report) 결과. positions 8개와 9.3절 표, M9 경보 파일의 지문이 G5 비교 표의 인계
  기록과 하나라도 다르거나 설정이 다르면 멈춘다.
- G5 판정 표: buffer_wf_report의 판정을 7장 표 모양으로 옮긴다. 값을 다시 계산하지 않는다.
- G7 표: B 확인값이다. 판정은 C-07이 같은 positions로 한다.
- 시드별 결과: G5 지표, 반등 포착률, 평균 현금, 연 비용.
- 반등 포착률 (6.6절): C의 rebound_capture. 일간 수익률 d−1→d에 d−1 종가 기준 regime_mc를 맞춘다. regime_mc는
  market_state_table(mode = no_exposure)다 (design_C). 일간 NAV는 G5 비교 표의 daily_nav.csv다. 국면 1인 날이
  없어 정의되지 않으면 None(미산출)으로 적고 완료 기준에 넣지 않는다.
- 원인 분해 (#58 PM 리뷰, 11장 'RL이 버퍼 규칙을 이기지 못함'): 주간 a_k를 세 몫으로 나눈다.
  - 현금 몫: −c_k × R_BM,k (c_k = 신호일 k의 목표 현금 비중)
  - 비용 몫: −(그 구간 장부의 체결·청산 비용 합 ÷ 체결 직전 NAV)
  - 나머지: a_k − 현금 몫 − 비용 몫 (비용 전 종목 선택과 비중 변화)
  연 값은 주간 평균 × 52(산술)다. 비용은 positions를 C 엔진(같은 비용·M9 일정)으로 다시 체결한 장부에서 읽는다.
  다시 체결한 a_k가 G5 비교 표의 weekly_excess.csv와 같아야 한다 (완료 기준).
- 연도별 표: 실행별 연 수익률(첫해는 첫 체결 직전 NAV 1부터), 코스피 200(첫 체결일 종가부터), 평균 E_t와 현금.
- 폴드 표: 폴드별 학습·검증 신호일 수(샘플 수), λ, 검증 IR과 시험 IR(5시드 평균). 검증·시험 IR 상관.
- 출력: reports/{run_id}/에 config.yaml, git_commit.txt, metrics.json, g5_judgment.csv, g7_check.csv, seeds.csv,
  rebound.csv, decomposition.csv, yearly.csv, folds.csv, report.md.
- 같은 run_id의 결과 폴더가 있으면 덮어쓰지 않고 멈춘다 (R-11).
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import metrics as m5
from src.common import io
from src.portfolio.buffer_wf_report import different_inputs, targets_by_date
from src.portfolio.engine_adapter import build_engine, cost_rates, engine_inputs
from src.portfolio.env_check_report import (
    FIRST_SIGNAL_DATE,
    LAST_SIGNAL_DATE,
    TABLES,
    input_fingerprints,
    kospi200,
    signal_dates,
)
from src.portfolio.params import PREREG_PATH, RiskLayerParams, load_prereg
from src.portfolio.risk_layer import CASH, market_state_table
from src.portfolio.rl_fold_report import ENGINE_TABLES, _file_sha256, m9_schedule
from src.portfolio.rl_wf_report import _next_trading_day
from src.portfolio.vol_target_report import ROOT, RUN_ID_RE, _git_commit

REPLAY_TOLERANCE = 1e-12
WEEKS = 52
BUFFER_NAMES = ("buffer_full", "buffer_off", "buffer_no_exposure")


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"결과가 없다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(root: Path, buffer_run_id: str, config: bytes) -> dict[str, Any]:
    """G5 비교 표, RL 시험 운용, 폴드 학습 결과. 설정 사본이 지금 설정과 같아야 한다."""
    folder = root / "reports" / buffer_run_id
    buf = _json(folder / "metrics.json")
    if buf.get("task") != "B-07" or "g5" not in buf or "handoff" not in buf:
        raise ValueError(f"{buffer_run_id}: B-07 G5 비교 표(buffer_wf_report) 결과가 아니다")
    if (folder / "config.yaml").read_bytes() != config:
        raise ValueError(f"{buffer_run_id}: config.yaml이 지금 설정과 다르다")
    rl_id = buf["rl_run"]["run_id"]
    rl = _json(root / "reports" / rl_id / "metrics.json")
    folds = {int(f["fold"]): _json(root / "reports" / f["run_id"] / "metrics.json") for f in rl["fold_runs"]["folds"]}
    return {
        "buffer": buf, "rl": rl, "folds": folds,
        "weekly": pd.read_csv(folder / "weekly_excess.csv", dtype={"signal_date": str, "week": str}),
        "nav": pd.read_csv(folder / "daily_nav.csv", index_col="date", parse_dates=["date"]),
        "fold_seed_ir": pd.read_csv(root / "reports" / rl_id / "fold_seed_ir.csv"),
    }


def load_positions(handoff: Mapping[str, Mapping[str, Any]]) -> dict[str, pd.DataFrame]:
    """인계 기록의 positions. 파일 지문이 기록과 다르면 멈춘다."""
    out = {}
    for name, h in handoff.items():
        path = io.ROOT / h["path"]
        if not path.exists():
            raise SystemExit(f"{path}가 없다")
        if _file_sha256(path) != h["sha256"]:
            raise ValueError(f"{name} positions 지문이 G5 비교 표 기록과 다르다: {h['path']}")
        out[name] = io.read_table("positions", run_id=h["run_id"])
    return out


# ---------------------------------------------------------------------------
# 다시 체결해 비용 읽기
# ---------------------------------------------------------------------------


def replay_costs(positions: pd.DataFrame, dates: Sequence[str], engine: Any) -> tuple[np.ndarray, np.ndarray]:
    """positions를 현금 100%에서 다시 체결해 구간마다 (a_k, 비용 몫)을 돌려준다. 비용 몫은 그 구간 장부에 남은
    체결·청산 비용의 합 ÷ 체결 직전 NAV다. engine은 record_ledger=True로 새로 만든 엔진이다."""
    if not getattr(engine, "record_ledger", False) or engine.ledger:
        raise ValueError("record_ledger=True로 새로 만든 엔진이어야 한다")
    targets = targets_by_date(positions)
    if sorted(targets) != list(dates):
        raise ValueError("positions의 신호일이 시험 신호일과 다르다")
    state, excess, cost = engine.initial_state(), [], []
    for d, nxt in zip(dates[:-1], dates[1:]):
        start = len(engine.ledger)
        res = engine.run_period(state, targets[d], d, nxt)
        state = res.state
        excess.append(res.r_net - res.r_bm)
        cost.append(sum(float(e["cost"]) for e in engine.ledger[start:]) / res.nav_start)
    return np.asarray(excess, dtype=np.float64), np.asarray(cost, dtype=np.float64)


# ---------------------------------------------------------------------------
# 표
# ---------------------------------------------------------------------------


def cash_by_date(positions: pd.DataFrame) -> pd.Series:
    """신호일(YYYY-MM-DD) → 목표 현금 비중."""
    c = positions[positions["ticker"] == CASH]
    return pd.Series(c["target_weight"].to_numpy(dtype=np.float64),
                     index=pd.to_datetime(c["date"]).dt.strftime("%Y-%m-%d").to_numpy())


def benchmark_weekly(bm: pd.Series, signal_dates_: Sequence[str], last_signal: str) -> np.ndarray:
    """체결 구간마다 R_BM = BM(다음 체결일) ÷ BM(체결일) − 1. 체결일은 신호일 다음 거래일이다."""
    days = bm.index
    fills = [_next_trading_day(days, d) for d in [*signal_dates_, last_signal]]
    close = bm.reindex(pd.DatetimeIndex(fills)).to_numpy(dtype=np.float64)
    return close[1:] / close[:-1] - 1.0


def decomposition(a: np.ndarray, cash: np.ndarray, r_bm: np.ndarray, cost: np.ndarray) -> dict[str, float]:
    """주간 a_k = 현금 몫 + 비용 몫 + 나머지. 연 값은 주간 평균 × 52(산술)."""
    cash_part, cost_part = -cash * r_bm, -cost
    rest = a - cash_part - cost_part
    return {
        "excess": float(a.mean() * WEEKS), "cash": float(cash_part.mean() * WEEKS),
        "cost": float(cost_part.mean() * WEEKS), "selection": float(rest.mean() * WEEKS),
        "mean_cash": float(cash.mean()),
    }


def rebound(nav: pd.Series, bm: pd.Series, regime: pd.Series) -> dict[str, Any]:
    """6.6절 반등 포착률 (C rebound_capture). 일간 수익률 d−1→d에 d−1 종가의 regime_mc를 맞춘다."""
    days = nav.index
    prev = regime.reindex(days).shift(1).iloc[1:]
    if prev.isna().any():
        raise ValueError("regime_mc가 없는 거래일이 있다")
    p = nav.to_numpy(dtype=np.float64)
    b = bm.reindex(days).to_numpy(dtype=np.float64)
    pr, br, reg = p[1:] / p[:-1] - 1.0, b[1:] / b[:-1] - 1.0, prev.to_numpy().astype(int)
    on = reg == 1
    return {
        "rebound_capture": m5.rebound_capture(pr, br, reg),
        "days": int(on.sum()),
        "portfolio_return": float(np.prod(1 + pr[on]) - 1) if on.any() else None,
        "benchmark_return": float(np.prod(1 + br[on]) - 1) if on.any() else None,
    }


def yearly(nav: pd.DataFrame, positions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """연 수익률. 실행은 첫해를 첫 체결 직전 NAV 1부터, 코스피 200은 첫 체결일 종가부터 센다."""
    year_end = nav.groupby(nav.index.year).last()
    start = nav.iloc[0].copy()
    start[[c for c in nav.columns if c != "KOSPI200"]] = 1.0
    prev = pd.concat([start.to_frame().T, year_end.iloc[:-1]])
    prev.index = year_end.index
    out = (year_end / prev - 1.0).add_suffix("_return")
    full = positions["buffer_full"].drop_duplicates("date")  # E_t는 신호일마다 하나
    out["buffer_full_mean_exposure"] = full.groupby(pd.to_datetime(full["date"]).dt.year)["exposure"].mean()
    for name, pos in positions.items():
        c = pos[pos["ticker"] == CASH]
        out[f"{name}_mean_cash"] = c.groupby(pd.to_datetime(c["date"]).dt.year)["target_weight"].mean()
    out.index.name = "year"
    return out


def fold_table(rl: Mapping[str, Any], folds: Mapping[int, Mapping[str, Any]], fold_seed_ir: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in rl["fold_runs"]["folds"]:
        year, fm = int(f["fold"]), folds[int(f["fold"])].get("fold", {})
        test = fold_seed_ir[fold_seed_ir["fold"] == year]["ir"]
        rows.append({
            "fold": year, "train_signal_dates": fm.get("train", {}).get("n"), "val_signal_dates": fm.get("val", {}).get("n"),
            "lambda": f["lambda"], "val_ir_mean": float(np.mean(list(f["best_val_ir"].values()))),
            "test_ir_mean": float(test.mean()), "test_ir_min": float(test.min()), "test_ir_max": float(test.max()),
        })
    return pd.DataFrame(rows)


def _corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def val_test_correlation(rl: Mapping[str, Any], fold_seed_ir: pd.DataFrame, table: pd.DataFrame) -> dict[str, Any]:
    pairs = [(float(v), float(fold_seed_ir[(fold_seed_ir["fold"] == int(f["fold"])) & (fold_seed_ir["seed"] == int(s))]["ir"].iloc[0]))
             for f in rl["fold_runs"]["folds"] for s, v in f["best_val_ir"].items()]
    return {"folds": _corr(table["val_ir_mean"], table["test_ir_mean"]),
            "fold_seed_pairs": _corr(*zip(*pairs)) if pairs else None, "n_pairs": len(pairs),
            "note": "피어슨 상관. 표본이 3개 미만이거나 한쪽이 상수면 None"}


def g5_judgment_table(g5: Mapping[str, Any]) -> pd.DataFrame:
    rl, buf = g5["rl_mean_over_seeds"], g5["buffer"]
    return pd.DataFrame([
        {"criterion": "IR_RL − IR_버퍼 ≥ min_ir_gain", "rl": rl["ir"], "buffer": buf["ir"], "value": g5["ir_gain"],
         "threshold": g5["min_ir_gain"], "pass": g5["pass_by_ir"]},
        {"criterion": "CAGR_RL ≥ CAGR_버퍼", "rl": rl["cagr"], "buffer": buf["cagr"], "value": rl["cagr"] - buf["cagr"],
         "threshold": 0.0, "pass": g5["cagr_not_lower"]},
        {"criterion": "연 회전율_RL ≤ max_turnover_ratio × 연 회전율_버퍼", "rl": rl["annual_turnover"],
         "buffer": buf["annual_turnover"], "value": g5["turnover_ratio"], "threshold": g5["max_turnover_ratio"],
         "pass": g5["turnover_ratio"] <= g5["max_turnover_ratio"]},
        {"criterion": "대안 기준 (CAGR 그리고 회전율)", "rl": None, "buffer": None, "value": None, "threshold": None,
         "pass": g5["pass_by_alternative"]},
        {"criterion": "G5 (IR 기준 또는 대안 기준)", "rl": None, "buffer": None, "value": None, "threshold": None,
         "pass": g5["pass"]},
    ])


def g7_table(g7: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {"criterion": "(MDD_미적용 − MDD_적용) ÷ MDD_미적용 ≥ min_mdd_reduction_rel", "applied": g7["mdd_applied"],
         "unapplied": g7["mdd_unapplied"], "value": g7["mdd_reduction_rel"], "threshold": g7["min_mdd_reduction_rel"],
         "pass": g7["mdd_reduction_rel"] >= g7["min_mdd_reduction_rel"]},
        {"criterion": "CAGR_적용 ≥ CAGR_미적용 − max_cagr_giveup", "applied": g7["cagr_applied"],
         "unapplied": g7["cagr_unapplied"], "value": g7["cagr_applied"] - g7["cagr_unapplied"],
         "threshold": -g7["max_cagr_giveup"], "pass": g7["cagr_applied"] >= g7["cagr_unapplied"] - g7["max_cagr_giveup"]},
        {"criterion": "G7 (B 확인값, 판정은 C-07)", "applied": None, "unapplied": None, "value": None,
         "threshold": None, "pass": g7["pass"]},
    ])


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "−"
    if isinstance(v, (bool, np.bool_)):
        return "통과" if v else "미달"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{100 * v:.1f}%" if pct else f"{v:.3f}"
    return str(v)


def md_table(df: pd.DataFrame, pct: Sequence[str] = ()) -> str:
    cols = list(df.columns)
    values = {c: df[c].tolist() for c in cols}  # 열마다 읽어 정수 열이 실수로 바뀌지 않게 한다 (iterrows는 바꾼다)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for i in range(len(df)):
        lines.append("| " + " | ".join(_fmt(values[c][i], c in pct) for c in cols) + " |")
    return "\n".join(lines)


def report_md(run_id: str, buffer_run_id: str, g5: pd.DataFrame, g7: pd.DataFrame, seeds: pd.DataFrame,
              decomp: pd.DataFrame, rebounds: pd.DataFrame, years: pd.DataFrame, folds: pd.DataFrame,
              corr: Mapping[str, Any]) -> str:
    year_cols = [c for c in years.columns if c.endswith("_return")]
    return "\n\n".join([
        f"# G5·G7 리포트 ({run_id})",
        f"입력: G5 비교 표 `{buffer_run_id}`. 모든 수치는 C의 M5 함수로 계산했다. G7은 B 확인값이고 판정은 C-07이 한다.",
        "## G5 판정 표 (7장)", md_table(g5),
        "## G7 확인값 (7장, 판정은 C-07)", md_table(g7),
        "## 시드별 결과", md_table(seeds, pct=("cagr", "annual_excess", "cost_annual", "mean_cash")),
        "## 반등 포착률 (6.6절)", md_table(rebounds, pct=("portfolio_return", "benchmark_return")),
        "## 원인 분해 (주간 a_k, 연 = 주간 평균 × 52)",
        md_table(decomp, pct=("excess", "cash", "cost", "selection", "mean_cash")),
        "## 연도별 수익률", md_table(years.reset_index()[["year", *year_cols]], pct=year_cols),
        "## 폴드 (샘플 수, λ, 검증·시험 IR)", md_table(folds),
        f"검증·시험 IR 상관: 폴드 평균 {_fmt(corr['folds'])}, 폴드 × 시드 {_fmt(corr['fold_seed_pairs'])} "
        f"({corr['n_pairs']}쌍).",
    ]) + "\n"


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="B-08 G5·G7 리포트")
    ap.add_argument("--run-id", required=True, help="{YYYYMMDD}_B_{짧은설명}")
    ap.add_argument("--buffer-run-id", required=True, help="G5 비교 표 run_id (reports/{값}/, buffer_wf_report)")
    ap.add_argument("--config", default=str(PREREG_PATH), help="부록 1 설정 (기본 configs/prereg.yaml)")
    args = ap.parse_args(argv)
    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit(f"run_id는 '{{YYYYMMDD}}_B_{{짧은설명}}' 형식이어야 한다: {args.run_id!r}")
    cfg_path = Path(args.config)
    cfg = load_prereg(cfg_path)
    risk = RiskLayerParams.from_prereg(cfg)
    years = [int(y) for y in cfg["splits"]["walk_forward_test_years"]]
    out = ROOT / "reports" / args.run_id
    if out.exists():
        raise SystemExit(f"{out}가 이미 있다. 결과를 덮어쓰지 않는다 (R-11)")

    start = perf_counter()
    src = load_inputs(ROOT, args.buffer_run_id, cfg_path.read_bytes())
    buf, rl, handoff = src["buffer"], src["rl"], src["buffer"]["handoff"]
    positions = load_positions(handoff["positions"])
    tables = {name: io.read_table(name) for name in TABLES}
    changed = different_inputs(input_fingerprints(tables, io.ROOT), handoff["inputs"])
    if changed:
        raise ValueError(f"9.3절 입력 표가 G5 비교 표 때와 다르다: {changed}")
    bm = kospi200(tables["index"])
    dates = signal_dates(tables["universe"], bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)
    test_dates = [d for d in dates if int(d[:4]) in years]
    inputs = engine_inputs(*(tables[n] for n in ENGINE_TABLES))
    m9 = m9_schedule(handoff["m9"]["alerts_run_id"], dates[0], dates[-1])
    keys = ("applied", "alerts_run_id", "sha256")
    if [m9.record.get(k) for k in keys] != [handoff["m9"].get(k) for k in keys]:
        raise ValueError(f"M9 경보 파일이 G5 비교 표 때와 다르다: 지금 {m9.record}, 기록 {handoff['m9']}")
    prepare_seconds = perf_counter() - start

    start = perf_counter()
    weekly, nav = src["weekly"], src["nav"]
    regime = market_state_table(inputs.bm_close, risk, mode="no_exposure")["regime_mc"]
    executed = test_dates[:-1]
    r_bm = benchmark_weekly(inputs.bm_close, executed, test_dates[-1])
    decomp, rebounds, replay_diff, cost_by = {}, {}, {}, {}
    for name, h in handoff["positions"].items():
        alerts = ({"cooldowns": m9.cooldowns, "event_liquidations": m9.liquidations}
                  if h["m9_alerts_run_id"] is not None else {})
        a, cost = replay_costs(positions[name], test_dates, build_engine(inputs, cfg, record_ledger=True, **alerts))
        recorded = weekly.loc[weekly["portfolio"] == name]
        same = list(recorded["signal_date"]) == executed
        replay_diff[name] = float(np.max(np.abs(a - recorded["a_k"].to_numpy()))) if same else float("inf")
        cash = cash_by_date(positions[name]).reindex(executed).to_numpy()
        decomp[name] = decomposition(a, cash, r_bm, cost)
        cost_by[name] = decomp[name]["cost"]
        rebounds[name] = rebound(nav[name], nav["KOSPI200"], regime)
    run_seconds = perf_counter() - start

    runs = buf["runs"]
    rl_names = sorted((n for n in handoff["positions"] if n.startswith("rl_s")), key=lambda n: int(n[4:]))
    seed_cols = ("ir", "cagr", "annual_excess", "mdd", "annual_turnover", "mean_cash")

    def seed_row(portfolio: str, seed: str, values: Mapping[str, Any]) -> dict[str, Any]:
        return {"portfolio": portfolio, "seed": seed, **{k: values[k] for k in seed_cols},
                "cost_annual": values["cost_annual"], "rebound_capture": values["rebound_capture"]}

    per = {n: {**runs[n], "cost_annual": -cost_by[n], "rebound_capture": rebounds[n]["rebound_capture"]}
           for n in handoff["positions"]}
    rows = [seed_row("rl", n[4:], per[n]) for n in rl_names]
    mean = {k: float(np.mean([per[n][k] for n in rl_names])) for k in (*seed_cols, "cost_annual")}
    rc = [per[n]["rebound_capture"] for n in rl_names]
    mean["rebound_capture"] = float(np.mean(rc)) if all(v is not None for v in rc) else None
    rows.append(seed_row("rl", "mean", mean))
    rows += [seed_row(n, "", per[n]) for n in BUFFER_NAMES]
    seeds = pd.DataFrame(rows)

    g5 = g5_judgment_table(buf["g5"])
    g7 = g7_table(buf["g7_check"])
    decomp_df = pd.DataFrame([{"portfolio": n, **v} for n, v in decomp.items()])
    rebound_df = pd.DataFrame([{"portfolio": n, **v} for n, v in rebounds.items()])
    years_df = yearly(nav, positions)
    folds = fold_table(rl, src["folds"], src["fold_seed_ir"])
    corr = val_test_correlation(rl, src["fold_seed_ir"], folds)

    # 반등 포착률이 정의되지 않으면(국면 1인 날이 없거나 BM 누적수익률이 0) 판정하지 않고 미산출로 적는다 (design_C)
    criteria = {
        "g5_from_complete_table": bool(buf["criteria"].get("g5_table_complete")),
        "replay_matches_g5_table": all(v <= REPLAY_TOLERANCE for v in replay_diff.values()),
        "every_fold_in_table": sorted(folds["fold"]) == years,
    }
    criteria["b08_tables_complete"] = all(criteria.values())
    # 7장 G7 판정은 C-07이 한다. C-07 결과를 받아 B 확인값과 맞춰 본 뒤 B-08이 끝난다
    criteria["c07_g7_received"] = False

    commit = _git_commit()  # 결과 폴더를 만들기 전에 (R-10)
    out.mkdir(parents=True)
    shutil.copyfile(cfg_path, out / "config.yaml")
    (out / "git_commit.txt").write_text(commit, encoding="utf-8")
    g5.to_csv(out / "g5_judgment.csv", index=False)
    g7.to_csv(out / "g7_check.csv", index=False)
    seeds.to_csv(out / "seeds.csv", index=False)
    rebound_df.to_csv(out / "rebound.csv", index=False)
    decomp_df.to_csv(out / "decomposition.csv", index=False)
    years_df.to_csv(out / "yearly.csv")
    folds.to_csv(out / "folds.csv", index=False)
    (out / "report.md").write_text(
        report_md(args.run_id, args.buffer_run_id, g5, g7, seeds, decomp_df, rebound_df, years_df, folds, corr),
        encoding="utf-8")

    rates = cost_rates(cfg)
    metrics = {
        "run_id": args.run_id,
        "task": "B-08",
        "criteria": criteria,
        "inputs": {"buffer_run_id": args.buffer_run_id, "rl_run_id": buf["rl_run"]["run_id"],
                   "fold_run_ids": [f["run_id"] for f in rl["fold_runs"]["folds"]],
                   "positions": {n: {"run_id": h["run_id"], "sha256": h["sha256"], "m8_mode": h["m8_mode"],
                                     "m9_alerts_run_id": h["m9_alerts_run_id"]} for n, h in handoff["positions"].items()},
                   "test_signals_run_id": handoff["test_signals"]["run_id"], "m9": m9.record},
        "g5": buf["g5"],
        "g7_check": buf["g7_check"],
        "rebound": rebounds,
        "decomposition": decomp,
        "val_test_ir_correlation": corr,
        "replay": {"tolerance": REPLAY_TOLERANCE, "max_abs_diff": replay_diff},
        "definitions": {
            "rebound": "C rebound_capture, 일간 수익률 d−1→d와 d−1 종가 regime_mc (market_state_table, no_exposure)",
            "decomposition": "a_k = −c_k·R_BM,k + (−비용_k) + 나머지, 연 = 주간 평균 × 52",
            "cost": "다시 체결한 C 엔진 장부의 비용 합 ÷ 체결 직전 NAV",
            "round_trip_cost": rates.commission_per_side * 2 + rates.slippage_per_side * 2 + rates.sell_tax,
        },
        "seconds": {"prepare": prepare_seconds, "runs": run_seconds},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    print(f"B-08 G5·G7 리포트: G5 {buf['g5']['pass']}, G7 확인값 {buf['g7_check']['pass']}, "
          f"표 완료 {criteria['b08_tables_complete']} → {out}")
    return out


if __name__ == "__main__":
    main()
