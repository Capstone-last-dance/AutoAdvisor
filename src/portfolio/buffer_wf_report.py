"""작업 B-07 완료 기준의 G5 비교 표 (6.5절): 버퍼 규칙 기준선의 walk-forward 운용과 RL 시험 운용 비교. R-10 기록을 남긴다.

    python -m src.portfolio.buffer_wf_report --run-id 20261117_B_buffer_wf --rl-run-id 20261114_B_rl_wf

- RL 시험 운용: --rl-run-id의 reports/{run_id}/ (rl_wf_report, task B-07). 설정 사본이 지금 설정과 같아야 한다.
- 입력은 RL 시험 운용과 같다. 시험 신호와 M9 경보는 RL 기록의 run_id를 쓴다. 9.3절 표, 시험 신호, M9 경보 파일의
  지문과 B가 만든 features 지문이 RL 기록과 하나라도 다르면 멈춘다.
- 버퍼 규칙 실행 3개 (run_buffer_rule, 6.9절). 결정적이라 시드가 없다. RL 시험 운용과 같게 첫 시험 신호일에
  현금 100%로 시작한다. 실행 자체는 RL 시험 운용처럼 마지막 시험 신호일의 비중만 기록하고 체결하지 않는다
  (다음 신호일이 홀드아웃). 지표용 다시 체결에서는 그 목표도 체결한다 (아래 판정 기간).
  - full: 버퍼 규칙 + M8 + M9. G4 포트폴리오, G5 기준선, G7의 '적용' 쪽을 겸한다
  - off: G7 'M8·M9 미적용'. Step 2~6과 M9 청산·쿨다운을 끈다 (E_t = 1)
  - no_exposure: G4 '노출 조절 제외' 진단. Step 4~5만 끄고 M9는 켠다. 판정에 쓰지 않는다 (C-10 입력)
- 지표: positions를 C 엔진(같은 비용, 같은 M9 일정)에 다시 넣어 일별 NAV, 주간 a_k, 체결일 회전율을 얻고 C의 M5
  함수로 계산한다 (R-05). RL positions도 같은 방법으로 다시 체결한다. 다시 체결한 a_k가 RL 기록(weekly_excess.csv)과
  같아야 하고, 버퍼 규칙도 실행 때의 a_k와 같아야 한다 (완료 기준).
  - 판정 기간 (명세 7장 판정 기간, design_C 5절): 첫 시험 신호일의 체결일부터 마지막 시험 연도의 마지막 거래일
    종가까지다. 마지막 시험 신호일의 목표도 다음 거래일에 체결하고 그 해 마지막 거래일 종가까지 평가한다.
    이 잔여 구간은 일간 NAV·CAGR·MDD·연 회전율에 넣고, 완료된 두 체결일이 없어 a_k·IR 표본에서는 뺀다.
  - NAV: 첫 체결일의 체결 직전 NAV 1에서 시작해 날마다 종가 NAV(체결일은 체결 뒤)를 잇는다. 거래일 수는 첫
    체결일부터 판정 기간 마지막 거래일까지의 일간 수익률 수다
  - IR = information_ratio(a_k), CAGR = cagr(NAV, 시작 NAV 1), 연 초과수익 = annual_excess_return(NAV, 코스피 200),
    MDD = max_drawdown(체결 후 일간 NAV), 연 회전율 = annual_turnover(잔여 구간까지의 체결일 회전율)
- G5 (7장, 부록 1 kpi.G5): RL은 시드별 지표의 5시드 평균이다. IR_RL − IR_버퍼 ≥ min_ir_gain 이거나
  (CAGR_RL ≥ CAGR_버퍼 이고 연 회전율_RL ≤ max_turnover_ratio × 연 회전율_버퍼)이면 통과한다.
- G7과 G4 수치는 B의 확인값이다. G7 판정은 C-07, G4 판정은 A, 2배 비용과 G4 진단은 C-10이 positions로 계산한다.
- G0: 버퍼 규칙 3개와 RL 5개 positions의 모든 신호일을 C의 validate_g0_positions로 검사한다 (실행의 M8 모드로).
- 출력: positions/wf_{run_id}_{mode}.parquet 3개 (9.3절, constructor = buffer). reports/{run_id}/에 config.yaml,
  git_commit.txt, metrics.json(완료 기준, G5 판정, C 인계 정보), g5_table.csv(시드별 결과 표 + 5시드 평균 + 버퍼 규칙),
  weekly_excess.csv(portfolio, signal_date, week, a_k, turnover), daily_nav.csv(date, 코스피 200, 실행별 종가 NAV).
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

from src.backtest import metrics as m5
from src.backtest.validation import validate_g0_positions
from src.common import io
from src.portfolio.buffer_rule import run_buffer_rule
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
from src.portfolio.params import PREREG_PATH, BufferRuleParams, RiskLayerParams, RLParams, _expect, load_prereg
from src.portfolio.risk_layer import CASH, RiskLayerViolation
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_fold_report import ENGINE_TABLES, _file_sha256, _versions, m9_schedule
from src.portfolio.rl_wf_report import _next_trading_day, test_signal_table
from src.portfolio.vol_target_report import ROOT, RUN_ID_RE, _git_commit

# 버퍼 규칙 실행: M8 모드 → (M9 적용 여부, 7장에서 쓰는 곳)
BUFFER_RUNS: dict[str, tuple[bool, str]] = {
    "full": (True, "G4 포트폴리오, G5 기준선, G7 '적용'"),
    "off": (False, "G7 'M8·M9 미적용' (6.9절)"),
    "no_exposure": (True, "G4 '노출 조절 제외' 진단 (판정에 쓰지 않음, C-10 입력)"),
}
REPLAY_TOLERANCE = 1e-12  # 다시 체결한 a_k와 기록의 차이 허용값. 실제 최대 차이는 metrics.json에 남긴다
SUMMARY_KEYS = ("ir", "cagr", "annual_excess", "mdd", "annual_turnover", "mean_cash")


# ---------------------------------------------------------------------------
# RL 시험 운용 기록
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RLRun:
    run_id: str
    folder: Path
    metrics: dict[str, Any]
    weekly: pd.DataFrame  # weekly_excess.csv


def load_rl_run(root: Path, run_id: str, config: bytes) -> RLRun:
    folder = root / "reports" / run_id
    if not (folder / "metrics.json").exists():
        raise SystemExit(f"RL 시험 운용 결과가 없다: {folder}/metrics.json (rl_wf_report를 먼저 돌린다)")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("task") != "B-07" or "positions" not in metrics:
        raise ValueError(f"{run_id}: B-07 시험 운용(rl_wf_report) 결과가 아니다")
    if (folder / "config.yaml").read_bytes() != config:
        raise ValueError(f"{run_id}: config.yaml이 지금 설정과 다르다")
    weekly = pd.read_csv(folder / "weekly_excess.csv", dtype={"signal_date": str, "week": str})
    return RLRun(run_id, folder, metrics, weekly)


def same_as_rl(name: str, now: Any, rl: Any) -> None:
    if now != rl:
        raise ValueError(f"{name}이(가) RL 시험 운용 때와 다르다: 지금 {now}, RL {rl}")


def different_inputs(now: Mapping[str, Any], rl: Mapping[str, Any]) -> list[str]:
    return sorted(n for n in set(now) | set(rl) if now.get(n) != rl.get(n))


# ---------------------------------------------------------------------------
# 다시 체결
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Replay:
    """positions를 C 엔진에 다시 넣은 결과."""

    signal_dates: list[str]  # a_k를 계산한 신호일 (마지막 시험 신호일 제외)
    excess: np.ndarray  # 구간마다 R_net − R_BM (a_k)
    turnover: np.ndarray  # a_k 구간마다 체결일 회전율
    nav: pd.Series  # 첫 체결일 ~ 마지막 평가일의 종가 NAV (체결일은 체결 뒤, 첫 체결일 체결 직전 NAV = 1)
    final_nav: float  # 마지막 평가일의 NAV
    last_turnover: float | None = None  # 잔여 구간(마지막 시험 신호일 목표 체결)의 회전율. 잔여 구간이 없으면 None


def targets_by_date(positions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """positions(9.3절) → 신호일 → 종목 목표 비중 (CASH 제외)."""
    day = pd.to_datetime(positions["date"]).dt.strftime("%Y-%m-%d")
    out: dict[str, dict[str, float]] = {}
    for d, g in positions.groupby(day, sort=True):
        s = g[g["ticker"] != CASH]
        out[str(d)] = {str(t): float(w) for t, w in zip(s["ticker"], s["target_weight"]) if w > 0.0}
    return out


def daily_nav(ledger: Sequence[Mapping[str, Any]]) -> pd.Series:
    """엔진 원장의 날마다 마지막 기록의 체결 뒤 NAV. 구간 끝 평가일은 다음 구간 첫 체결일과 같은 날이라
    뒤 기록(그날 체결 뒤 종가 NAV)을 쓴다."""
    frame = pd.DataFrame({"date": [r["date"] for r in ledger], "nav": [float(r["nav_after"]) for r in ledger]})
    nav = frame.groupby("date", sort=False)["nav"].last()
    nav.index = pd.DatetimeIndex(nav.index)
    if not nav.index.is_monotonic_increasing:
        raise ValueError("원장 날짜가 오름차순이 아니다")
    return nav


def period_end(calendar: pd.DatetimeIndex, last_signal: str, year: int) -> tuple[str, str]:
    """판정 기간의 잔여 구간: (run_period에 넘길 next_signal_date 자리, 평가 종료일).

    평가 종료일은 마지막 시험 연도의 마지막 거래일이다. run_period는 next_signal_date 다음 거래일의 종가까지
    평가하므로 그 전 거래일을 넘긴다. 마지막 시험 신호일의 체결일이 평가 종료일보다 앞서야 한다."""
    days = pd.DatetimeIndex(calendar).sort_values()
    in_year = days[days.year == year]
    if len(in_year) == 0:
        raise ValueError(f"{year}년 거래일이 없다")
    end = in_year[-1]
    after = days[days > pd.Timestamp(last_signal)]
    if len(after) == 0 or not after[0] < end:
        raise ValueError(f"마지막 시험 신호일 {last_signal}의 체결일이 {year}년 마지막 거래일 "
                         f"{end:%Y-%m-%d}보다 앞서지 않는다")
    return days[days.get_loc(end) - 1].strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def replay(positions: pd.DataFrame, dates: Sequence[str], engine: Any,
           last_period: tuple[str, str] | None = None) -> Replay:
    """positions의 목표 비중을 현금 100%에서 신호일마다 다시 체결한다. engine은 record_ledger=True로 새로 만든
    엔진이다 (원장이 비어 있어야 한다).

    last_period(period_end의 결과)를 주면 마지막 신호일의 목표도 체결하고 평가 종료일까지 평가한다. 이 잔여
    구간은 NAV와 last_turnover에만 들어가고 a_k(excess·turnover)에는 넣지 않는다. 주지 않으면 마지막 신호일은
    체결하지 않는다."""
    if not getattr(engine, "record_ledger", False) or engine.ledger:
        raise ValueError("record_ledger=True로 새로 만든 엔진이어야 한다")
    targets = targets_by_date(positions)
    if sorted(targets) != list(dates):
        raise ValueError("positions의 신호일이 시험 신호일과 다르다")
    state, excess, turnover, nav_end = engine.initial_state(), [], [], 1.0
    for d, nxt in zip(dates[:-1], dates[1:]):
        res = engine.run_period(state, targets[d], d, nxt)
        state, nav_end = res.state, res.nav_end
        excess.append(res.r_net - res.r_bm)
        turnover.append(res.turnover)
    last_turnover = None
    if last_period is not None:
        res = engine.run_period(state, targets[dates[-1]], dates[-1], last_period[0])
        nav_end, last_turnover = res.nav_end, float(res.turnover)
    nav = daily_nav(engine.ledger)
    if abs(nav.iloc[-1] - nav_end) > 1e-9 * max(1.0, nav_end):
        raise ValueError("원장의 마지막 NAV가 마지막 구간의 nav_end와 다르다")
    if last_period is not None and nav.index[-1].strftime("%Y-%m-%d") != last_period[1]:
        raise ValueError(f"원장의 마지막 날 {nav.index[-1]:%Y-%m-%d}이 평가 종료일 {last_period[1]}과 다르다")
    return Replay(list(dates[:-1]), np.asarray(excess, dtype=np.float64), np.asarray(turnover, dtype=np.float64),
                  nav, float(nav_end), last_turnover)


def summary(rep: Replay, bm: pd.Series, positions: pd.DataFrame) -> dict[str, Any]:
    """C의 M5 함수로 계산한 실행 지표 (모듈 설명의 NAV·거래일 정의)."""
    days = rep.nav.index
    n = len(days) - 1
    nav = np.r_[1.0, rep.nav.to_numpy()[1:]]  # cagr·연 초과수익: 시작 NAV 1 ~ 마지막 평가일 (일간 수익률 n개)
    bench = bm.reindex(days).to_numpy(dtype=np.float64)
    cash = positions.loc[positions["ticker"] == CASH, "target_weight"]
    turns = rep.turnover if rep.last_turnover is None else np.r_[rep.turnover, rep.last_turnover]
    return {
        "ir": m5.information_ratio(rep.excess),
        "cagr": m5.cagr(nav, trading_days=n),
        "annual_excess": m5.annual_excess_return(nav, bench, trading_days=n),
        "mdd": m5.max_drawdown(rep.nav.to_numpy()),  # 체결 후 일간 NAV 시계열 (design_C 5절)
        "annual_turnover": m5.annual_turnover(turns, trading_days=n),
        "mean_cash": float(cash.mean()),
        "weeks": len(rep.excess),
        "trading_days": n,
        "first_fill": days[0].strftime("%Y-%m-%d"),
        "last_day": days[-1].strftime("%Y-%m-%d"),
        "final_nav": rep.final_nav,
        "last_turnover": rep.last_turnover,
    }


def benchmark_summary(bm: pd.Series, days: pd.DatetimeIndex) -> dict[str, Any]:
    close = bm.reindex(days).to_numpy(dtype=np.float64)
    return {"cagr": m5.cagr(close, trading_days=len(days) - 1), "mdd": m5.max_drawdown(close)}


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if len(a) else 0.0


def g0_violations(positions: pd.DataFrame, sector: pd.DataFrame, dates: Sequence[str], risk: RiskLayerParams,
                  mode: str) -> list[str]:
    """C의 validate_g0_positions로 신호일마다 검사한다. 한도 위반(RiskLayerViolation)이 난 신호일을 돌려준다.
    형식 오류(ValueError)는 positions를 만든 코드의 잘못이라 그대로 낸다."""
    bad = []
    for d in dates:
        try:
            validate_g0_positions(positions, sector, d, risk, m8_mode=mode)
        except RiskLayerViolation:
            bad.append(d)
    return bad


# ---------------------------------------------------------------------------
# 판정
# ---------------------------------------------------------------------------


def _defined(values: Mapping[str, Mapping[str, Any]], key: str) -> None:
    missing = [name for name, s in values.items() if s[key] is None]
    if missing:
        raise ValueError(f"{key}가 정의되지 않은 실행이 있다: {missing}")


def g5_judgment(rl: Mapping[str, Mapping[str, Any]], buffer: Mapping[str, Any], kpi: Mapping[str, Any]) -> dict[str, Any]:
    """7장 G5. rl은 시드 → 지표, buffer는 버퍼 규칙 + M8 + M9 지표다."""
    g = kpi["G5"]
    _expect(g["aggregate"], "mean_over_seeds", "kpi.G5.aggregate")
    _expect(bool(g["alternative"]["cagr_not_lower"]), True, "kpi.G5.alternative.cagr_not_lower")
    _defined({**{f"rl_s{s}": v for s, v in rl.items()}, "buffer": buffer}, "ir")
    mean = {k: float(np.mean([float(s[k]) for s in rl.values()])) for k in ("ir", "cagr", "annual_turnover")}
    ir_gain = mean["ir"] - float(buffer["ir"])
    ratio = mean["annual_turnover"] / float(buffer["annual_turnover"])
    by_ir = ir_gain >= float(g["min_ir_gain"])
    cagr_not_lower = mean["cagr"] >= float(buffer["cagr"])
    turnover_ok = ratio <= float(g["alternative"]["max_turnover_ratio"])
    return {
        "rl_mean_over_seeds": mean,
        "buffer": {k: float(buffer[k]) for k in ("ir", "cagr", "annual_turnover")},
        "ir_gain": ir_gain, "min_ir_gain": float(g["min_ir_gain"]), "pass_by_ir": by_ir,
        "cagr_not_lower": cagr_not_lower, "turnover_ratio": ratio,
        "max_turnover_ratio": float(g["alternative"]["max_turnover_ratio"]),
        "pass_by_alternative": cagr_not_lower and turnover_ok,
        "pass": by_ir or (cagr_not_lower and turnover_ok),
    }


def g7_check(applied: Mapping[str, Any], unapplied: Mapping[str, Any], kpi: Mapping[str, Any]) -> dict[str, Any]:
    """7장 G7의 B 확인값 (판정은 C-07). applied = 버퍼 규칙 + M8 + M9, unapplied = 'M8·M9 미적용'."""
    g = kpi["G7"]
    reduction = (unapplied["mdd"] - applied["mdd"]) / unapplied["mdd"] if unapplied["mdd"] > 0 else float("nan")
    mdd_ok = bool(reduction >= float(g["min_mdd_reduction_rel"]))
    cagr_ok = bool(applied["cagr"] >= unapplied["cagr"] - float(g["max_cagr_giveup"]))
    return {
        "note": "B 확인값. 판정은 C-07",
        "mdd_applied": applied["mdd"], "mdd_unapplied": unapplied["mdd"], "mdd_reduction_rel": reduction,
        "min_mdd_reduction_rel": float(g["min_mdd_reduction_rel"]),
        "cagr_applied": applied["cagr"], "cagr_unapplied": unapplied["cagr"],
        "max_cagr_giveup": float(g["max_cagr_giveup"]),
        "pass": mdd_ok and cagr_ok,
    }


def g5_table(rl: Mapping[str, Mapping[str, Any]], buffer: Mapping[str, Any]) -> pd.DataFrame:
    """6.5절 완료 기준의 시드별 결과 표와 버퍼 규칙 비교 표."""
    rows = [{"portfolio": "rl", "seed": str(s), **{k: v[k] for k in SUMMARY_KEYS}} for s, v in sorted(rl.items())]
    rows.append({"portfolio": "rl", "seed": "mean",
                 **{k: float(np.mean([float(v[k]) for v in rl.values()])) for k in SUMMARY_KEYS}})
    rows.append({"portfolio": "buffer", "seed": "", **{k: buffer[k] for k in SUMMARY_KEYS}})
    return pd.DataFrame(rows, columns=["portfolio", "seed", *SUMMARY_KEYS])


def weekly_rows(name: str, rep: Replay, days: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "portfolio": name,
        "signal_date": rep.signal_dates,
        "week": [_next_trading_day(days, d) for d in rep.signal_dates],
        "a_k": rep.excess,
        "turnover": rep.turnover,
    })


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="B-07 G5 비교 표: 버퍼 규칙 기준선 walk-forward와 RL 시험 운용")
    ap.add_argument("--run-id", required=True, help="{YYYYMMDD}_B_{짧은설명}")
    ap.add_argument("--rl-run-id", required=True, help="RL 시험 운용 run_id (reports/{값}/, rl_wf_report)")
    ap.add_argument("--config", default=str(PREREG_PATH), help="부록 1 설정 (기본 configs/prereg.yaml)")
    args = ap.parse_args(argv)
    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit(f"run_id는 '{{YYYYMMDD}}_B_{{짧은설명}}' 형식이어야 한다: {args.run_id!r}")
    cfg_path = Path(args.config)
    cfg = load_prereg(cfg_path)
    rl_params, risk = RLParams.from_prereg(cfg), RiskLayerParams.from_prereg(cfg)
    buffer = BufferRuleParams.from_prereg(cfg)
    years = [int(y) for y in cfg["splits"]["walk_forward_test_years"]]
    out = ROOT / "reports" / args.run_id
    position_ids = {mode: f"{args.run_id}_{mode}" for mode in BUFFER_RUNS}
    existing = [out] if out.exists() else []
    existing += [p for p in (io.ROOT / "positions" / f"wf_{r}.parquet" for r in position_ids.values()) if p.exists()]
    if existing:
        raise SystemExit(f"{existing[0]}가 이미 있다. 결과를 덮어쓰지 않는다 (R-11)")

    start = perf_counter()
    rl_run = load_rl_run(ROOT, args.rl_run_id, cfg_path.read_bytes())
    rl_m = rl_run.metrics
    tables = {name: io.read_table(name) for name in TABLES}
    fingerprints = input_fingerprints(tables, io.ROOT)
    changed = different_inputs(fingerprints, rl_m["inputs"])
    if changed:
        raise ValueError(f"9.3절 입력 표가 RL 시험 운용 때와 다르다: {changed}")
    bm = kospi200(tables["index"])
    dates = signal_dates(tables["universe"], bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)
    test_dates = [d for d in dates if int(d[:4]) in years]
    signals, signal_record = test_signal_table(rl_m["test_signals"]["run_id"], test_dates)
    same_as_rl("시험 신호 파일 지문", signal_record["sha256"], rl_m["test_signals"]["sha256"])
    members = members_by_date(tables["universe"], test_dates)
    features = feature_table(tables["prices"], bm, members, risk)
    same_as_rl("features 지문", content_sha256(features), rl_m["generated"]["features"]["sha256"])
    inputs = engine_inputs(*(tables[n] for n in ENGINE_TABLES))
    if not rl_m["m9"]["applied"]:
        raise ValueError(f"{args.rl_run_id}: M9 없이 운용한 RL 결과라 G5 비교 대상이 아니다")
    m9 = m9_schedule(rl_m["m9"]["alerts_run_id"], dates[0], dates[-1])
    keys = ("applied", "alerts_run_id", "sha256")
    same_as_rl("M9 경보 파일", [m9.record.get(k) for k in keys], [rl_m["m9"].get(k) for k in keys])
    snaps = build_snapshots(test_dates, signals, features, tables["universe"], tables["sector"], inputs.bm_close,
                            m9.cooldown_at, risk)
    last_period = period_end(inputs.bm_close.index, test_dates[-1], years[-1])

    def engine(with_m9: bool, ledger: bool):
        alerts = {"cooldowns": m9.cooldowns, "event_liquidations": m9.liquidations} if with_m9 else {}
        return build_engine(inputs, cfg, record_ledger=ledger, **alerts)

    rl_positions = {}
    for seed, rec in sorted(rl_m["positions"].items(), key=lambda kv: int(kv[0])):
        path = io.ROOT / rec["path"]
        same_as_rl(f"RL positions 시드 {seed} 지문", _file_sha256(path), rec["sha256"])
        rl_positions[int(seed)] = io.read_table("positions", run_id=f"{args.rl_run_id}_s{seed}")
    if sorted(rl_positions) != list(rl_params.seeds):
        raise ValueError(f"RL positions 시드 {sorted(rl_positions)}가 사전등록 시드 {list(rl_params.seeds)}가 아니다")
    prepare_seconds = perf_counter() - start

    start = perf_counter()
    runs, positions, replays, replay_diff = {}, {}, {}, {}
    for mode, (with_m9, _) in BUFFER_RUNS.items():
        run = run_buffer_rule(snaps, engine(with_m9, ledger=False), buffer, risk, mode)  # type: ignore[arg-type]
        runs[mode], positions[f"buffer_{mode}"] = run, run.positions
        rep = replay(run.positions, test_dates, engine(with_m9, ledger=True), last_period)
        replays[f"buffer_{mode}"] = rep
        replay_diff[f"buffer_{mode}"] = max(max_abs_diff(rep.excess, run.excess_returns),
                                            max_abs_diff(rep.turnover, run.turnover))
    for seed, pos in rl_positions.items():
        name = f"rl_s{seed}"
        positions[name] = pos
        rep = replays[name] = replay(pos, test_dates, engine(True, ledger=True), last_period)
        recorded = rl_run.weekly[rl_run.weekly["seed"] == seed]
        same_dates = list(recorded["signal_date"]) == rep.signal_dates
        replay_diff[name] = max_abs_diff(rep.excess, recorded["a_k"]) if same_dates else float("inf")
    run_seconds = perf_counter() - start

    days = replays["buffer_full"].nav.index
    if any(not r.nav.index.equals(days) for r in replays.values()):
        raise ValueError("실행마다 NAV 날짜가 다르다")
    modes = {**{f"buffer_{m}": m for m in BUFFER_RUNS}, **{f"rl_s{s}": "full" for s in rl_positions}}
    start = perf_counter()
    g0 = {name: g0_violations(positions[name], tables["sector"], test_dates, risk, modes[name]) for name in positions}
    g0_seconds = perf_counter() - start
    summaries = {name: summary(rep, inputs.bm_close, positions[name]) for name, rep in replays.items()}
    rl_summaries = {s: summaries[f"rl_s{s}"] for s in rl_positions}
    g5 = g5_judgment(rl_summaries, summaries["buffer_full"], cfg["kpi"])
    table = g5_table(rl_summaries, summaries["buffer_full"])

    criteria = {
        "rl_run_complete": bool(rl_m["criteria"].get("b07_complete")),
        "m9_applied": bool(m9.record["applied"]),
        "every_test_date_in_positions": all(
            positions[f"buffer_{m}"]["date"].nunique() == len(test_dates) for m in BUFFER_RUNS),
        "buffer_replay_matches_run": all(replay_diff[f"buffer_{m}"] <= REPLAY_TOLERANCE for m in BUFFER_RUNS),
        "rl_replay_matches_record": all(replay_diff[f"rl_s{s}"] <= REPLAY_TOLERANCE for s in rl_positions),
        "g0_violations_zero": not any(g0.values()),
    }
    # 6.5절 B-07 완료 기준의 '버퍼 규칙과의 비교 표(G5)'와 '시드별 결과 표'가 같은 입력으로 만들어졌다
    criteria["g5_table_complete"] = all(criteria.values())

    commit = _git_commit()  # 결과 폴더·positions를 만들기 전에 (R-10)
    out.mkdir(parents=True)
    shutil.copyfile(cfg_path, out / "config.yaml")
    (out / "git_commit.txt").write_text(commit, encoding="utf-8")
    handoff = {}
    for mode, (with_m9, role) in BUFFER_RUNS.items():
        path = io.write_table("positions", runs[mode].positions, run_id=position_ids[mode])
        handoff[f"buffer_{mode}"] = {
            "run_id": position_ids[mode], "path": path.relative_to(io.ROOT).as_posix(), "sha256": _file_sha256(path),
            "rows": len(runs[mode].positions), "signal_dates": int(runs[mode].positions["date"].nunique()),
            "constructor": "buffer", "m8_mode": mode, "m9_alerts_run_id": m9.record["alerts_run_id"] if with_m9 else None,
            "role": role,
        }
    for seed in rl_positions:
        rec = rl_m["positions"][str(seed)]
        handoff[f"rl_s{seed}"] = {
            "run_id": f"{args.rl_run_id}_s{seed}", "path": rec["path"], "sha256": rec["sha256"], "rows": rec["rows"],
            "signal_dates": rec["signal_dates"], "constructor": "rl", "m8_mode": "full",
            "m9_alerts_run_id": m9.record["alerts_run_id"], "role": "G5 판정 대상 (RL + M8 + M9)",
        }

    table.to_csv(out / "g5_table.csv", index=False)
    pd.concat([weekly_rows(name, rep, inputs.bm_close.index) for name, rep in replays.items()],
              ignore_index=True).to_csv(out / "weekly_excess.csv", index=False)
    nav = pd.DataFrame({"KOSPI200": inputs.bm_close.reindex(days), **{n: r.nav for n, r in replays.items()}})
    nav.index = days.strftime("%Y-%m-%d")
    nav.rename_axis("date").to_csv(out / "daily_nav.csv")

    metrics = {
        "run_id": args.run_id,
        "task": "B-07",
        "criteria": criteria,
        "g5": g5,
        "g7_check": g7_check(summaries["buffer_full"], summaries["buffer_off"], cfg["kpi"]),
        "g4_values": {"note": "참고값. G4 판정은 A, 2배 비용과 노출 조절 제외 진단은 C-10",
                      "buffer_full": {k: summaries["buffer_full"][k] for k in ("annual_excess", "ir", "mdd")},
                      "buffer_no_exposure": {k: summaries["buffer_no_exposure"][k] for k in ("annual_excess", "ir", "mdd")},
                      "kospi200": benchmark_summary(inputs.bm_close, days)},
        "runs": summaries,
        "replay": {"tolerance": REPLAY_TOLERANCE, "max_abs_diff": replay_diff,
                   "note": "buffer는 실행 때의 a_k·회전율, rl은 RL 기록 weekly_excess.csv의 a_k와 비교"},
        "g0": {"function": "src.backtest.validation.validate_g0_positions",
               "violations": {n: len(v) for n, v in g0.items()}, "violation_dates": {n: v[:10] for n, v in g0.items() if v}},
        "handoff": {
            "note": "C 요청 (#58, 11/16): 실행 run_id, M8 모드, M9 경보 run_id, 원본 파일 지문",
            "positions": handoff,
            "inputs": fingerprints,
            "test_signals": signal_record,
            "m9": m9.record,
            "costs": dict(cfg["costs"]),
            "commit": commit.splitlines()[0].strip(),
        },
        "rl_run": {"run_id": args.rl_run_id, "criteria": rl_m["criteria"], "ir_mean_over_seeds": rl_m["ir"]["mean_over_seeds"]},
        "test": {"years": years, "signal_dates": len(test_dates), "first": test_dates[0], "last": test_dates[-1],
                 "start": "첫 시험 신호일에 현금 100%",
                 "period": {"first_fill": _next_trading_day(inputs.bm_close.index, test_dates[0]),
                            "last_signal_fill": _next_trading_day(inputs.bm_close.index, test_dates[-1]),
                            "end": last_period[1],
                            "note": "명세 판정 기간·design_C 5절: 마지막 시험 신호일 목표도 체결해 그 해 마지막 거래일 "
                                    "종가까지. 잔여 구간은 NAV·CAGR·MDD·연 회전율에 넣고 a_k·IR에서는 뺀다"},
                 "nav": "첫 체결일 체결 직전 NAV 1에서 시작, 날마다 종가 NAV (daily_nav.csv는 첫 체결일 체결 뒤부터). "
                        "MDD는 체결 후 일간 NAV"},
        "metric_functions": {"ir": "information_ratio", "cagr": "cagr", "annual_excess": "annual_excess_return",
                             "mdd": "max_drawdown", "annual_turnover": "annual_turnover", "module": "src.backtest.metrics"},
        "generated": {"features": {"rows": len(features), "sha256": content_sha256(features)}},
        "seconds": {"prepare": prepare_seconds, "runs": run_seconds, "g0": g0_seconds},
        "versions": _versions(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    print(f"B-07 G5 비교: IR RL 평균 {g5['rl_mean_over_seeds']['ir']:.3f} vs 버퍼 {g5['buffer']['ir']:.3f}, "
          f"G5 통과 {g5['pass']}, 완료 기준 {criteria['g5_table_complete']} → {out}")
    return out


if __name__ == "__main__":
    main()
