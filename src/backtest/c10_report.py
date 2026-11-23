"""C-10: pinned v7 period, stress-window and G4 exposure diagnostics.

The 2x-cost replay is run only when B's exact market and M9 alert inputs are
available. Missing M9 events must never be interpreted as an empty schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import metrics as m5
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.params import load_prereg


PORTFOLIOS = ("buffer_full", "buffer_no_exposure", "buffer_off")
MARKET_TABLES = ("prices", "status", "index", "delisted", "delisted_reused")


@dataclass(frozen=True)
class Replay:
    excess: np.ndarray
    turnover: np.ndarray
    nav: pd.Series
    last_turnover: float


def _period_end(calendar: pd.DatetimeIndex, last_signal: str) -> tuple[str, str]:
    days = pd.DatetimeIndex(calendar).sort_values()
    end = days[days.year == 2025][-1]
    after = days[days > pd.Timestamp(last_signal)]
    if not len(after) or after[0] >= end:
        raise ValueError("마지막 목표를 체결할 평가일이 없습니다")
    return days[days.get_loc(end) - 1].strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _replay(positions: pd.DataFrame, dates: list[str], engine: Any,
            last_period: tuple[str, str]) -> Replay:
    if not engine.record_ledger or engine.ledger:
        raise ValueError("빈 원장을 가진 엔진이 필요합니다")
    targets = {}
    for day, frame in positions.groupby(pd.to_datetime(positions["date"]).dt.strftime("%Y-%m-%d")):
        selected = frame.loc[frame["ticker"] != "CASH"]
        targets[day] = {str(t): float(w) for t, w in zip(selected["ticker"], selected["target_weight"]) if w > 0}
    if sorted(targets) != dates:
        raise ValueError("비중의 신호일이 입력과 다릅니다")
    state, excess, turnover = engine.initial_state(), [], []
    for day, next_day in zip(dates[:-1], dates[1:]):
        result = engine.run_period(state, targets[day], day, next_day)
        state = result.state
        excess.append(result.r_net - result.r_bm)
        turnover.append(result.turnover)
    result = engine.run_period(state, targets[dates[-1]], dates[-1], last_period[0])
    frame = pd.DataFrame({"date": [row["date"] for row in engine.ledger],
                          "nav": [float(row["nav_after"]) for row in engine.ledger]})
    nav = frame.groupby("date", sort=False)["nav"].last()
    nav.index = pd.DatetimeIndex(nav.index)
    if not nav.index.is_monotonic_increasing or nav.index[-1].strftime("%Y-%m-%d") != last_period[1]:
        raise ValueError("재생 NAV의 종료일이 다릅니다")
    if abs(float(nav.iloc[-1]) - result.nav_end) > 1e-9:
        raise ValueError("원장 최종 NAV와 엔진 반환값이 다릅니다")
    return Replay(np.asarray(excess, dtype=float), np.asarray(turnover, dtype=float), nav, float(result.turnover))


def _summary(rep: Replay, bm: pd.Series, positions: pd.DataFrame) -> dict:
    days = rep.nav.index
    n = len(days) - 1
    nav = np.r_[1.0, rep.nav.to_numpy()[1:]]
    benchmark = bm.reindex(days).to_numpy(dtype=float)
    return {
        "ir": m5.information_ratio(rep.excess),
        "cagr": m5.cagr(nav, trading_days=n),
        "annual_excess": m5.annual_excess_return(nav, benchmark, trading_days=n),
        "mdd": m5.max_drawdown(rep.nav.to_numpy()),
        "annual_turnover": m5.annual_turnover(np.r_[rep.turnover, rep.last_turnover], trading_days=n),
        "mean_cash": float(positions.loc[positions["ticker"] == "CASH", "target_weight"].mean()),
        "weeks": len(rep.excess), "trading_days": n,
        "first_fill": days[0].strftime("%Y-%m-%d"),
        "last_day": days[-1].strftime("%Y-%m-%d"),
        "final_nav": float(rep.nav.iloc[-1]),
        "last_turnover": rep.last_turnover,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window(nav: pd.DataFrame, start: str, end: str, portfolio: str,
            weekly: pd.DataFrame | None = None) -> dict:
    """Use prior close as base; for the first fill use pre-trade NAV 1."""
    dates = nav.index[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
    if len(dates) < 2:
        raise ValueError(f"{start}~{end}: 평가 거래일이 부족합니다")
    before = nav.index[nav.index < dates[0]]
    base_date = before[-1] if len(before) else dates[0]
    initial = dates[0] == nav.index[0]
    port_base = 1.0 if initial else float(nav.at[base_date, portfolio])
    bm_base = float(nav.at[base_date, "KOSPI200"])
    port_values = np.r_[port_base, nav.loc[dates, portfolio].to_numpy(dtype=float)]
    bm_values = np.r_[bm_base, nav.loc[dates, "KOSPI200"].to_numpy(dtype=float)]
    if initial:
        # The official full-period CAGR has the first fill's pre-trade NAV 1
        # at the first close and counts returns from the following close.
        port_values = np.r_[1.0, nav.loc[dates[1:], portfolio].to_numpy(dtype=float)]
        bm_values = nav.loc[dates, "KOSPI200"].to_numpy(dtype=float)
        n = len(dates) - 1
        dd_values = nav.loc[dates, portfolio].to_numpy(dtype=float)
    else:
        n = len(dates)
        dd_values = port_values
    port_return = float(port_values[-1] / port_values[0] - 1)
    bm_return = float(bm_values[-1] / bm_values[0] - 1)
    result = {
        "portfolio": portfolio, "first_day": dates[0].strftime("%Y-%m-%d"),
        "last_day": dates[-1].strftime("%Y-%m-%d"),
        "base_day": "first_fill_pretrade" if initial else base_date.strftime("%Y-%m-%d"),
        "daily_returns": n, "return": port_return, "bm_return": bm_return,
        "cagr": m5.cagr(port_values, trading_days=n),
        "bm_cagr": m5.cagr(bm_values, trading_days=n),
        "annual_excess": m5.annual_excess_return(port_values, bm_values, trading_days=n),
        "mdd": m5.max_drawdown(dd_values),
        "bm_mdd": m5.max_drawdown(bm_values),
        "capture_ratio": m5.capture_ratio(port_return, bm_return),
    }
    if weekly is not None:
        samples = weekly.loc[(weekly["portfolio"] == portfolio)
                             & (weekly["week"] >= dates[0]) & (weekly["week"] <= dates[-1]), "a_k"]
        result["weeks"] = len(samples)
        result["ir"] = m5.information_ratio(samples.to_numpy(dtype=float))
    return result


def _cost_replay(root: Path, market: Path, alerts_path: Path, handoff: dict, cfg: dict,
                 official_nav: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    from src.events.alerts import AlertBook

    expected = handoff["inputs"]
    files = [market / f"{name}.parquet" for name in MARKET_TABLES]
    for name, path in zip(MARKET_TABLES, files):
        if not path.is_file() or sha256(path) != expected[name]["sha256"]:
            raise ValueError(f"B 인계와 시장 입력 지문이 다릅니다: {path}")
    if sha256(alerts_path) != handoff["m9"]["sha256"]:
        raise ValueError("B 인계와 M9 경보 지문이 다릅니다")
    inputs = engine_inputs(*(pd.read_parquet(p) for p in files))
    alerts = pd.read_parquet(alerts_path)
    book = AlertBook(alerts, inputs.bm_close.index, int(cfg["events"]["cooldown_trading_days"]),
                     cfg["calendar"]["signal_cutoff_time"])
    positions_rec = handoff["positions"]["buffer_full"]
    position_path = root / positions_rec["path"]
    positions = pd.read_parquet(position_path)
    dates = sorted(pd.to_datetime(positions["date"]).dt.strftime("%Y-%m-%d").unique())
    if len(dates) != 520:
        raise ValueError("B 인계와 신호일 개수가 다릅니다")
    cooldowns, liquidations = book.engine_alerts(dates[0], dates[-1])
    last = _period_end(inputs.bm_close.index, dates[-1])
    kwargs = {"cooldowns": cooldowns, "event_liquidations": liquidations, "record_ledger": True}
    base = _replay(positions, dates, build_engine(inputs, cfg, **kwargs), last)
    benchmark = inputs.bm_close.reindex(base.nav.index)
    if not base.nav.index.equals(official_nav.index) or not np.allclose(
            base.nav.to_numpy(), official_nav["buffer_full"].to_numpy(), rtol=0, atol=1e-10):
        raise ValueError("1배 비용 재생 NAV가 B의 공식 장부와 일치하지 않습니다")
    doubled = _replay(positions, dates, build_engine(inputs, cfg, stress=True, **kwargs), last)
    result = _summary(doubled, benchmark, positions)
    result["excess_positive"] = result["annual_excess"] > 0
    result["base_nav_max_abs_diff"] = float(np.max(np.abs(
        base.nav.to_numpy() - official_nav["buffer_full"].to_numpy())))
    result["market_sha256"] = {name: sha256(path) for name, path in zip(MARKET_TABLES, files)}
    result["alerts_sha256"] = sha256(alerts_path)
    return result, pd.DataFrame({"date": doubled.nav.index, "buffer_full_2x": doubled.nav.to_numpy()})


def run(root: Path, out: Path, *, market: Path | None = None, alerts: Path | None = None) -> dict:
    if out.exists():
        raise FileExistsError(f"결과를 덮어쓰지 않습니다: {out}")
    if (market is None) != (alerts is None):
        raise ValueError("2배 비용 재생에는 시장 입력 폴더와 M9 경보 파일이 모두 필요합니다")
    b = root / "reports" / "20261118_B_buffer_wf"
    cfg_path = root / "configs" / "prereg.yaml"
    cfg = load_prereg(cfg_path)
    handoff = json.loads((b / "metrics.json").read_text(encoding="utf-8"))["handoff"]
    nav = pd.read_csv(b / "daily_nav.csv", parse_dates=["date"]).set_index("date")
    weekly = pd.read_csv(b / "weekly_excess.csv", parse_dates=["week"])
    if len(nav) != 2447 or nav.index.max().year != 2025 or nav.index.has_duplicates:
        raise ValueError("고정 v7 장부의 기간·행 수가 다르거나 홀드아웃이 섞였습니다")
    if len(weekly.loc[weekly["portfolio"] == "buffer_full"]) != 519:
        raise ValueError("고정 v7 주간 표본 519개가 필요합니다")
    input_hashes = {p.name: sha256(p) for p in (b / "metrics.json", b / "daily_nav.csv", b / "weekly_excess.csv", cfg_path)}
    for name in PORTFOLIOS:
        rec = handoff["positions"][name]
        path = root / rec["path"]
        if not path.is_file() or sha256(path) != rec["sha256"]:
            raise ValueError(f"B 인계 비중 파일이 없거나 지문이 다릅니다: {name}")
        input_hashes[rec["path"]] = rec["sha256"]
    periods = []
    for name, (start, end) in cfg["analysis_periods"].items():
        if name == "H":
            continue
        for portfolio in PORTFOLIOS:
            periods.append({"period": name, **_window(nav, start, end, portfolio, weekly)})
    stress = []
    for name in ("S1", "S2"):
        start, end = cfg["stress_windows"][name]
        for portfolio in PORTFOLIOS:
            stress.append({"window": name, **_window(nav, start, end, portfolio)})
    overall = {name: _window(nav, "2016-01-01", "2025-12-31", name, weekly) for name in PORTFOLIOS}
    for name in ("buffer_full", "buffer_no_exposure"):
        reference = json.loads((b / "metrics.json").read_text(encoding="utf-8"))["g4_values"][name]
        for key, value in reference.items():
            if key in overall[name] and abs(overall[name][key] - value) > 1e-10:
                raise ValueError(f"{name}의 {key}가 B의 G4 확인값과 다릅니다")
    kpi = cfg["kpi"]["G4"]
    full = overall["buffer_full"]
    checks = {
        "annual_excess": full["annual_excess"] >= float(kpi["min_excess_cagr"]),
        "ir": full["ir"] is not None and full["ir"] >= float(kpi["min_ir"]),
        "mdd": full["mdd"] <= full["bm_mdd"] + float(kpi["max_mdd_gap_vs_bm"]),
    }
    cost = None
    cost_nav = None
    if market is not None and alerts is not None:
        cost, cost_nav = _cost_replay(root, market, alerts, handoff, cfg, nav)
        checks["double_cost_excess"] = cost["excess_positive"]
    result = {
        "run_id": out.name, "classification": "C-10 preregistered 2016-2025 v7; no 2026 holdout",
        "input_sha256": input_hashes, "overall": overall,
        "g4_checks": checks, "g4_passed": all(checks.values()) if cost is not None else False,
        "double_cost": cost, "double_cost_status": "complete" if cost is not None else "pending_exact_M9_alert_file",
        "period_rows": len(periods), "stress_rows": len(stress),
    }
    out.mkdir(parents=True)
    shutil.copyfile(cfg_path, out / "config.yaml")
    (out / "git_commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() + "\n")
    pd.DataFrame(periods).to_csv(out / "periods.csv", index=False)
    pd.DataFrame(stress).to_csv(out / "stress_windows.csv", index=False)
    if cost_nav is not None:
        cost_nav.to_csv(out / "double_cost_nav.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--market", type=Path)
    parser.add_argument("--alerts", type=Path)
    args = parser.parse_args()
    result = run(args.root, args.out, market=args.market, alerts=args.alerts)
    print(json.dumps({"g4_checks": result["g4_checks"], "double_cost_status": result["double_cost_status"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
