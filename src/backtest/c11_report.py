"""C-11의 2026 후보 비중 재생·현금 보정·실행 기록 구성요소.

시장·경보·후보 파일의 인계가 끝나면 실행기는 먼저 ``check_open``을
호출하고, 각 후보마다 새로운 C 엔진을 만들어 ``replay_candidate``를
호출한다. 원본 현금 0% NAV와 CD 91일물 보정 NAV를 모두 보존한다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from datetime import datetime, date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.backtest import metrics as m5
from src.backtest.cash_return import cash_fraction, cd91_adjust
from src.common.guards import DataAccessPolicy, assert_outcome_dates_allowed

KST = ZoneInfo("Asia/Seoul")


def _monthly_for_nav(monthly: pd.Series, nav: pd.Series) -> tuple[pd.Series, list[str]]:
    """발표 전 월평균이 없는 마지막 한 달만 직전 월 금리로 연장한다."""
    if not isinstance(monthly.index, pd.PeriodIndex) or monthly.index.freqstr != "M" or monthly.empty:
        raise ValueError("CD 91일물 금리는 월별 Series여야 합니다")
    needed = nav.index[-2].to_period("M")
    last = monthly.index.max()
    if needed in monthly.index:
        return monthly, []
    if needed != last + 1:
        raise KeyError(f"CD 91일물 금리가 없는 달: {needed}")
    extended = pd.concat([monthly, pd.Series([float(monthly.loc[last])],
                                              index=pd.PeriodIndex([needed], freq="M"))])
    return extended, [str(needed)]


def check_open(config: Mapping[str, Any], *, run_at: datetime, allow_holdout: bool) -> None:
    """2026 입력 파일을 열기 전에 DEC-38 개봉 가드를 검사한다."""
    if not isinstance(run_at, datetime) or run_at.tzinfo is None or run_at.utcoffset() is None:
        raise ValueError("실행 시각은 시간대가 있는 datetime이어야 합니다")
    split = config["splits"]
    start, end = (date.fromisoformat(x) for x in split["holdout_signal_range"])
    policy = DataAccessPolicy(date(2025, 12, 31), start, end,
                              date.fromisoformat(split["holdout_open_date"]))
    assert_outcome_dates_allowed([start, end], policy=policy,
                                 allow_holdout=allow_holdout, run_date=run_at.astimezone(KST).date())


def replay_candidate(positions: pd.DataFrame, engine: Any, monthly: pd.Series,
                     *, end_date: str, benchmark: pd.Series) -> tuple[dict[str, Any], pd.DataFrame]:
    """한 후보의 모든 신호를 같은 C 엔진으로 재생해 CD 기준 성과를 계산한다.

    ``benchmark``는 같은 거래일의 코스피 종합 가격이다. ``engine``은
    KOSPI200을 내부 기준으로 쓰는 기존 C 엔진이며 원장 기록을 켜야 한다.
    """
    required = {"date", "ticker", "target_weight"}
    if required - set(positions) or positions.empty:
        raise ValueError("후보 비중에 date, ticker, target_weight가 필요합니다")
    if not engine.record_ledger or engine.ledger:
        raise ValueError("원장이 비어 있는 record_ledger=True C 엔진이 필요합니다")
    frame = positions.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame["ticker"] = frame["ticker"].astype(str)
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("후보 비중에 중복된 신호일·종목이 있습니다")
    weights = pd.to_numeric(frame["target_weight"], errors="coerce")
    if not np.isfinite(weights.to_numpy()).all() or (weights < 0).any():
        raise ValueError("후보 비중은 유한한 0 이상이어야 합니다")
    frame["target_weight"] = weights.astype(float)
    by_day = frame.groupby("date", sort=True)
    if any("CASH" not in set(part["ticker"]) or
           abs(float(part["target_weight"].sum()) - 1.0) > 1e-9
           for _, part in by_day):
        raise ValueError("각 신호일에 CASH 행과 합계 1의 목표 비중이 필요합니다")
    targets = {day: {str(row.ticker): float(row.target_weight)
                     for row in part.itertuples() if row.ticker != "CASH" and row.target_weight > 0}
               for day, part in by_day}
    dates = sorted(targets)
    calendar = list(engine.days)
    if end_date not in engine.day_index or dates[0] not in engine.day_index:
        raise ValueError("평가 종료일 또는 첫 신호일이 거래일 달력에 없습니다")
    end_i = engine.day_index[end_date]
    if any(day not in engine.day_index or engine.day_index[day] >= end_i for day in dates):
        raise ValueError("신호일이 평가 구간의 거래일이어야 합니다")
    if end_i < 2 or engine.day_index[dates[-1]] + 1 > end_i:
        raise ValueError("마지막 신호의 체결일이 평가 종료일 뒤입니다")
    state = engine.initial_state()
    turns: list[float] = []
    for day, next_day in zip(dates[:-1], dates[1:]):
        result = engine.run_period(state, targets[day], day, next_day)
        state = result.state
        turns.append(float(result.turnover))
    last_fill_at_end = engine.day_index[dates[-1]] + 1 == end_i
    if last_fill_at_end:
        # B-09: 11/27 신호의 목표는 11/30 종료 종가에 체결하지 않는다.
        # 직전 구간이 이미 11/30 종가까지 평가했으므로 추가 비용도 없다.
        if len(dates) < 2:
            raise ValueError("마지막 목표가 미체결이면 앞선 신호가 필요합니다")
    else:
        result = engine.run_period(state, targets[dates[-1]], dates[-1], calendar[end_i - 1])
        turns.append(float(result.turnover))
    ledger = engine.ledger
    raw = pd.DataFrame({"date": [r["date"] for r in ledger],
                        "nav": [float(r["nav_after"]) for r in ledger]})
    nav = raw.groupby("date", sort=False)["nav"].last()
    nav.index = pd.DatetimeIndex(pd.to_datetime(nav.index)).as_unit("ns")
    if not nav.index.is_monotonic_increasing or nav.index[-1].strftime("%Y-%m-%d") != end_date:
        raise ValueError("엔진 원장이 평가 종료일까지 이어지지 않습니다")
    if abs(float(nav.iloc[-1]) - float(result.nav_end)) > 1e-9 * max(1.0, float(result.nav_end)):
        raise ValueError("엔진 반환 NAV와 원장 마지막 NAV가 다릅니다")
    fractions = cash_fraction(ledger)
    rates, filled_months = _monthly_for_nav(monthly, nav)
    adjusted = cd91_adjust(nav, fractions, rates)
    bm = benchmark.reindex(nav.index).to_numpy(dtype=float)
    if not np.isfinite(bm).all() or (bm <= 0).any():
        raise ValueError("코스피 종합에 평가일 가격이 빠져 있습니다")
    n = len(nav) - 1
    def stats(values: pd.Series) -> dict[str, float]:
        # B 판정 장부와 같은 첫 체결 직전 NAV=1·일간 수익률 n개 기준.
        start_nav = np.r_[1.0, values.to_numpy(dtype=float)[1:]]
        return {"cagr": m5.cagr(start_nav, trading_days=n),
                "mdd": m5.max_drawdown(values.to_numpy(dtype=float)),
                "annual_excess_vs_kospi": m5.annual_excess_return(start_nav, bm, trading_days=n),
                "final_nav": float(values.iloc[-1])}
    report = {"signal_days": len(dates), "executed_signals": len(turns),
              "last_signal_unexecuted_at_end": last_fill_at_end,
              "cd91_filled_months": filled_months,
              "first_signal": dates[0], "last_signal": dates[-1],
              "first_fill": nav.index[0].strftime("%Y-%m-%d"), "last_day": end_date,
              "trading_days": n, "annual_turnover": m5.annual_turnover(turns, trading_days=n),
              "cash0": stats(nav), "cd91": stats(adjusted)}
    daily = pd.DataFrame({"date": nav.index, "nav_cash0": nav.to_numpy(),
                          "nav_cd91": adjusted.to_numpy(), "cash_fraction": fractions.to_numpy(),
                          "kospi_close": bm})
    return report, daily


def write_run_record(out: Path, *, config_path: Path, inputs: Mapping[str, Path],
                     metrics: Mapping[str, Any], daily_nav: Mapping[str, pd.DataFrame],
                     run_at: datetime, allow_holdout: bool) -> None:
    """R-10 방식으로 결과·설정·입력 지문을 새 디렉터리에만 기록한다."""
    if not isinstance(run_at, datetime) or run_at.tzinfo is None or run_at.utcoffset() is None:
        raise ValueError("실행 시각은 시간대가 있는 datetime이어야 합니다")
    if any(not name or any(c in name for c in "/\\") for name in daily_nav):
        raise ValueError("후보 이름은 파일명으로 쓸 수 있어야 합니다")
    paths = {"prereg": config_path, **inputs}
    if not paths or any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("설정 또는 입력 파일이 없습니다")
    fingerprints = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            fingerprints[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, out / "config.yaml")
    record = {"run_at_kst": run_at.astimezone(KST).isoformat(),
              "allow_holdout": allow_holdout, "git_commit": commit, "git_dirty": dirty,
              "inputs_sha256": fingerprints, "cash_source": "FRED IR3TIB01KRM156N",
              "cash_method": "previous_day_cash_fraction_x_previous_monthly_rate_calendar_days",
              "metrics": metrics}
    (out / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                                 allow_nan=False), encoding="utf-8")
    for name, frame in daily_nav.items():
        frame.to_csv(out / f"daily_nav_{name}.csv", index=False)
