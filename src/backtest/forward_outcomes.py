"""M7 확정 신호의 사후 ER·실제 포트폴리오 성과 (C-04)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.engine import ExecutionEngine
from src.common.guards import DataAccessPolicy, assert_outcome_dates_allowed


@dataclass(frozen=True)
class ForwardOutcome:
    mean_er_top20: float | None
    mean_er_flagged: float | None
    portfolio_return: float | None
    bm_return: float | None
    missing_top20: int
    missing_flagged: int


def evaluate_forward_record(
    *,
    engine: ExecutionEngine,
    record: Mapping[str, object],
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    forward_log: pd.DataFrame,
    recorded_targets: Mapping[str, Mapping[str, float]],
    run_at: datetime,
    policy: DataAccessPolicy,
    allow_holdout: bool = False,
    horizon: int = 20,
) -> ForwardOutcome:
    """기록 당시 신호·비중을 바꾸지 않고, t+1+h 이후 사후 열을 계산한다.

    ``forward_log``는 모든 기록일과 ``has_positions``를 제공한다. D는
    각 날짜의 ``ft-{YYYYMMDD}`` 태그에 있는 원본 비중을
    ``recorded_targets``로 전달한다. 비중 기록이 하나라도 누락되면
    실제 NAV를 임의로 이어 붙이지 않고 오류를 낸다.
    """
    if (not isinstance(run_at, datetime) or run_at.tzinfo is None or run_at.utcoffset() is None
            or type(horizon) is not int or horizon < 1):
        raise ValueError("run_at은 시간대가 있는 시각, horizon은 양의 정수여야 합니다")
    required_record = {"signal_date", "label_ready_date", "has_positions", "commit_hash", "signal_source"}
    if required_record - set(record):
        raise ValueError(f"forward 기록 열이 없습니다: {sorted(required_record - set(record))}")
    signal_day = pd.Timestamp(record["signal_date"])
    signal_date = signal_day.strftime("%Y-%m-%d")
    if not {"signal_date", "label_ready_date", "has_positions", "commit_hash", "signal_source"}.issubset(forward_log):
        raise ValueError("forward_log 필수 열이 없습니다")
    log = forward_log.copy()
    log["signal_date"] = pd.to_datetime(log["signal_date"]).dt.strftime("%Y-%m-%d")
    if log["signal_date"].duplicated().any():
        raise ValueError("forward_log에 중복 신호일이 있습니다")
    original = log.loc[log["signal_date"] == signal_date]
    if len(original) != 1 or any(
        str(original.iloc[0][key]) != str(record[key])
        for key in ("commit_hash", "signal_source", "has_positions")
    ) or pd.Timestamp(original.iloc[0]["label_ready_date"]) != pd.Timestamp(record["label_ready_date"]):
        raise ValueError("forward 기록이 원본 로그와 다릅니다")
    run_kst = run_at.astimezone(ZoneInfo("Asia/Seoul"))
    assert_outcome_dates_allowed([signal_date], policy=policy, allow_holdout=allow_holdout,
                                 run_date=run_kst.date())
    if signal_date not in engine.day_index:
        raise ValueError("신호일이 거래일 목록에 없습니다")
    signal_i = engine.day_index[signal_date]
    end_i = signal_i + 1 + horizon
    if end_i >= len(engine.days):
        raise ValueError("라벨 종료 거래일까지 가격이 필요합니다")
    end_date = engine.days[end_i]
    ready_date = pd.Timestamp(record["label_ready_date"]).strftime("%Y-%m-%d")
    if ready_date != end_date:
        raise ValueError("forward 기록의 label_ready_date가 KRX 거래일 기준과 다릅니다")
    if run_kst <= pd.Timestamp(f"{end_date} 15:30:00", tz="Asia/Seoul"):
        raise ValueError("라벨 종료일 장 마감 전에는 사후 성과를 계산할 수 없습니다")
    if not {"date", "ticker", "rank", "crash_flag"}.issubset(signals):
        raise ValueError("signals 필수 열이 없습니다")
    if not {"date", "ticker", "h", "er"}.issubset(labels):
        raise ValueError("labels 필수 열이 없습니다")
    day_signals = signals.loc[pd.to_datetime(signals["date"]) == signal_day].copy()
    day_labels = labels.loc[(pd.to_datetime(labels["date"]) == signal_day) & (labels["h"] == horizon),
                            ["ticker", "er"]].copy()
    if day_signals.empty or day_signals["ticker"].duplicated().any() or day_labels["ticker"].duplicated().any():
        raise ValueError("신호일 신호가 없거나 신호·라벨에 중복 종목이 있습니다")
    if not day_signals["crash_flag"].isin([0, 1]).all() or not day_signals["rank"].between(1, len(day_signals)).all():
        raise ValueError("crash_flag 또는 순위가 잘못됐습니다")
    if day_signals["rank"].duplicated().any():
        raise ValueError("신호일 순위가 중복됐습니다")
    joined = day_signals.merge(day_labels, on="ticker", how="left", validate="one_to_one")
    joined["er"] = pd.to_numeric(joined["er"], errors="coerce")

    def _mean(group: pd.DataFrame) -> tuple[float | None, int]:
        valid = group["er"].notna() & group["er"].map(lambda value: isfinite(value) if pd.notna(value) else False)
        missing = int((~valid).sum())
        return (float(group["er"].mean()) if len(group) and not missing else None), missing

    top20 = joined.loc[joined["rank"] <= 20]
    flagged = joined.loc[joined["crash_flag"] == 1]
    mean_top20, missing_top20 = _mean(top20)
    mean_flagged, missing_flagged = _mean(flagged)
    if len(top20) != min(20, len(joined)):
        raise ValueError("상위 20종목 순위가 불완전합니다")

    if record["has_positions"] not in (0, 1):
        raise ValueError("has_positions는 0/1이어야 합니다")
    if record["has_positions"] == 0:
        return ForwardOutcome(mean_top20, mean_flagged, None, None, missing_top20, missing_flagged)

    # t+21 종가의 NAV는 그날 새 신호의 체결 *직전* 값이다. 종료일에
    # 체결될 목표(전 신호일 기록)는 사후 수익에 포함하지 않는다.
    known = set(log.loc[(log["has_positions"] == 1) & (log["signal_date"] <= end_date), "signal_date"])
    expected = sorted(day for day in known if day in engine.day_index
                      and engine.day_index[day] + 1 < end_i)
    if not expected or signal_date not in expected:
        raise ValueError("평가 대상의 positions 기록이 없습니다")
    if not set(expected).issubset(recorded_targets) or not set(recorded_targets).issubset(known):
        raise ValueError("기록된 positions와 제공된 목표 비중 날짜가 다릅니다")
    state = engine.initial_state()
    nav_start = None
    nav_end = None
    for i, day in enumerate(expected):
        next_day = expected[i + 1] if i + 1 < len(expected) else end_date
        if next_day == day:
            continue
        if next_day < day or day not in engine.day_index:
            raise ValueError("positions 날짜가 거래일 목록과 맞지 않습니다")
        result = engine.run_to_signal_close(state, recorded_targets[day], day, next_day)
        if day == signal_date:
            nav_start = result.nav_start
        state = result.state
        nav_end = result.nav_end
    if nav_start is None or nav_end is None or nav_start <= 0:
        raise ValueError("평가 기간의 실제 NAV를 재생하지 못했습니다")
    first_i = signal_i + 1
    return ForwardOutcome(
        mean_top20, mean_flagged, nav_end / nav_start - 1.0,
        float(engine.bm[end_i] / engine.bm[first_i] - 1.0), missing_top20, missing_flagged,
    )
