"""빅카인즈 일별 건수의 DEC-09 시점·결측 검증 (C 평가용)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from math import isfinite
from pathlib import Path

import pandas as pd
import yaml


PREREG_PATH = Path(__file__).resolve().parents[2] / "configs" / "prereg.yaml"


@dataclass(frozen=True)
class NewsWindows:
    recent: tuple[int, int]
    baseline: tuple[int, int]
    cutoff_time: time

    def __post_init__(self) -> None:
        a, b = self.baseline
        c, d = self.recent
        if not (a < b < c < d < 0 and b + 1 == c):
            raise ValueError("뉴스 기준·최근 창은 겹치지 않는 연속 과거 거래일이어야 합니다")


def load_news_windows(config_path: str | Path = PREREG_PATH) -> NewsWindows:
    """사전등록 YAML의 뉴스 창을 읽는다. 코드에 기간 기본값을 두지 않는다."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    try:
        features = config["features"]
        cutoff = config["calendar"]["signal_cutoff_time"]
        recent = features["news_window_offsets"]
        baseline = features["news_baseline_offsets"]
        if any(type(value) is not int for value in (*recent, *baseline)):
            raise ValueError("뉴스 창 오프셋은 정수여야 합니다")
        if len(recent) != 2 or len(baseline) != 2:
            raise ValueError("뉴스 창에는 시작·끝 오프셋이 필요합니다")
        if not isinstance(cutoff, str):
            raise ValueError("사전등록 신호 컷오프 시각은 문자열이어야 합니다")
        try:
            cutoff_time = time.fromisoformat(cutoff)
        except ValueError as exc:
            raise ValueError("사전등록 신호 컷오프 시각 형식이 잘못됐습니다") from exc
        if cutoff_time.tzinfo is not None:
            raise ValueError("사전등록 신호 컷오프 시각에는 시간대를 붙이지 않습니다")
        return NewsWindows(tuple(recent), tuple(baseline), cutoff_time)
    except (KeyError, TypeError) as exc:
        raise ValueError("사전등록 뉴스 창 설정이 없습니다") from exc


@dataclass(frozen=True)
class NewsCountWindow:
    spike: float | None
    missing_days: tuple[str, ...]
    late_days: tuple[str, ...]


def news_count_spike_at(
    news_daily: pd.DataFrame, trading_days: pd.DatetimeIndex,
    signal_date: str, ticker: str, *, windows: NewsWindows | None = None,
) -> NewsCountWindow:
    """사전등록된 기준·최근 창의 건수 비율을 누수 없이 계산한다.

    행이 없는 날짜만 수집 실패·미확인으로 취급한다. 저장된 행의
    ``news_count`` 결측은 스키마 위반이다. 주말·휴장일을 이전 거래일에
    합친 행의 ``available_at``도 실제 묶음의 마지막 날이어야 한다.
    """
    windows = windows or load_news_windows()
    if not isinstance(trading_days, pd.DatetimeIndex) or trading_days.has_duplicates or not trading_days.is_monotonic_increasing:
        raise ValueError("KRX 거래일은 중복 없는 오름차순 DatetimeIndex여야 합니다")
    day = pd.Timestamp(signal_date)
    if day not in trading_days:
        raise ValueError("신호일이 KRX 거래일에 없습니다")
    i = trading_days.get_loc(day)
    if i < -windows.baseline[0]:
        raise ValueError("신호일 이전 뉴스 기준 창의 거래일 이력이 부족합니다")
    baseline_days = trading_days[i + windows.baseline[0]:i + windows.baseline[1] + 1]
    recent_days = trading_days[i + windows.recent[0]:i + windows.recent[1] + 1]
    required = baseline_days.append(recent_days)
    if not {"date", "available_at", "ticker", "news_count"}.issubset(news_daily):
        raise ValueError("news_daily 필수 열이 없습니다")
    rows = news_daily.loc[news_daily["ticker"] == ticker].copy()
    rows["date"] = pd.to_datetime(rows["date"])
    if rows.duplicated("date").any():
        raise ValueError("news_daily에 중복 (date, ticker)가 있습니다")
    rows = rows.set_index("date")
    cutoff = pd.Timestamp(f"{day:%Y-%m-%d} {windows.cutoff_time.isoformat()}", tz="Asia/Seoul")
    late: list[str] = []
    missing: list[str] = []
    counts: list[int] = []
    for source_day in required:
        label = source_day.strftime("%Y-%m-%d")
        if source_day not in rows.index:
            missing.append(label)
            continue
        row = rows.loc[source_day]
        if pd.isna(row["news_count"]):
            raise ValueError(f"{label} news_count 결측 행은 스키마 위반입니다")
        value = row["news_count"]
        if isinstance(value, bool) or not isfinite(value) or int(value) != value or value < 0:
            raise ValueError(f"{label} 뉴스 건수는 0 이상의 정수여야 합니다")
        available = pd.Timestamp(row["available_at"])
        if pd.isna(available) or available.tzinfo is None or available.utcoffset() != timedelta(hours=9):
            raise ValueError(f"{label} available_at은 필수 KST 시각이어야 합니다")
        if available < pd.Timestamp(source_day, tz="Asia/Seoul"):
            raise ValueError(f"{label} available_at이 집계 날짜보다 빠릅니다")
        if available >= cutoff:
            late.append(label)
        counts.append(int(value))
    if missing or late:
        return NewsCountWindow(None, tuple(missing), tuple(late))
    baseline = sum(counts[:len(baseline_days)])
    recent = sum(counts[len(baseline_days):])
    scale = len(recent_days) / len(baseline_days)
    return NewsCountWindow((recent + 1) / (baseline * scale + 1), (), ())
