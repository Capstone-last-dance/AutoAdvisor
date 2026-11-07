"""빅카인즈 일별 건수의 DEC-09 시점·결측 검증 (C 평가용)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import isfinite

import pandas as pd


@dataclass(frozen=True)
class NewsCountWindow:
    spike: float | None
    missing_days: tuple[str, ...]
    late_days: tuple[str, ...]


def news_count_spike_at(
    news_daily: pd.DataFrame, trading_days: pd.DatetimeIndex,
    signal_date: str, ticker: str,
) -> NewsCountWindow:
    """t−5~t−1 / t−65~t−6의 건수 비율을 누수 없이 계산한다.

    행이 없는 날짜는 수집 실패·미확인으로 취급한다. 성공한 0건 행과
    다르며, 창의 하루라도 없거나 신호 컷오프 이후 이용 가능하면
    결과를 미산출로 둔다. 주말·휴장일을 이전 거래일에 합친 행의
    ``available_at``도 실제 묶음의 마지막 날이어야 한다.
    """
    if not isinstance(trading_days, pd.DatetimeIndex) or trading_days.has_duplicates or not trading_days.is_monotonic_increasing:
        raise ValueError("KRX 거래일은 중복 없는 오름차순 DatetimeIndex여야 합니다")
    day = pd.Timestamp(signal_date)
    if day not in trading_days:
        raise ValueError("신호일이 KRX 거래일에 없습니다")
    i = trading_days.get_loc(day)
    if i < 65:
        raise ValueError("신호일 이전 65거래일의 뉴스 이력이 필요합니다")
    required = trading_days[i - 65:i]
    if not {"date", "available_at", "ticker", "news_count"}.issubset(news_daily):
        raise ValueError("news_daily 필수 열이 없습니다")
    rows = news_daily.loc[news_daily["ticker"] == ticker].copy()
    rows["date"] = pd.to_datetime(rows["date"])
    if rows.duplicated("date").any():
        raise ValueError("news_daily에 중복 (date, ticker)가 있습니다")
    rows = rows.set_index("date").reindex(required)
    cutoff = pd.Timestamp(f"{day:%Y-%m-%d} 15:30:00", tz="Asia/Seoul")
    late: list[str] = []
    missing: list[str] = []
    counts: list[int] = []
    for source_day, row in rows.iterrows():
        label = source_day.strftime("%Y-%m-%d")
        if pd.isna(row["news_count"]):
            missing.append(label)
            continue
        value = row["news_count"]
        if isinstance(value, bool) or not isfinite(value) or int(value) != value or value < 0:
            raise ValueError(f"{label} 뉴스 건수는 0 이상의 정수여야 합니다")
        available = pd.Timestamp(row["available_at"])
        if pd.isna(available) or available.tzinfo is None or available.utcoffset() != timedelta(hours=9):
            raise ValueError(f"{label} available_at은 필수 KST 시각이어야 합니다")
        if available >= cutoff:
            late.append(label)
        counts.append(int(value))
    if missing or late:
        return NewsCountWindow(None, tuple(missing), tuple(late))
    baseline = sum(counts[:60])
    recent = sum(counts[60:])
    return NewsCountWindow((recent + 1) / (baseline * 5 / 60 + 1), (), ())
