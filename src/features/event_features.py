"""이벤트 피처 `ev_*` (명세서 6.2절). PM 소유.

신호일 t의 U_t 종목마다, 접수일이 거래일 기준 t−19~t이고 `available_at < t 15:30 KST`인 해당 유형 공시가
있으면 1, 없으면 0이다 (DEC-09). 유형은 `features.event_keywords`로 정하고 `events.title_filters`의
후속 정정·철회·취소는 뺀다 (Q-13). 자회사 제외는 T4·T5 전용 규칙이라 적용하지 않는다.
종목 연결은 M9와 같다 (DEC-09 (2), `link_tickers`).

공시 보관 구간 밖의 신호일은 0으로 채우지 않고 오류를 낸다. 0은 '공시가 없었다'는 뜻이기 때문이다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from src.events.alerts import KST, LINKED, link_tickers
from src.events.triggers import common_exclusion, strip_prefixes

EVENT_COLUMNS = ["feature", "ticker", "rcept_date", "available_at"]


def event_names(title_norm: str, keywords: Mapping[str, Sequence[str]], filters: Mapping[str, Any]) -> list[str]:
    """보고서명 하나가 해당하는 `ev_*` 이름. 후속 정정·철회·취소면 빈 목록이다."""
    title = strip_prefixes(title_norm, filters["strip_prefixes"])
    if common_exclusion(title, filters):
        return []
    return [name for name, words in keywords.items() if any(word in title for word in words)]


def tag_events(disclosures: pd.DataFrame, tickers: pd.DataFrame, keywords: Mapping[str, Sequence[str]],
               filters: Mapping[str, Any]) -> pd.DataFrame:
    """종목에 연결된 `ev_*` 공시를 한 행에 한 유형씩 돌려준다. `disclosures`에는 `stock_code` 열이 있어야 한다."""
    linked = link_tickers(disclosures, tickers)
    linked = linked[linked["link_status"] == LINKED]
    names = {title: event_names(title, keywords, filters) for title in linked["report_nm_norm"].unique()}
    tagged = linked.assign(feature=linked["report_nm_norm"].map(names))
    return tagged[EVENT_COLUMNS].explode("feature").dropna(subset=["feature"]).reset_index(drop=True)


def compute_event_features(events: pd.DataFrame, universe: pd.DataFrame, trading_days: Iterable,
                           names: Sequence[str], coverage: tuple, window_days: int = 20,
                           cutoff_time: str = "15:30") -> pd.DataFrame:
    """신호일별 U_t 전 종목의 `date, ticker, ev_*`(0 또는 1).

    coverage: 공시를 빠짐없이 보관한 접수일 구간 (시작, 끝). 창이 이 구간을 벗어나는 신호일이 있으면 ValueError.
    """
    days = pd.DatetimeIndex(sorted(pd.to_datetime(list(trading_days))))
    members = universe.loc[universe["in_universe"], ["date", "ticker"]]
    if members.duplicated().any():
        raise ValueError("universe에 같은 (date, ticker)가 두 번 있다")
    signals = sorted(pd.to_datetime(members["date"].unique()))
    starts = {t: _window_start(days, t, window_days) for t in signals}
    _check_coverage(starts, coverage)

    cutoff = pd.Timedelta(hours=int(cutoff_time[:2]), minutes=int(cutoff_time[3:]))
    received = pd.to_datetime(events["rcept_date"])
    available = pd.to_datetime(events["available_at"], utc=True)
    frames = []
    for t, group in members.groupby("date", sort=True):
        t = pd.Timestamp(t)
        known = (received >= starts[t]) & (received <= t) & (available < t.tz_localize(KST) + cutoff)
        hits = events.loc[known, ["feature", "ticker"]]
        frame = group.copy()
        for name in names:
            frame[name] = frame["ticker"].isin(hits.loc[hits["feature"] == name, "ticker"]).astype("int8")
        frames.append(frame.sort_values("ticker"))
    columns = ["date", "ticker", *names]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def build_event_features(disclosures: pd.DataFrame, tickers: pd.DataFrame, universe: pd.DataFrame,
                         trading_days: Iterable, cfg: Mapping[str, Any], coverage: tuple) -> pd.DataFrame:
    """사전등록 설정(`configs/prereg.yaml` 내용)으로 `ev_*` 표를 만든다."""
    keywords = cfg["features"]["event_keywords"]
    events = tag_events(disclosures, tickers, keywords, cfg["events"]["title_filters"])
    return compute_event_features(events, universe, trading_days, list(keywords), coverage,
                                  int(cfg["features"]["event_window_days"]), cfg["calendar"]["signal_cutoff_time"])


def _window_start(days: pd.DatetimeIndex, t: pd.Timestamp, window_days: int) -> pd.Timestamp:
    position = int(days.searchsorted(t))
    if position >= len(days) or days[position] != t:
        raise ValueError(f"신호일 {t.date()}이 거래일 목록에 없다")
    if position < window_days - 1:
        raise ValueError(f"신호일 {t.date()} 앞에 거래일이 {window_days - 1}개보다 적다")
    return days[position - window_days + 1]


def _check_coverage(starts: Mapping[pd.Timestamp, pd.Timestamp], coverage: tuple) -> None:
    first, last = pd.Timestamp(coverage[0]), pd.Timestamp(coverage[1])
    outside = [t for t, start in starts.items() if start < first or t > last]
    if outside:
        raise ValueError(f"공시 보관 구간 {first.date()}~{last.date()} 밖 신호일 {len(outside)}개: "
                         f"{outside[0].date()}~{outside[-1].date()}. 공시를 더 수집하거나 신호일을 줄여야 한다")
