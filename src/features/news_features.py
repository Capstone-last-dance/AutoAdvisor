"""뉴스 피처 `news_cnt_spike`, `news_sent_5` (명세서 6.2절, DEC-06). PM 소유.

입력은 빅카인즈 제목 범위 분석 대상 기사를 종목에 붙인 표다(한 행 = 기사 하나·종목 하나, 열 date·ticker·sentiment).

- 날짜만 아는 자료라 신호일 t의 기사는 쓰지 않는다 (DEC-09 (1)). 창은 거래일 t−5~t−1, 기준 창은 t−65~t−6이다
  (`features.news_window_offsets`, `features.news_baseline_offsets`).
- 주말·휴장일 기사는 직전 거래일에 더한다. 그래서 거래일 창 [t+a, t+b]는 달력일 [t+a 거래일, t+b+1 거래일의 전날]과
  같다. 최근 창의 끝은 t의 전날이고, t 전날까지의 기사를 모두 쓴다.
- news_cnt_spike = (최근 창 건수 + 1) / (기준 창 건수 × 5/60 + 1). 5와 60은 두 창의 거래일 수다.
- news_sent_5 = 최근 창 기사 감성 점수의 평균. 기사가 없으면 결측이다.
- 건수를 구하지 못한 종목·구간은 0이 아니라 결측이다 (DEC-06 (4)). 수집하지 않은 종목, 수집 구간 밖, 현재 상호를 쓰기
  전 구간(옛 상호 기사는 세지 못한다)이 여기에 든다. 창의 달력일이 하나라도 이 범위 밖이면 그 피처는 결측이다.
- 순위 변환과 결측 처리는 A와 같은 규칙(`price_features.rank_and_fill`)으로 신호일별 U_t 안에서 한다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.features.price_features import rank_and_fill

NEWS_FEATURES = ["news_cnt_spike", "news_sent_5"]
OUTPUT_COLUMNS = ["date", "ticker", "news_cnt_spike", "news_cnt_spike_na", "news_sent_5", "news_sent_5_na"]
ONE_DAY = pd.Timedelta(days=1)


def compute_news_features(articles: pd.DataFrame, universe: pd.DataFrame, trading_days: Iterable,
                          collected: Iterable[str], coverage: tuple,
                          covered_from: Mapping[str, Any] | None = None,
                          window: tuple[int, int] = (-5, -1), baseline: tuple[int, int] = (-65, -6),
                          covered_until: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """신호일별 U_t 전 종목의 `date, ticker, news_cnt_spike, news_sent_5` 원값(순위 변환 전).

    articles: date(기사 달력일), ticker, sentiment. 분석 대상 기사만 넣는다. 감성 점수가 없는 기사는 건수에만 든다.
    collected: 건수를 수집한 종목. 여기 없는 U_t 종목은 두 피처 모두 결측이다.
    coverage: 빠짐없이 수집한 달력일 구간 (시작, 끝).
    covered_from: {종목: 현재 상호를 쓰기 시작한 날}. 그 전 달력일이 창에 들면 결측이다.
    covered_until: {종목: 건수를 믿을 수 있는 마지막 날}. 그 뒤 달력일이 창에 들면 결측이다(예: 새 상호가 일반어·지명과 겹침).
    """
    days = pd.DatetimeIndex(sorted(pd.to_datetime(list(trading_days)))).normalize().unique()
    members = universe.loc[universe["in_universe"], ["date", "ticker"]].copy()
    members["date"] = pd.to_datetime(members["date"]).dt.normalize()
    members["ticker"] = members["ticker"].astype(str)
    if members.duplicated().any():
        raise ValueError("universe에 같은 (date, ticker)가 두 번 있다")
    if members.empty:
        return pd.DataFrame(columns=["date", "ticker", *NEWS_FEATURES])

    signals = pd.DatetimeIndex(sorted(members["date"].unique()))
    recent = _spans(days, signals, window)
    base = _spans(days, signals, baseline)
    scale = (window[1] - window[0] + 1) / (baseline[1] - baseline[0] + 1)

    tickers = np.array(sorted(members["ticker"].unique()), dtype=object)
    column = {t: i for i, t in enumerate(tickers)}
    first = min(base[0].min(), recent[0].min())
    calendar = pd.date_range(first, max(base[1].max(), recent[1].max()), freq="D")

    rows = articles[["date", "ticker", "sentiment"]].copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows["ticker"] = rows["ticker"].astype(str)
    rows = rows[rows["ticker"].isin(column) & (rows["date"] >= calendar[0]) & (rows["date"] <= calendar[-1])]
    day_at = ((rows["date"] - calendar[0]) // ONE_DAY).to_numpy(dtype="int64")
    col_at = rows["ticker"].map(column).to_numpy(dtype="int64")
    sentiment = rows["sentiment"].to_numpy(dtype="float64")
    has_score = ~np.isnan(sentiment)

    shape = (len(calendar) + 1, len(tickers))          # 첫 행은 0: 누적합으로 구간 합을 구한다
    count, score, scored = (np.zeros(shape) for _ in range(3))
    np.add.at(count, (day_at + 1, col_at), 1.0)
    np.add.at(score, (day_at[has_score] + 1, col_at[has_score]), sentiment[has_score])
    np.add.at(scored, (day_at[has_score] + 1, col_at[has_score]), 1.0)
    count, score, scored = (np.cumsum(m, axis=0) for m in (count, score, scored))

    # 종목별로 건수를 믿을 수 있는 달력일 [lo, hi]
    start, end = pd.Timestamp(coverage[0]).normalize(), pd.Timestamp(coverage[1]).normalize()
    since = {str(k): pd.Timestamp(v).normalize() for k, v in (covered_from or {}).items() if pd.notna(v)}
    known = set(map(str, collected))
    until = {str(k): pd.Timestamp(v).normalize() for k, v in (covered_until or {}).items() if pd.notna(v)}
    lo = np.array([max(start, since.get(t, start)) if t in known else pd.Timestamp.max for t in tickers],
                  dtype="datetime64[ns]")
    hi_all = np.array([min(end, until.get(t, end)) for t in tickers], dtype="datetime64[ns]")

    frames = []
    for i, (t, group) in enumerate(members.groupby("date", sort=True)):
        names = np.array(sorted(group["ticker"]), dtype=object)
        cols = np.array([column[n] for n in names])
        r0, r1, b0, b1 = recent[0][i], recent[1][i], base[0][i], base[1][i]
        recent_n = _range_sum(count, calendar, r0, r1, cols)
        base_n = _range_sum(count, calendar, b0, b1, cols)
        score_sum = _range_sum(score, calendar, r0, r1, cols)
        score_n = _range_sum(scored, calendar, r0, r1, cols)
        hi = hi_all[cols]
        ok_recent = (lo[cols] <= np.datetime64(r0)) & (np.datetime64(r1) <= hi)
        ok_all = (lo[cols] <= np.datetime64(min(r0, b0))) & (np.datetime64(max(r1, b1)) <= hi)
        spike = np.where(ok_all, (recent_n + 1.0) / (base_n * scale + 1.0), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            sent = np.where(ok_recent & (score_n > 0), score_sum / score_n, np.nan)
        frames.append(pd.DataFrame({"date": t, "ticker": names, "news_cnt_spike": spike, "news_sent_5": sent}))
    return pd.concat(frames, ignore_index=True)


def rank_news_features(raw: pd.DataFrame) -> pd.DataFrame:
    """신호일별 U_t 안 백분위 순위, 결측은 0.5와 `{피처}_na` (6.2절 변환표, A와 같은 함수)."""
    frames = []
    for t, group in raw.groupby("date", sort=True):
        ranked = rank_and_fill(group.set_index("ticker")[NEWS_FEATURES], rank_cols=NEWS_FEATURES)
        ranked.insert(0, "ticker", ranked.index.astype(str))
        ranked.insert(0, "date", t)
        frames.append(ranked.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]
    return out.astype({"news_cnt_spike_na": "int8", "news_sent_5_na": "int8"})


def build_news_features(articles: pd.DataFrame, universe: pd.DataFrame, trading_days: Iterable,
                        cfg: Mapping[str, Any], collected: Iterable[str], coverage: tuple,
                        covered_from: Mapping[str, Any] | None = None,
                        covered_until: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """사전등록 설정(`configs/prereg.yaml` 내용)으로 순위 변환까지 마친 뉴스 피처 표를 만든다."""
    features = cfg["features"]
    raw = compute_news_features(articles, universe, trading_days, collected, coverage, covered_from,
                                tuple(features["news_window_offsets"]), tuple(features["news_baseline_offsets"]),
                                covered_until)
    return rank_news_features(raw)


def analysis_articles(links: pd.DataFrame, sentiment: pd.DataFrame | None = None) -> pd.DataFrame:
    """빅카인즈 (기사, 종목) 표에서 분석 대상 기사(`bk_excluded`가 빈 행)만 골라 `date, ticker, sentiment`로 만든다.

    sentiment: news_id별 감성 점수(`news_id`, `sentiment`). 주면 news_id로 붙이고, 없으면 감성 점수는 결측이다.
    """
    excluded = links["bk_excluded"].fillna("").astype(str).str.strip() != ""
    rows = links.loc[~excluded, ["news_id", "date", "ticker"]].copy()
    if sentiment is not None:
        scores = sentiment[["news_id", "sentiment"]].drop_duplicates("news_id")
        rows = rows.merge(scores, on="news_id", how="left", validate="many_to_one")
    else:
        rows["sentiment"] = np.nan
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows["ticker"] = rows["ticker"].astype(str)
    rows["sentiment"] = rows["sentiment"].astype("float64")
    return rows[["date", "ticker", "sentiment"]].reset_index(drop=True)


def current_name_since(renames: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """상호 변경 공시 표(`ticker`, `change_date`)에서 종목별 마지막 변경일 = 현재 상호를 쓰기 시작한 날."""
    dates = pd.to_datetime(renames["change_date"], errors="coerce")
    latest = dates.groupby(renames["ticker"].astype(str)).max().dropna()
    return {ticker: day.normalize() for ticker, day in latest.items()}


def _spans(days: pd.DatetimeIndex, signals: pd.DatetimeIndex, offsets: tuple[int, int]):
    """거래일 창 [t+a, t+b]의 달력일 구간 (시작, 끝). 끝은 t+b+1 거래일의 전날이라 주말·휴장일 기사가 직전 거래일에 든다."""
    a, b = offsets
    if a > b or b >= 0:
        raise ValueError(f"창 {offsets}: 시작 ≤ 끝 < 0이어야 한다 (t일 기사는 쓰지 않는다)")
    position = days.searchsorted(signals)
    if (position >= len(days)).any() or not (days[np.minimum(position, len(days) - 1)] == signals).all():
        missing = signals[(position >= len(days)) | (days[np.minimum(position, len(days) - 1)] != signals)]
        raise ValueError(f"신호일 {missing[0].date()}이 거래일 목록에 없다")
    if (position + a < 0).any():
        short = signals[position + a < 0]
        raise ValueError(f"신호일 {short[0].date()} 앞에 거래일이 {-a}개보다 적다")
    return days[position + a], days[position + b + 1] - ONE_DAY


def _range_sum(cumulative: np.ndarray, calendar: pd.DatetimeIndex, first: pd.Timestamp, last: pd.Timestamp,
               cols: np.ndarray) -> np.ndarray:
    i = int((first - calendar[0]) // ONE_DAY)
    j = int((last - calendar[0]) // ONE_DAY)
    return cumulative[j + 1, cols] - cumulative[i, cols]
