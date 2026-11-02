"""PM-05 뉴스 표준화. 종목 매핑·시각을 추정해 채우지 않는다."""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, time
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit

from .naver_client import KST

# DEC-09: published_at은 실제 발행 시각(모르면 결측), available_at은 신호에 쓸 수 있게 되는 시각(필수).
NEWS_COLUMNS = ["news_id", "published_at", "available_at", "ticker", "title", "url", "source", "sentiment"]
DATE_ONLY_AVAILABLE_TIME = time(23, 59, 59)


def date_only_available_at(day: date) -> datetime:
    """발행 날짜만 아는 기사는 그날 23:59:59 KST에 나온 것으로 본다 (DEC-09 (1))."""
    if not isinstance(day, date) or isinstance(day, datetime):
        raise NewsRecordError("발행 날짜(date)가 필요합니다")
    return datetime.combine(day, DATE_ONLY_AVAILABLE_TIME, tzinfo=KST)


class NewsRecordError(ValueError):
    """종목·시각·필수 필드가 검증되지 않은 기사."""


class _TitleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def parse_naver_time(value: str) -> datetime:
    try:
        if not isinstance(value, str) or not value.strip():
            raise ValueError
        timestamp = parsedate_to_datetime(value)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError
        return timestamp.astimezone(KST)
    except (ValueError, TypeError, OverflowError):
        raise NewsRecordError("네이버 pubDate의 유효한 시간대 포함 시각이 필요합니다") from None


def normalize_naver_item(item: dict, *, ticker: str, observed_at: datetime) -> dict:
    """ticker는 별도로 검증한 매핑 결과만 전달한다. 검색어로 자동 배정하지 않는다."""
    if not isinstance(ticker, str) or re.fullmatch(r"[0-9A-Z]{6}", ticker) is None:
        raise NewsRecordError("검증된 6자리 문자열 종목코드가 필요합니다")
    if not isinstance(item, dict):
        raise NewsRecordError("기사 객체가 필요합니다")
    published_at = parse_naver_time(item.get("pubDate"))
    if not isinstance(observed_at, datetime) or observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise NewsRecordError("관측 시각은 시간대가 있어야 합니다")
    if published_at > observed_at:
        raise NewsRecordError("발행 시각이 관측 시각보다 늦어 격리합니다")
    title = item.get("title")
    if not isinstance(title, str):
        raise NewsRecordError("기사 제목이 필요합니다")
    parser = _TitleText()
    parser.feed(title)
    parser.close()
    title = " ".join("".join(parser.parts).split())
    if not title:
        raise NewsRecordError("기사 제목이 비어 있습니다")
    url = None
    for value in (item.get("originallink"), item.get("link")):
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        try:
            parsed = urlsplit(candidate)
            if parsed.scheme in {"https", "http"} and parsed.hostname and not parsed.username and not parsed.password:
                url = candidate
                break
        except ValueError:
            continue
    if url is None:
        raise NewsRecordError("HTTP(S) 기사 URL이 필요합니다")
    return {"news_id": "naver:" + hashlib.sha256(url.encode("utf-8")).hexdigest(),
            "published_at": published_at, "available_at": published_at, "ticker": ticker, "title": title, "url": url,
            "source": "naver", "sentiment": None}
