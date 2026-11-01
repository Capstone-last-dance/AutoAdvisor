"""뉴스의 종목코드·표준 시각·원문 URL 규약 검증."""
from datetime import datetime

import pytest

from src.data.naver_client import KST
from src.data.news import NEWS_COLUMNS, NewsRecordError, normalize_naver_item, parse_naver_time

OBSERVED = datetime(2026,11,1,16,0,tzinfo=KST)


def article(**changes):
    value = {"title":"<b>가나다전자</b> &amp; 반도체 &lt;기술&gt;", "originallink":"https://publisher.example/articles/1", "link":"https://news.example/1", "pubDate":"Sun, 01 Nov 2026 06:30:00 +0000"}
    value.update(changes)
    return value


def test_canonical_schema_html_and_kst_cutoff():
    observed = datetime(2026,11,1,15,30,tzinfo=KST)
    row = normalize_naver_item(article(),ticker="005930",observed_at=observed)
    assert list(row) == NEWS_COLUMNS
    assert row["ticker"] == "005930"
    assert row["published_at"] == observed
    assert row["title"] == "가나다전자 & 반도체 <기술>"
    assert row["url"] == "https://publisher.example/articles/1"
    assert row["sentiment"] is None and row["source"] == "naver"


def test_url_identity_survives_title_edits_and_ticker_mapping():
    first = normalize_naver_item(article(),ticker="005930",observed_at=OBSERVED)
    second = normalize_naver_item(article(title="제목 변경"),ticker="000660",observed_at=OBSERVED)
    third = normalize_naver_item(article(originallink="https://publisher.example/articles/2"),ticker="005930",observed_at=OBSERVED)
    assert first["news_id"] == second["news_id"]
    assert first["news_id"] != third["news_id"]


@pytest.mark.parametrize("value",[None,"2026-11-01","Sun, 01 Nov 2026 15:30:00","not-a-date",123])
def test_date_only_invalid_or_naive_time_is_refused(value):
    with pytest.raises(NewsRecordError):
        parse_naver_time(value)


@pytest.mark.parametrize("ticker",[5930,"5930",None,"0059300","abc123"])
def test_ticker_is_never_invented_or_zero_padded(ticker):
    with pytest.raises(NewsRecordError):
        normalize_naver_item(article(),ticker=ticker,observed_at=OBSERVED)


def test_future_item_is_refused_at_signal_cutoff():
    with pytest.raises(NewsRecordError):
        normalize_naver_item(article(pubDate="Sun, 01 Nov 2026 15:31:00 +0900"),ticker="005930",observed_at=datetime(2026,11,1,15,30,tzinfo=KST))
    with pytest.raises(NewsRecordError):
        normalize_naver_item(article(),ticker="005930",observed_at=datetime(2026,11,1,15,30))


@pytest.mark.parametrize("url",["javascript:alert(1)","file:///data/private","https://user:pass@publisher.example/1",""])
def test_invalid_urls_are_refused(url):
    with pytest.raises(NewsRecordError):
        normalize_naver_item(article(originallink=url,link=url),ticker="005930",observed_at=OBSERVED)


def test_missing_original_url_uses_documented_link():
    row = normalize_naver_item(article(originallink=""),ticker="005930",observed_at=OBSERVED)
    assert row["url"] == "https://news.example/1"


@pytest.mark.parametrize("title",[None," ","<b></b>"])
def test_empty_title_is_refused(title):
    with pytest.raises(NewsRecordError):
        normalize_naver_item(article(title=title),ticker="005930",observed_at=OBSERVED)


def test_observation_time_cannot_be_omitted():
    with pytest.raises(TypeError):
        normalize_naver_item(article(),ticker="005930")
