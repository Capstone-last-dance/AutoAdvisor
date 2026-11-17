"""네이버 공식 뉴스 검색 API. 인증 값은 응답·로그·공개 예외에 넣지 않는다."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

KST = timezone(timedelta(hours=9))
NEWS_ENDPOINT = "https://openapi.naver.com/v1/search/news.json"


class NaverError(Exception):
    """인증 값이 없는 공개 오류."""


class NaverAuthError(NaverError):
    """클라이언트 인증 또는 API 권한 실패."""


class NaverLimitError(NaverError):
    """서버 제한 또는 이 클라이언트의 호출 상한 도달."""


class NaverResponseError(NaverError):
    """예상하지 않은 HTTP 응답 또는 JSON 구조."""


@dataclass(frozen=True)
class NewsSearchRequest:
    query: str
    display: int = 100
    start: int = 1
    sort: str = "date"

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("검색어가 필요합니다")
        if type(self.display) is not int or not 1 <= self.display <= 100:
            raise ValueError("display는 1~100 정수여야 합니다")
        if type(self.start) is not int or not 1 <= self.start <= 1000:
            raise ValueError("start는 1~1000 정수여야 합니다")
        if self.sort not in {"date", "sim"}:
            raise ValueError("sort는 date 또는 sim이어야 합니다")

    def params(self):
        return {"query": self.query, "display": self.display, "start": self.start, "sort": self.sort}


@dataclass(frozen=True)
class NewsSearchResponse:
    request: NewsSearchRequest
    body: bytes
    http_status: int
    fetched_at: datetime
    elapsed_ms: int

    def require_success(self):
        if self.http_status in {401, 403}:
            raise NaverAuthError(f"네이버 인증 또는 권한 실패 (HTTP {self.http_status})")
        if self.http_status == 429:
            raise NaverLimitError("네이버 호출 제한 (HTTP 429)")
        if self.http_status != 200:
            raise NaverResponseError(f"네이버 응답 실패 (HTTP {self.http_status})")

    def payload(self):
        self.require_success()
        try:
            value = json.loads(self.body)
            if not isinstance(value, dict):
                raise ValueError
            if type(value["total"]) is not int or value["total"] < 0:
                raise ValueError
            if type(value["start"]) is not int or value["start"] != self.request.start:
                raise ValueError
            if type(value["display"]) is not int or not 0 <= value["display"] <= self.request.display:
                raise ValueError
            if not isinstance(value["items"], list) or len(value["items"]) > self.request.display:
                raise ValueError
            if not all(isinstance(item, dict) for item in value["items"]):
                raise ValueError
            return value
        except (ValueError, KeyError, TypeError):
            raise NaverResponseError("네이버 JSON 응답 형식 오류") from None


class NaverNewsClient:
    """기본 상한은 1회. 제한·인증·연결 오류를 자동 재시도하지 않는다."""

    def __init__(self, client_id=None, client_secret=None, *, session=None, max_calls=1,
                 now=None, clock=time.monotonic, timeout=(5, 20)):
        self._client_id = (client_id if client_id is not None else os.environ.get("NAVER_CLIENT_ID", "")).strip()
        self._client_secret = (client_secret if client_secret is not None else os.environ.get("NAVER_CLIENT_SECRET", "")).strip()
        if not self._client_id or not self._client_secret:
            raise NaverAuthError("NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET 설정이 필요합니다")
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("호출 상한은 양의 정수여야 합니다")
        self.session = session if session is not None else requests.Session()
        self.max_calls = max_calls
        self.calls_attempted = 0
        self.now = now or (lambda: datetime.now(KST))
        self.clock = clock
        self.timeout = timeout

    def search(self, request: NewsSearchRequest) -> NewsSearchResponse:
        if not isinstance(request, NewsSearchRequest):
            raise TypeError("NewsSearchRequest가 필요합니다")
        if self.calls_attempted >= self.max_calls:
            raise NaverLimitError("이 실행의 네이버 호출 상한에 도달했습니다")
        self.calls_attempted += 1
        started = self.clock()
        try:
            response = self.session.get(
                NEWS_ENDPOINT, params=request.params(),
                headers={"X-Naver-Client-Id": self._client_id, "X-Naver-Client-Secret": self._client_secret},
                timeout=self.timeout, allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise NaverError(f"네이버 연결 실패 ({type(exc).__name__})") from None
        body = response.content
        if any(value.encode("utf-8") in body for value in (self._client_id, self._client_secret)):
            raise NaverResponseError("응답에 인증 값이 포함돼 보관을 거부했습니다")
        fetched_at = self.now()
        if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
            raise ValueError("수집 시각은 시간대가 있어야 합니다")
        return NewsSearchResponse(request, body, response.status_code, fetched_at.astimezone(KST),
                                  round((self.clock() - started) * 1000))
