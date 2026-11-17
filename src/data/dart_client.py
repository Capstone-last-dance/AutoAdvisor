"""OpenDART 공시검색 클라이언트. 인증 정보는 외부로 내보내지 않는다."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

import requests

KST = timezone(timedelta(hours=9), "KST")
BASE_URL = "https://opendart.fss.or.kr/api"
LIST_ENDPOINT = "list.json"
API_KEY_ENV = "OPENDART_API_KEY"
MIN_INTERVAL_SEC = 1.1
TIMEOUT_SEC = 30
RETRY_BACKOFF_SEC = (5, 15, 45)
MAINTENANCE_BACKOFF_SEC = (60, 300)
PAGE_COUNT = 100


class DartError(Exception):
    """인증 정보를 포함하지 않는 외부 공개 예외."""


class DartAuthError(DartError):
    """키 또는 접근 권한 오류."""


class DartRequestError(DartError):
    """요청 인자 또는 페이지 번호 오류."""


class DartRateLimitError(DartError):
    """서버 호출 제한."""


class DartUnavailableError(DartError):
    """재시도 후에도 이용 불가."""


class DailyBudgetExceeded(DartError):
    """로컬 호출 예산 소진."""


@dataclass(frozen=True)
class ListRequest:
    bgn_de: str
    end_de: str
    pblntf_ty: str | None
    page_no: int
    page_count: int = PAGE_COUNT
    corp_code: str | None = None
    last_reprt_at: str = "N"

    def __post_init__(self):
        try:
            bgn = datetime.strptime(self.bgn_de, "%Y%m%d").date()
            end = datetime.strptime(self.end_de, "%Y%m%d").date()
        except (ValueError, TypeError):
            raise ValueError("검색 날짜는 YYYYMMDD 형식이어야 합니다") from None
        if not re.fullmatch(r"\d{8}", self.bgn_de) or not re.fullmatch(r"\d{8}", self.end_de):
            raise ValueError("검색 날짜는 8자리여야 합니다")
        if bgn > end or self.page_no < 1 or self.page_count != PAGE_COUNT:
            raise ValueError("날짜 순서, 페이지 번호 또는 페이지 크기가 잘못되었습니다")
        if self.last_reprt_at != "N":
            raise ValueError("정정 전 원본 보존을 위해 last_reprt_at=N만 허용합니다")
        if self.pblntf_ty is not None and self.pblntf_ty not in tuple("ABCDEFGHIJ"):
            raise ValueError("알 수 없는 공시유형입니다")
        if self.corp_code is not None and not re.fullmatch(r"\d{8}", self.corp_code):
            raise ValueError("기업코드는 8자리 문자열이어야 합니다")
        if self.corp_code is None:
            month = bgn.month - 1 + 3
            year, month = bgn.year + month // 12, month % 12 + 1
            import calendar
            bound = bgn.replace(year=year, month=month, day=min(bgn.day, calendar.monthrange(year, month)[1]))
            if end >= bound:
                raise ValueError("기업코드 없는 조회는 3개월 미만의 날짜 범위만 허용합니다")

    def params(self) -> dict[str, str]:
        """키와 조회 시점의 기업 분류는 절대 포함하지 않는다."""
        result = {"bgn_de": self.bgn_de, "end_de": self.end_de,
                  "page_no": str(self.page_no), "page_count": str(self.page_count),
                  "last_reprt_at": "N"}
        if self.pblntf_ty is not None:
            result["pblntf_ty"] = self.pblntf_ty
        if self.corp_code is not None:
            result["corp_code"] = self.corp_code
        return result


@dataclass(frozen=True)
class ListPage:
    request: ListRequest
    status: str
    message: str
    page_no: int
    total_count: int
    total_page: int
    items: tuple[dict[str, str], ...]
    body: bytes
    http_status: int
    fetched_at: datetime
    elapsed_ms: int


class CallBudget:
    """호출을 보내기 전에 KST 날짜별 예산을 차감한다."""

    def __init__(self, limit_per_day: int, used_by_date: dict[str, int] | None = None):
        if limit_per_day < 1:
            raise ValueError("하루 예산은 양수여야 합니다")
        self.limit_per_day = limit_per_day
        self._used = dict(used_by_date or {})

    def consume(self, now: datetime) -> None:
        key = now.astimezone(KST).date().isoformat()
        if self._used.get(key, 0) >= self.limit_per_day:
            raise DailyBudgetExceeded("KST 기준 하루 호출 예산 소진")
        self._used[key] = self._used.get(key, 0) + 1

    @property
    def used_by_date(self) -> dict[str, int]:
        return dict(self._used)


class DartClient:
    def __init__(self, api_key: str | None = None, *, session=None,
                 min_interval: float = MIN_INTERVAL_SEC, timeout: float = TIMEOUT_SEC,
                 budget: CallBudget | None = None, clock: Callable = time.monotonic,
                 sleep: Callable = time.sleep, now: Callable = lambda: datetime.now(KST)):
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not key or not key.strip():
            raise DartAuthError("OPENDART_API_KEY 환경 변수가 필요합니다")
        if min_interval < MIN_INTERVAL_SEC or timeout <= 0:
            raise ValueError("요청 간격은 1.1초 이상, 시간 제한은 양수여야 합니다")
        self._key = key.strip()
        self.session = session if session is not None else requests.Session()
        self.min_interval, self.timeout = min_interval, timeout
        self.budget = budget if budget is not None else CallBudget(18000)
        self.clock, self.sleep, self.now = clock, sleep, now
        self._last_start = None
        self._calls = 0
        self._rate_limited_date = None
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def __repr__(self):
        return f"DartClient(calls={self.calls})"

    @property
    def calls(self) -> int:
        return self._calls

    def _redact(self, text: str) -> str:
        return re.sub(r"crtfc_key=[^&\s]+", "crtfc_key=***", str(text).replace(self._key, "***"))

    def fetch_list_page(self, req: ListRequest) -> ListPage:
        transient_tries = maintenance_tries = 0
        while True:
            day = self.now().astimezone(KST).date()
            if self._rate_limited_date == day:
                raise DartRateLimitError("이 KST 날짜에는 서버 호출 제한에 도달했습니다")
            self.budget.consume(self.now())
            if self._last_start is not None:
                wait = self.min_interval - (self.clock() - self._last_start)
                if wait > 0:
                    self.sleep(wait)
            started = self.clock()
            self._last_start = started
            self._calls += 1
            failure = "응답 형식 오류"
            maintenance = False
            try:
                response = self.session.get(f"{BASE_URL}/{LIST_ENDPOINT}",
                                            params=req.params() | {"crtfc_key": self._key},
                                            timeout=self.timeout)
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise DartRequestError(f"HTTP {response.status_code}: {req.params()}")
                if response.status_code != 200:
                    raise ValueError(f"HTTP {response.status_code}")
                body = response.content
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict) or "status" not in payload:
                    raise ValueError("status 없음")
                status = str(payload["status"])
                message = self._redact(str(payload.get("message", "")))
                if status in {"010", "011", "012", "901"}:
                    raise DartAuthError(f"OpenDART 인증 오류 {status}")
                if status == "020":
                    self._rate_limited_date = day
                    raise DartRateLimitError("OpenDART 호출 제한 (020)")
                if status in {"100", "101", "021"}:
                    raise DartRequestError(self._redact(f"OpenDART {status}: {message}; {req.params()}"))
                if status == "013":
                    return ListPage(req, status, message, req.page_no, 0, 0, (), body,
                                    200, self.now(), int((self.clock() - started) * 1000))
                if status == "000":
                    number = int(payload["page_no"])
                    if number != req.page_no:
                        raise DartRequestError(f"응답 페이지 번호 불일치: 요청 {req.page_no}, 응답 {number}")
                    total, pages, items = int(payload["total_count"]), int(payload["total_page"]), payload["list"]
                    if (total < 0 or pages < 0 or (total > 0 and pages < 1)
                            or not isinstance(items, list) or len(items) > req.page_count
                            or not all(isinstance(item, dict) for item in items)):
                        raise ValueError("응답 구조 오류")
                    return ListPage(req, status, message, number, total, pages, tuple(items), body,
                                    200, self.now(), int((self.clock() - started) * 1000))
                maintenance = status == "800"
                failure = f"OpenDART status {status}"
            except (DartAuthError, DartRequestError, DartRateLimitError):
                raise
            except (requests.RequestException, ValueError, TypeError, KeyError, UnicodeError) as exc:
                # requests 예외의 문자열과 원래 예외 연결에는 키가 든 URL이 있을 수 있다.
                failure = type(exc).__name__
            if maintenance:
                if maintenance_tries >= len(MAINTENANCE_BACKOFF_SEC):
                    raise DartUnavailableError("OpenDART 점검 재시도 소진") from None
                self.sleep(MAINTENANCE_BACKOFF_SEC[maintenance_tries])
                maintenance_tries += 1
            else:
                if transient_tries >= len(RETRY_BACKOFF_SEC):
                    raise DartUnavailableError(self._redact(f"일시 오류 재시도 소진: {failure}; {req.params()}")) from None
                self.sleep(RETRY_BACKOFF_SEC[transient_tries])
                transient_tries += 1
