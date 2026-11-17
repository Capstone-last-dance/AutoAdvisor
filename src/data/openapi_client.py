"""KRX Open API 클라이언트 (A-02). 표준 라이브러리만 쓴다. pykrx 경로(krx_client)와 무관하다.

- 키는 환경 변수 KRX_OPENAPI_KEY에서만 읽고, 오류 메시지·로그·저장 파일에 쓰지 않는다 (R-16).
- 서비스 목록(SERVICES)의 `verified=True`는 경로와 필드명을 Spec.docx 또는 승인 후 실제 응답으로 확인했다는 뜻이다. 필드의 의미(단위,
  조정 여부 등)는 서비스별 Spec으로 따로 확인한다. verified가 아닌 서비스를 새로 추가하면 `verified` 인자로 명시해야 호출한다 (R-08).
- 호출 한도는 인자로 받아 날짜별 누계를 파일에 남긴다. 한도 값은 공식 확인 전이라 기본값이 없다.
- 401/403은 재시도 없이 즉시 중단한다. 5xx와 네트워크 오류만 백오프 후 재시도한다.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

BASE_URL = "https://data-dbg.krx.co.kr/svc/apis/"   # Spec.docx 1.2절
KEY_ENV = "KRX_OPENAPI_KEY"
AUTH_HEADER = "AUTH_KEY"   # Spec.docx에 인증 방식 기재 없음. 서드파티 소스 기준. 승인 후 첫 호출에서 확인
DATE_PARAM = "basDd"       # Spec.docx 1.3.1절 (YYYYMMDD)
DATA_BLOCK = "OutBlock_1"  # Spec.docx 1.4.1절


@dataclass(frozen=True)
class Service:
    name: str
    path: str
    schedule: str    # weekdays: 평일 전체(거래일 캘린더 원천), trading: 거래일, signal: 주 마지막 거래일(신호일)
    verified: bool   # Spec 문서로 확인했는가 (R-08)
    source: str


_REAL = "2026-11-02 실제 응답으로 경로·필드명 확인(승인 후). 필드 의미는 서비스별 Spec으로 확인 필요"
SERVICES = {s.name: s for s in [
    Service("krx_dd_trd", "idx/krx_dd_trd", "weekdays", True, "OPENAPI_KRX_API/Spec.docx + 2026-11-02 실제 응답이 Spec 필드와 일치"),
    Service("kospi_dd_trd", "idx/kospi_dd_trd", "weekdays", True, _REAL + " (필드는 krx_dd_trd와 동일)"),
    Service("kosdaq_dd_trd", "idx/kosdaq_dd_trd", "weekdays", True, _REAL + " (필드는 krx_dd_trd와 동일)"),
    Service("stk_bydd_trd", "sto/stk_bydd_trd", "trading", True, _REAL),
    Service("ksq_bydd_trd", "sto/ksq_bydd_trd", "trading", True, _REAL),
    Service("stk_isu_base_info", "sto/stk_isu_base_info", "signal", True, _REAL),
    Service("ksq_isu_base_info", "sto/ksq_isu_base_info", "signal", True, _REAL),
]}


class OpenApiError(RuntimeError):
    pass


class OpenApiAuthError(OpenApiError):
    """401/403. 키 미승인·미활성이거나 서비스 미승인이다. 재시도하지 않는다."""


class QuotaExceeded(OpenApiError):
    pass


class OpenApiRateLimited(QuotaExceeded):
    """HTTP 429. 서버가 요청이 너무 많다고 알려 온 것이므로 재시도하지 않고 전체를 멈춘다."""


def load_key(env_path: Path | None = None) -> str:
    """환경 변수에 없으면 .env에서 KRX_OPENAPI_KEY 한 줄만 읽는다. 값은 반환만 하고 어디에도 쓰지 않는다."""
    key = os.environ.get(KEY_ENV)
    if not key and env_path and Path(env_path).exists():
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            if line.startswith(KEY_ENV + "="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise OpenApiError(f"{KEY_ENV} 환경변수가 없습니다 (.env 확인)")
    return key


class Quota:
    """날짜별 호출 누계. 재시도를 포함한 모든 HTTP 시도를 센다. 파일은 data/raw/_openapi_quota/{날짜}.json.
    호출마다 파일을 다시 읽고 원자적으로 쓴다(자정을 넘겨 도는 실행과 재시작에서도 날짜별로 센다). 같은 키를 쓰는 프로세스를
    동시에 여러 개 돌리는 것까지는 막지 못한다. 파일이 깨졌으면 0으로 되돌리지 않고 멈춘다(한도를 넘길 수 있다)."""

    def __init__(self, directory: Path, limit: int, today: date | None = None):
        self.limit, self.directory, self._today = limit, Path(directory), today
        self.calls = self._read()

    def _path(self) -> Path:
        return self.directory / f"{self._today or date.today():%Y-%m-%d}.json"

    def _read(self) -> int:
        p = self._path()
        if not p.exists():
            return 0
        try:
            return int(json.loads(p.read_text())["calls"])
        except (ValueError, KeyError, TypeError) as e:
            raise OpenApiError(f"호출 누계 파일이 깨졌다: {p}. 오늘 이미 호출한 횟수를 확인해 직접 고친 뒤 다시 실행한다") from e

    def take(self):
        self.calls = self._read()
        if self.calls >= self.limit:
            raise QuotaExceeded(f"오늘 호출 {self.calls}회로 한도 {self.limit}회에 도달했습니다")
        self.calls += 1
        p = self._path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"calls": self.calls}))
        tmp.replace(p)


def _urllib_transport(url: str, headers: dict, timeout: float):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 0, f"{type(e).__name__}"   # 상세 문자열에 헤더가 섞일 일은 없지만 본문은 버린다


class OpenApiClient:
    def __init__(self, key: str, quota: Quota, *, transport=_urllib_transport, retries=3, backoff=2.0,
                 timeout=30.0, sleep=time.sleep):
        self._key, self.quota, self._transport = key, quota, transport
        self.retries, self.backoff, self.timeout, self._sleep = retries, backoff, timeout, sleep

    def _scrub(self, s: str) -> str:
        return s.replace(self._key, "<KEY>")[:200]

    def get(self, service: Service, bas_dd: date) -> str:
        """응답 본문을 그대로 반환한다(원천 보관용). 형식이 기대와 다르면 OpenApiError."""
        url = f"{BASE_URL}{service.path}?" + urllib.parse.urlencode({DATE_PARAM: f"{bas_dd:%Y%m%d}"})
        last = None
        for i in range(self.retries):
            self.quota.take()
            status, body = self._transport(url, {AUTH_HEADER: self._key}, self.timeout)
            if status == 429:
                raise OpenApiRateLimited(f"429 {service.name}: 요청이 너무 많다는 응답을 받았습니다. 간격(--interval)과 하루 한도를 낮추세요")
            if status in (401, 403):
                raise OpenApiAuthError(f"{status} {service.name}: {self._scrub(body)}")
            if status == 200:
                try:
                    block = json.loads(body).get(DATA_BLOCK)
                except ValueError:   # 본문이 비었거나 JSON이 아님(2026-11-02 일시 발생). 서버 일시 오류로 보고 재시도한다
                    last = f"200 본문 해석 실패 {self._scrub(body)!r}"
                    self._sleep(self.backoff * (i + 1))
                    continue
                except AttributeError:
                    block = None
                if not isinstance(block, list):
                    raise OpenApiError(f"{service.name} {bas_dd}: 응답에 {DATA_BLOCK} 목록이 없음: {self._scrub(body)}")
                return body
            if status == 0 or status >= 500:
                last = f"{status} {self._scrub(body)}"
                self._sleep(self.backoff * (i + 1))
                continue
            raise OpenApiError(f"{status} {service.name} {bas_dd}: {self._scrub(body)}")
        raise OpenApiError(f"{service.name} {bas_dd}: {self.retries}회 재시도 실패 ({last})")
