"""KRX(pykrx) 호출 공통 모듈.

- 로그인 출력 차단: pykrx가 로그인할 때마다 계정 ID를 출력하므로(R-16) import 전에 필터를 건다.
- 재시도: 연결 리셋 등 일시 오류는 백오프 후 재시도한다. 세션 만료는 pykrx가 자동 재로그인한다.
- 빈 응답 감지: 로그인 실패와 데이터 없음이 모두 빈 응답이라 구분되지 않으므로, 비어 있으면 예외로 올린다.
"""
import io
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
_BLOCKED = ("로그인 ID",)


class _Filter(io.TextIOBase):
    def __init__(self, out):
        self._out = out

    def write(self, s):
        if any(b in s for b in _BLOCKED):
            return len(s)
        return self._out.write(s)

    def flush(self):
        self._out.flush()


def _load():
    load_dotenv(Path(os.environ.get("KRX_ENV_PATH", ROOT / ".env")))
    if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        raise RuntimeError("KRX_ID/KRX_PW 환경변수가 없습니다 (.env 확인)")
    if not isinstance(sys.stdout, _Filter):
        sys.stdout = _Filter(sys.stdout)


_load()
from pykrx import stock  # noqa: E402  (필터를 건 뒤에 import해야 로그인 출력이 차단된다)


import threading  # noqa: E402

from pykrx.website.comm import auth as _auth, webio as _webio  # noqa: E402

_lock = threading.Lock()
_aborted = threading.Event()
_orig_get_auth_session = _auth.get_auth_session


class KrxAuthAbort(RuntimeError):
    """세션 갱신 실패. 반복 로그인 시도는 계정 잠금(CD007)으로 이어질 수 있어 전체를 중단한다."""


def _locked_get_auth_session():
    # pykrx의 갱신 코드에는 락이 없어, 만료 시점에 스레드마다 동시에 재로그인(CD011)할 수 있다.
    with _lock:
        if _aborted.is_set():
            raise KrxAuthAbort("이전 세션 갱신 실패로 중단됨")
        sess = _orig_get_auth_session()
        if sess is None:
            _aborted.set()
            raise KrxAuthAbort("KRX 세션 갱신 실패. KRX 사이트에서 로그인 상태와 비밀번호 만료(CD010)를 확인하세요")
        return sess


_webio.get_auth_session = _locked_get_auth_session  # webio.get_session()이 이 이름을 쓴다


class KrxEmpty(Exception):
    pass


_PROGRAMMING_ERRORS = (TypeError, KeyError, AttributeError, NameError)   # 다시 불러도 같은 결과이고 KRX에는 요청만 쌓인다


def call(fn, *args, retries=6, backoff=2.0, expect_rows=True, **kwargs):
    if retries < 1:
        raise ValueError("retries는 1 이상이어야 한다")
    for i in range(retries):
        try:
            r = fn(*args, **kwargs)
            if expect_rows and r is not None and hasattr(r, "__len__") and len(r) == 0:
                raise KrxEmpty(f"빈 응답: {getattr(fn, '__name__', fn)} {args}")
            return r
        except (KrxAuthAbort, *_PROGRAMMING_ERRORS):
            raise
        except Exception:  # 일시 오류와 빈 응답은 재시도. 마지막 시도 뒤에는 기다리지 않는다
            if i == retries - 1:
                raise
            time.sleep(backoff * (i + 1))
