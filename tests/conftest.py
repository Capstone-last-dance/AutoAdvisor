"""네트워크와 실제 대기를 차단하는 PM-04 합성 테스트 도구."""

import json
from collections import defaultdict
from datetime import datetime, timedelta

import pytest
import requests

from src.data.dart_client import KST, CallBudget, DartClient


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("단위 테스트에서 실제 네트워크를 호출했습니다")
    monkeypatch.setattr(requests.Session, "get", fail)


class FakeClock:
    def __init__(self, now=None):
        self.value = 0.0
        self.origin = now or datetime(2026, 10, 28, 21, tzinfo=KST)
        self.sleeps = []

    def clock(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds

    def now(self):
        return self.origin + timedelta(seconds=self.value)


class FakeResponse:
    def __init__(self, payload=None, *, content=None, status=200):
        self.content = content if content is not None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.status_code = status


class FakeSession:
    def __init__(self, scenario, clock=None, elapsed=0):
        self.scenario = scenario
        self.clock = clock
        self.elapsed = elapsed
        self.requests = []
        self.counts = defaultdict(int)
        self.starts = []

    def get(self, url, params, timeout):
        key = (params["bgn_de"], params["end_de"], params.get("pblntf_ty"), int(params["page_no"]))
        self.requests.append(key)
        self.starts.append(self.clock.clock() if self.clock else 0)
        self.counts[key] += 1
        if callable(self.scenario):
            value = self.scenario(key, self.counts[key])
        elif isinstance(self.scenario, list):
            value = self.scenario.pop(0)
        else:
            value = self.scenario[key]
            if isinstance(value, list):
                value = value.pop(0)
        if isinstance(value, BaseException):
            raise value
        if self.clock:
            self.clock.value += self.elapsed
        return value if isinstance(value, FakeResponse) else FakeResponse(value)


def item(number=1, *, day="20250101", **changes):
    return {"rcept_no": f"{day}{number:06d}", "rcept_dt": day,
            "corp_code": "00000001", "corp_name": "가나다전자", "stock_code": "005930",
            "corp_cls": "Y", "rm": "", "flr_nm": "가나다전자",
            "report_nm": "[기재정정]유상 ㆍ 증자·결정", **changes}


def page(items=(), *, total=None, pages=1, number=1, status="000"):
    if status != "000":
        return {"status": status, "message": "합성 응답"}
    return {"status": status, "message": "정상", "page_no": number, "page_count": 100,
            "total_count": len(items) if total is None else total, "total_page": pages, "list": list(items)}


@pytest.fixture
def client_factory():
    def create(scenario, *, limit=18000, clock=None, elapsed=0):
        clock = clock or FakeClock()
        session = FakeSession(scenario, clock, elapsed)
        client = DartClient("TESTKEY-not-real", session=session, clock=clock.clock,
                            sleep=clock.sleep, now=clock.now, budget=CallBudget(limit))
        return client, session, clock
    return create
