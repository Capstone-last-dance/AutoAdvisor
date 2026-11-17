"""합성 응답만 사용해 인증 헤더·호출 제한·불변 원본 보관 검증."""
import hashlib
import json
from datetime import datetime

import pytest
import requests

from src.data.collect_news import capture_naver_sample, main
from src.data.naver_client import (KST, NEWS_ENDPOINT, NaverAuthError, NaverError,
                                   NaverLimitError, NaverNewsClient, NaverResponseError,
                                   NewsSearchRequest)

TEST_ID = "SYNTHETIC_CLIENT_ID"
TEST_SECRET = "SYNTHETIC_SECRET_VALUE"
FETCHED = datetime(2026,11,1,16,0,tzinfo=KST)


class Response:
    def __init__(self,payload=None,*,status=200,body=None):
        self.status_code = status
        self.content = body if body is not None else json.dumps(payload).encode("utf-8")


class Session:
    def __init__(self,response):
        self.response = response
        self.calls = []

    def get(self,url,**kwargs):
        self.calls.append((url,kwargs))
        if isinstance(self.response,BaseException):
            raise self.response
        return self.response


def payload():
    return {"lastBuildDate":"Sun, 01 Nov 2026 16:00:00 +0900", "total":1,"start":1,"display":1,
            "items":[{"title":"<b>합성 기사</b>","pubDate":"Sun, 01 Nov 2026 15:30:00 +0900","link":"https://news.example/1","originallink":"https://publisher.example/1"}]}


def client(response,**kwargs):
    session = Session(response)
    return NaverNewsClient(TEST_ID,TEST_SECRET,session=session,now=lambda:FETCHED,**kwargs),session


def test_headers_only_fixed_endpoint_no_redirects_and_one_call_budget():
    api,session = client(Response(payload()))
    request = NewsSearchRequest("합성기업",display=1)
    assert api.search(request).payload() == payload()
    url,kwargs = session.calls[0]
    assert url == NEWS_ENDPOINT
    assert kwargs["headers"] == {"X-Naver-Client-Id":TEST_ID,"X-Naver-Client-Secret":TEST_SECRET}
    assert TEST_ID not in json.dumps(kwargs["params"])
    assert kwargs["allow_redirects"] is False
    with pytest.raises(NaverLimitError):
        api.search(request)
    assert len(session.calls) == 1


@pytest.mark.parametrize("changes",[{"display":0},{"display":101},{"display":True},{"start":0},{"start":1001},{"start":True},{"sort":"unknown"},{"query":" "}])
def test_documented_request_bounds(changes):
    values = {"query":"합성기업"}
    values.update(changes)
    with pytest.raises(ValueError):
        NewsSearchRequest(**values)


def test_start_max_is_not_a_combined_start_plus_display_constraint():
    assert NewsSearchRequest("합성기업",start=1000,display=100).start == 1000


@pytest.mark.parametrize("status,error",[(401,NaverAuthError),(403,NaverAuthError),(429,NaverLimitError),(302,NaverResponseError),(500,NaverResponseError)])
def test_http_failure_is_not_retried(status,error):
    api,session = client(Response({"errorCode":"024","errorMessage":"합성 오류"},status=status))
    page = api.search(NewsSearchRequest("합성기업"))
    with pytest.raises(error):
        page.payload()
    assert len(session.calls) == 1


def test_transport_exception_and_echoed_response_never_expose_credentials():
    api,_ = client(requests.ConnectionError(TEST_SECRET))
    with pytest.raises(NaverError) as caught:
        api.search(NewsSearchRequest("합성기업"))
    assert TEST_SECRET not in str(caught.value)
    api,_ = client(Response(body=TEST_SECRET.encode()))
    with pytest.raises(NaverResponseError) as caught:
        api.search(NewsSearchRequest("합성기업"))
    assert TEST_SECRET not in str(caught.value)


@pytest.mark.parametrize("changes",[{"total":-1},{"total":True},{"start":2},{"display":101},{"items":"bad"},{"items":["bad"]}])
def test_invalid_response_structure_is_refused(changes):
    values=payload()
    values.update(changes)
    api,_ = client(Response(values))
    with pytest.raises(NaverResponseError):
        api.search(NewsSearchRequest("합성기업")).payload()


def test_empty_successful_snapshot():
    api,_ = client(Response({"total":0,"start":1,"display":0,"items":[]}))
    assert api.search(NewsSearchRequest("합성기업")).payload()["items"] == []


def test_official_body_and_safe_metadata_are_preserved_without_ticker_assignment(tmp_path):
    api,_ = client(Response(payload()))
    report=capture_naver_sample(NewsSearchRequest("합성기업"),api,run_id="synthetic_sample",data_root=tmp_path/"raw",reports_root=tmp_path/"reports")
    page=next((tmp_path/"raw").rglob("response_0001.json"))
    meta=json.loads(page.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert page.read_bytes() == Response(payload()).content
    assert meta["sha256"] == hashlib.sha256(page.read_bytes()).hexdigest()
    assert report["unmapped_items"] == 1 and report["ticker_mapping_performed"] is False
    assert report["news_parquet_written"] is False and report["historical_complete"] is False
    for path in tmp_path.rglob("*.json"):
        assert TEST_ID not in path.read_text(encoding="utf-8") and TEST_SECRET not in path.read_text(encoding="utf-8")
    second_api,second_session = client(Response(payload()))
    with pytest.raises(FileExistsError):
        capture_naver_sample(NewsSearchRequest("합성기업"),second_api,run_id="synthetic_sample",data_root=tmp_path/"raw",reports_root=tmp_path/"reports")
    assert not second_session.calls


def test_auth_failure_body_is_archived_and_run_is_failed(tmp_path):
    body=Response({"errorCode":"024"},status=401)
    api,session=client(body)
    result=capture_naver_sample(NewsSearchRequest("합성기업"),api,run_id="auth_failed",data_root=tmp_path/"raw",reports_root=tmp_path/"reports")
    assert result["status"] == "failed" and result["http_status"] == 401
    assert result["error_type"] == "NaverAuthError" and len(session.calls)==1
    assert next((tmp_path/"raw").rglob("response_0001.json")).read_bytes() == body.content


def test_future_dates_are_reported_without_building_processed_data(tmp_path):
    data=payload()
    data["items"][0]["pubDate"]="Sun, 01 Nov 2026 16:01:00 +0900"
    api,_=client(Response(data))
    result=capture_naver_sample(NewsSearchRequest("합성기업"),api,run_id="future_date",data_root=tmp_path/"raw",reports_root=tmp_path/"reports")
    assert result["invalid_or_future_dates"]==1
    assert result["published_max"] is None and result["news_parquet_written"] is False


def test_cli_preflight_missing_credentials_makes_no_network_calls(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NAVER_CLIENT_ID",raising=False)
    monkeypatch.delenv("NAVER_CLIENT_SECRET",raising=False)
    assert main(["--query","합성기업"]) == 2
