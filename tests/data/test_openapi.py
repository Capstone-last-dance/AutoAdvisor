import json
from datetime import date

import pytest

from src.data import collect_openapi as co
from src.data.openapi_client import (SERVICES, OpenApiAuthError, OpenApiClient, OpenApiError, Quota, QuotaExceeded,
                                     load_key)

KEY = "TESTKEY-1234"
OK = json.dumps({"OutBlock_1": [{"BAS_DD": "20260105", "IDX_NM": "코스피 200", "CLSPRC_IDX": "300.1"}]})
EMPTY = json.dumps({"OutBlock_1": []})


class Fake:
    """transport 대역: 응답 목록을 차례로 돌려주고 호출 기록을 남긴다."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, url, headers, timeout):
        self.calls.append((url, dict(headers)))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return r


def _client(tmp_path, fake, limit=100, **kw):
    return OpenApiClient(KEY, Quota(tmp_path / "q", limit, today=date(2026, 10, 30)), transport=fake,
                         sleep=lambda s: None, **kw)


def test_get_sends_key_in_header_only_and_returns_body_verbatim(tmp_path):
    f = Fake((200, OK))
    body = _client(tmp_path, f).get(SERVICES["krx_dd_trd"], date(2026, 1, 5))
    url, headers = f.calls[0]
    assert body == OK
    assert "basDd=20260105" in url and KEY not in url and headers == {"AUTH_KEY": KEY}


def test_auth_error_is_not_retried_and_message_hides_key(tmp_path):
    f = Fake((401, json.dumps({"respMsg": f"Unauthorized {KEY}", "respCode": "401"})))
    with pytest.raises(OpenApiAuthError) as e:
        _client(tmp_path, f).get(SERVICES["krx_dd_trd"], date(2026, 1, 5))
    assert len(f.calls) == 1 and KEY not in str(e.value)


def test_server_error_is_retried_then_succeeds_and_each_attempt_counts_toward_quota(tmp_path):
    f = Fake((500, "x"), (0, "URLError"), (200, OK))
    c = _client(tmp_path, f)
    assert c.get(SERVICES["krx_dd_trd"], date(2026, 1, 5)) == OK
    assert c.quota.calls == 3


def test_retries_exhausted_raises(tmp_path):
    f = Fake((503, "busy"))
    with pytest.raises(OpenApiError, match="재시도 실패"):
        _client(tmp_path, f).get(SERVICES["krx_dd_trd"], date(2026, 1, 5))
    assert len(f.calls) == 3


def test_unexpected_shape_raises_without_retry(tmp_path):
    f = Fake((200, json.dumps({"respCode": "999"})))
    with pytest.raises(OpenApiError, match="OutBlock_1"):
        _client(tmp_path, f).get(SERVICES["krx_dd_trd"], date(2026, 1, 5))
    assert len(f.calls) == 1


def test_quota_stops_before_the_call_and_persists_across_instances(tmp_path):
    f = Fake((200, OK))
    c = _client(tmp_path, f, limit=2)
    c.get(SERVICES["krx_dd_trd"], date(2026, 1, 5))
    c.get(SERVICES["krx_dd_trd"], date(2026, 1, 6))
    with pytest.raises(QuotaExceeded):
        c.get(SERVICES["krx_dd_trd"], date(2026, 1, 7))
    assert len(f.calls) == 2
    assert Quota(tmp_path / "q", 2, today=date(2026, 10, 30)).calls == 2
    assert Quota(tmp_path / "q", 2, today=date(2026, 10, 31)).calls == 0   # 다음 날은 새로 센다


def test_load_key_prefers_env_and_reads_only_that_line(tmp_path, monkeypatch):
    envf = tmp_path / ".env"
    envf.write_text("KRX_ID=someone\nKRX_OPENAPI_KEY=\"abc\"\n")
    monkeypatch.delenv("KRX_OPENAPI_KEY", raising=False)
    assert load_key(envf) == "abc"
    monkeypatch.setenv("KRX_OPENAPI_KEY", "fromenv")
    assert load_key(envf) == "fromenv"
    monkeypatch.delenv("KRX_OPENAPI_KEY")
    with pytest.raises(OpenApiError):
        load_key(tmp_path / "none")


def test_unverified_services_need_explicit_verification(monkeypatch):
    from src.data.openapi_client import Service
    monkeypatch.setitem(SERVICES, "new_svc", Service("new_svc", "sto/new_svc", "trading", False, "추정"))
    co.check_runnable(["krx_dd_trd", "stk_bydd_trd"], [])                 # 실제 응답으로 확인된 서비스는 바로 실행된다
    with pytest.raises(OpenApiError, match="Spec 확인 전"):
        co.check_runnable(["krx_dd_trd", "new_svc"], [])
    co.check_runnable(["krx_dd_trd", "new_svc"], ["new_svc"])
    with pytest.raises(OpenApiError, match="알 수 없는"):
        co.check_runnable(["nope"], [])


def test_weekdays_skips_weekends():
    assert co.weekdays(date(2026, 10, 30), date(2026, 11, 3)) == [date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 3)]


def _seed(out, service, resp):
    for d, body in resp.items():
        co.write_raw(co.raw_path(out, service, d), body)


def test_fetch_saves_verbatim_skips_existing_and_keeps_empty_days(tmp_path):
    out = tmp_path / "raw"
    d1, d2, d3 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
    _seed(out, "krx_dd_trd", {d1: "KEEP"})      # 이미 있는 파일은 덮어쓰지 않는다
    f = Fake((200, OK), (200, EMPTY))
    fails = []
    n = co.fetch_all(_client(tmp_path, f), out, "krx_dd_trd", [d1, d2, d3], 0, fails, sleep=lambda s: None)
    assert n == 2 and len(f.calls) == 2 and fails == []
    assert co.raw_path(out, "krx_dd_trd", d1).read_text() == "KEEP"
    assert co.raw_path(out, "krx_dd_trd", d3).read_text() == EMPTY   # 빈 날도 저장해 재호출을 막는다
    assert not list(out.rglob("*.tmp"))


def test_fetch_aborts_after_consecutive_failures_but_records_them(tmp_path):
    f = Fake((400, "bad"))
    fails = []
    ds = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8)]
    with pytest.raises(OpenApiError, match="연속 3회"):
        co.fetch_all(_client(tmp_path, f), tmp_path / "raw", "krx_dd_trd", ds, 0, fails, sleep=lambda s: None)
    assert len(fails) == 3 and len(f.calls) == 3


def test_trading_days_from_calendar_service_and_signal_days(tmp_path):
    out = tmp_path / "raw"
    days = co.weekdays(date(2026, 10, 26), date(2026, 11, 6))   # 평일 10일
    holiday = date(2026, 11, 5)
    _seed(out, "kospi_dd_trd", {d: (EMPTY if d == holiday else OK) for d in days})
    td = co.calendar_days(out, "kospi_dd_trd", days)
    assert holiday not in td and len(td) == 9
    assert co.dates_for("trading", days, td) == td
    assert co.dates_for("signal", days, td) == [date(2026, 10, 30), date(2026, 11, 6)]   # 주 마지막 거래일
    other = json.dumps({"OutBlock_1": [{"IDX_NM": "코스닥", "CLSPRC_IDX": "-"}]})
    _seed(out, "kospi_dd_trd", {days[0]: other})
    assert days[0] not in co.calendar_days(out, "kospi_dd_trd", days, idx_name="코스피 200")


def test_calendar_missing_days_is_reported(tmp_path):
    days = co.weekdays(date(2026, 10, 26), date(2026, 10, 28))
    with pytest.raises(OpenApiError, match="받지 못한 날"):
        co.calendar_days(tmp_path / "raw", "kospi_dd_trd", days)


def test_dry_run_makes_no_network_call(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(co, "ROOT", tmp_path)
    def boom(*a, **k):
        raise AssertionError("dry-run에서 키를 읽거나 호출하면 안 된다")
    monkeypatch.setattr(co, "load_key", boom)
    rc = co.main(["--start", "2026-10-26", "--end", "2026-10-30", "--snapshot", "s", "--services", "krx_dd_trd",
                  "--daily-limit", "10", "--dry-run"])
    assert rc == 0 and "krx_dd_trd: 남은 호출 5" in capsys.readouterr().out


def test_main_stops_with_status_2_on_auth_error_and_writes_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(co, "ROOT", tmp_path)
    monkeypatch.setattr(co, "load_key", lambda p: KEY)
    f = Fake((401, json.dumps({"respMsg": "Unauthorized API Call", "respCode": "401"})))
    monkeypatch.setattr(co, "OpenApiClient", lambda key, quota: OpenApiClient(key, quota, transport=f, sleep=lambda s: None))
    rc = co.main(["--start", "2026-10-26", "--end", "2026-10-30", "--snapshot", "s", "--services", "krx_dd_trd",
                  "--daily-limit", "10", "--interval", "0"])
    meta = list((tmp_path / "data/raw/s/openapi").glob("meta_*.json"))
    assert rc == 2 and len(f.calls) == 1 and len(meta) == 1
    assert KEY not in meta[0].read_text()


def test_quota_rereads_file_each_call_and_writes_atomically(tmp_path):
    q1, q2 = Quota(tmp_path / "q", 5, today=date(2026, 10, 30)), Quota(tmp_path / "q", 5, today=date(2026, 10, 30))
    q1.take(); q2.take(); q1.take()
    assert q2.calls == 2 and q1.calls == 3 and Quota(tmp_path / "q", 5, today=date(2026, 10, 30)).calls == 3
    assert not list((tmp_path / "q").glob("*.tmp"))


def test_quota_corrupt_file_stops_instead_of_resetting_to_zero(tmp_path):
    d = tmp_path / "q"
    d.mkdir()
    (d / "2026-10-30.json").write_text('{"calls": ')
    with pytest.raises(OpenApiError, match="깨졌다"):
        Quota(d, 5, today=date(2026, 10, 30))


def test_rate_limited_response_stops_immediately_without_retry(tmp_path):
    from src.data.openapi_client import OpenApiRateLimited
    f = Fake((429, json.dumps({"respMsg": "Too Many Requests", "respCode": "429"})))
    c = _client(tmp_path, f)
    with pytest.raises(OpenApiRateLimited):
        c.get(SERVICES["krx_dd_trd"], date(2026, 1, 5))
    assert len(f.calls) == 1 and issubclass(OpenApiRateLimited, QuotaExceeded)   # 수집기는 QuotaExceeded와 같게 중단한다(종료 코드 3)


def test_empty_or_non_json_200_body_is_retried_as_transient(tmp_path):
    f = Fake((200, ""), (200, "<html>"), (200, OK))
    c = _client(tmp_path, f)
    assert c.get(SERVICES["krx_dd_trd"], date(2026, 1, 5)) == OK and len(f.calls) == 3
    with pytest.raises(OpenApiError, match="재시도 실패"):
        _client(tmp_path / "x", Fake((200, ""))).get(SERVICES["krx_dd_trd"], date(2026, 1, 5))
