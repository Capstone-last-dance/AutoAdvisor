"""오류 분류, 호출 간격, 예산과 인증 정보 보호 검증."""

import traceback
from datetime import timedelta

import pytest
import requests

from conftest import FakeResponse, item, page
from src.data.dart_client import (CallBudget, DailyBudgetExceeded, DartAuthError, DartClient,
                                  DartRateLimitError, DartRequestError, DartUnavailableError, ListRequest)

REQ = ListRequest("20250101", "20250331", "B", 1)


def test_params_exclude_key_and_fix_flags():
    assert REQ.params() == {"bgn_de": "20250101", "end_de": "20250331", "pblntf_ty": "B",
                            "page_no": "1", "page_count": "100", "last_reprt_at": "N"}
    assert "pblntf_ty" not in ListRequest("20250101", "20250101", None, 1).params()
    with pytest.raises(ValueError):
        ListRequest("20250101", "20250331", "B", 1, last_reprt_at="Y")


def test_min_interval_between_request_starts(client_factory):
    client, session, _ = client_factory([page(), page(status="900"), page()])
    client.fetch_list_page(REQ)
    client.fetch_list_page(REQ)
    assert all(b - a >= 1.1 for a, b in zip(session.starts, session.starts[1:]))
    assert client.calls == 3


def test_slow_response_needs_no_extra_wait(client_factory):
    client, session, clock = client_factory([page(), page()], elapsed=2)
    client.fetch_list_page(REQ)
    client.fetch_list_page(REQ)
    assert session.starts == [0, 2]
    assert not clock.sleeps


def test_013_returns_empty_page(client_factory):
    client, _, _ = client_factory([page(status="013")])
    result = client.fetch_list_page(REQ)
    assert (result.total_count, result.total_page, result.items) == (0, 0, ())


def test_020_raises_without_retry(client_factory):
    client, session, _ = client_factory([page(status="020")])
    with pytest.raises(DartRateLimitError):
        client.fetch_list_page(REQ)
    with pytest.raises(DartRateLimitError):
        client.fetch_list_page(REQ)
    assert len(session.requests) == 1


@pytest.mark.parametrize("status", ["010", "011", "012", "901"])
def test_auth_statuses(client_factory, monkeypatch, status):
    client, session, _ = client_factory([page(status=status)])
    with pytest.raises(DartAuthError):
        client.fetch_list_page(REQ)
    assert len(session.requests) == 1
    monkeypatch.delenv("OPENDART_API_KEY", raising=False)
    with pytest.raises(DartAuthError):
        DartClient()


@pytest.mark.parametrize("status", ["100", "101", "021"])
def test_request_statuses(client_factory, status):
    client, _, _ = client_factory([page(status=status)])
    with pytest.raises(DartRequestError):
        client.fetch_list_page(REQ)


@pytest.mark.parametrize("bad", [page(status="900"), requests.ConnectionError("보호할 URL"),
                                 requests.Timeout(), requests.exceptions.ChunkedEncodingError(),
                                 FakeResponse(content=b"broken"), FakeResponse({}, status=503),
                                 FakeResponse({}, status=429), {}, {"status": "000"}])
def test_transient_errors_retry_then_unavailable(client_factory, bad):
    client, session, clock = client_factory([bad] * 4)
    with pytest.raises(DartUnavailableError):
        client.fetch_list_page(REQ)
    assert len(session.requests) == 4
    assert clock.sleeps == [5, 15, 45]


def test_maintenance_backoff(client_factory):
    client, _, clock = client_factory([page(status="800")] * 3)
    with pytest.raises(DartUnavailableError):
        client.fetch_list_page(REQ)
    assert clock.sleeps == [60, 300]


def test_page_no_echo_mismatch(client_factory):
    client, session, _ = client_factory([page(number=4)])
    with pytest.raises(DartRequestError):
        client.fetch_list_page(ListRequest("20250101", "20250331", "B", 5))
    assert len(session.requests) == 1


def test_key_never_leaks(client_factory, caplog):
    secret = "TESTKEY-not-real"
    error = requests.ConnectionError(f"https://example.invalid/?crtfc_key={secret}")
    client, _, _ = client_factory([error] * 4)
    try:
        client.fetch_list_page(REQ)
    except DartUnavailableError as exc:
        assert secret not in str(exc) + repr(exc) + "".join(traceback.format_exception(exc))
        assert exc.__suppress_context__
    else:
        pytest.fail("예외가 필요합니다")
    assert secret not in repr(client) + caplog.text
    assert client._redact(f"crtfc_key={secret}&other=1") == "crtfc_key=***&other=1"


def test_budget_blocks_before_request(client_factory):
    client, session, clock = client_factory([page()], limit=1)
    client.fetch_list_page(REQ)
    with pytest.raises(DailyBudgetExceeded):
        client.fetch_list_page(REQ)
    assert len(session.requests) == 1
    clock.origin += timedelta(days=1)
    budget = CallBudget(1, client.budget.used_by_date)
    budget.consume(clock.now())
    assert len(budget.used_by_date) == 2


def test_item_content_not_retried(client_factory):
    client, session, _ = client_factory([page([item(rcept_no="잘못된 번호")])])
    assert client.fetch_list_page(REQ).items[0]["rcept_no"] == "잘못된 번호"
    assert len(session.requests) == 1
