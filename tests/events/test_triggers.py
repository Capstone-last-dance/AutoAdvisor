"""M9 제목 판정 테스트. 실제 사전등록 설정(configs/prereg.yaml)의 events를 쓴다."""

import pytest

from src.events.triggers import ALERT, EXCLUDED, NEEDS_DOCUMENT, classify, strip_prefixes
from src.portfolio.params import load_prereg


@pytest.fixture(scope="module")
def events():
    return load_prereg()["events"]


def only(title, events):
    decisions = classify(title, events)
    assert len(decisions) == 1, decisions
    return decisions[0]


def test_strip_prefixes_repeats():
    prefixes = ["[첨부추가]", "[정정명령부과]"]
    assert strip_prefixes("[정정명령부과][첨부추가]부도발생", prefixes) == "부도발생"
    assert strip_prefixes("[정정명령부과][기재정정]부도발생", prefixes) == "[기재정정]부도발생"


@pytest.mark.parametrize("title,kind", [
    ("상장적격성실질심사대상결정", "T2"),
    ("상장폐지사유발생", "T2"),
    ("횡령ㆍ배임혐의발생".replace("ㆍ", ""), "T3"),
    ("주요사항보고서(회생절차개시신청)", "T4"),
    ("주요사항보고서(부도발생)", "T5"),
])
def test_new_events_become_alerts(title, kind, events):
    decision = only(title, events)
    assert (decision.trigger_type, decision.status) == (kind, ALERT)


def test_prefix_is_stripped_before_judging(events):
    assert only("[첨부추가]주요사항보고서(부도발생)", events).status == ALERT


@pytest.mark.parametrize("title", ["[기재정정]주요사항보고서(부도발생)", "[첨부정정]주요사항보고서(부도발생)",
                                   "[정정명령부과][기재정정]주요사항보고서(부도발생)"])
def test_amendments_are_excluded(title, events):
    decision = only(title, events)
    assert decision.status == EXCLUDED and "후속 정정" in decision.reason


def test_withdrawal_is_excluded_for_every_type(events):
    assert only("주요사항보고서(회생절차개시신청)철회", events).status == EXCLUDED


@pytest.mark.parametrize("title", ["자회사의주요경영사항(회생절차개시신청)", "주요사항보고서(회생절차개시신청기각)"])
def test_t4_type_exclusions(title, events):
    assert only(title, events).status == EXCLUDED


def test_subsidiary_default_is_excluded_for_t5(events):
    assert only("자회사의주요경영사항(부도발생)", events).status == EXCLUDED


def test_t2_requires_any_term(events):
    decision = only("상장적격성실질심사관련기타", events)
    assert decision.status == EXCLUDED and decision.reason == "require_any 없음"


def test_t2_exclude_terms(events):
    assert only("상장적격성실질심사대상제외결정", events).status == EXCLUDED


def test_t1_needs_original_document(events):
    decision = only("관리종목지정", events)
    assert (decision.trigger_type, decision.status) == ("T1", NEEDS_DOCUMENT)


def test_t1_title_exclusions_come_first(events):
    assert only("관리종목지정우려", events).status == EXCLUDED


def test_t3_rumor_needs_company_answer(events):
    decision = only("횡령배임혐의발생조회공시답변", events)
    assert (decision.trigger_type, decision.status) == ("T3", NEEDS_DOCUMENT)


def test_t3_counts_once_for_two_keywords(events):
    decisions = classify("횡령배임혐의발생", events)
    assert [d.trigger_type for d in decisions] == ["T3"]


def test_t6_needs_document(events):
    decision = only("주요사항보고서(유상증자결정)", events)
    assert (decision.trigger_type, decision.status) == ("T6", NEEDS_DOCUMENT)
    assert "DEC-10" in decision.reason


def test_unrelated_title_has_no_decision(events):
    assert classify("주요사항보고서(자기주식취득결정)", events) == []
