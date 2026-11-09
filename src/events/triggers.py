"""M9 트리거 제목 판정 (명세서 6.10절, 부록 1 `events`). PM 소유.

정규화한 보고서명(`report_nm_norm`)과 사전등록 규칙만 본다. 원문이 있어야 판정할 수 있는 후보는
경보로 만들지 않고 NEEDS_DOCUMENT로 남긴다 (T1 신규 지정, T3 풍문·조회공시 답변, T6 증자방식·금액).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ALERT = "alert"
NEEDS_DOCUMENT = "needs_document"
EXCLUDED = "excluded"


@dataclass(frozen=True)
class Decision:
    trigger_type: str
    status: str
    reason: str = ""


def strip_prefixes(title: str, prefixes: Sequence[str]) -> str:
    """제목 앞의 표시를 앞에서부터 반복해 뗀다. `[정정명령부과][첨부추가]`는 둘 다 뗀다."""
    found = True
    while found:
        found = False
        for prefix in prefixes:
            if title.startswith(prefix):
                title = title[len(prefix):]
                found = True
    return title


def classify(title_norm: str, events: Mapping[str, Any]) -> list[Decision]:
    """보고서명 하나의 유형별 판정. 키워드가 없는 유형은 결과에 넣지 않는다."""
    filters = events["title_filters"]
    title = strip_prefixes(title_norm, filters["strip_prefixes"])
    common = _common_exclusion(title, filters)
    decisions = []
    for trigger_type, rule in events["triggers"].items():
        if not any(keyword in title for keyword in rule["keywords"]):
            continue
        if common:
            decisions.append(Decision(trigger_type, EXCLUDED, common))
        else:
            decisions.append(_judge(trigger_type, rule, title))
    return decisions


def _common_exclusion(title: str, filters: Mapping[str, Any]) -> str:
    for prefix in filters["exclude_prefixes"]:
        if title.startswith(prefix):
            return f"후속 정정 {prefix}"
    for term in filters["exclude_terms"]:
        if term in title:
            return f"공통 제외 {term}"
    return ""


def _judge(trigger_type: str, rule: Mapping[str, Any], title: str) -> Decision:
    for term in rule.get("require_company_confirmation_if_any", []):
        if term in title:
            return Decision(trigger_type, NEEDS_DOCUMENT, f"회사 답변 원문 확인 필요 ({term})")
    for term in rule.get("exclude_terms", []):
        if term in title:
            return Decision(trigger_type, EXCLUDED, f"유형 제외 {term}")
    required = rule.get("require_any")
    if required and not any(term in title for term in required):
        return Decision(trigger_type, EXCLUDED, "require_any 없음")
    if rule.get("require_confirmed_designation"):
        return Decision(trigger_type, NEEDS_DOCUMENT, "신규 지정 원문 확인 필요")
    if "min_amount_to_mcap" in rule:
        return Decision(trigger_type, NEEDS_DOCUMENT, "증자방식·발행금액 원문 확인 필요 (DEC-10)")
    return Decision(trigger_type, ALERT)
