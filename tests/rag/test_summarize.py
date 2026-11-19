"""M6 LLM 요약의 검사·대체 규칙 (6.7절, PM-11). 모델 없이 가짜 생성기로 검증한다."""
import json

import pandas as pd
import pytest

from src.common import io
from src.pipeline.run_weekly import StepContext
from src.rag import evidence
from src.rag import summarize as s
from tests.conftest import item
from tests.rag.test_evidence import AFTER_CUTOFF, DAY, T, archive_filings, inputs, ticker  # noqa: F401  (inputs는 fixture)


def record(*titles, date="2026-11-27", ticker="000001"):
    sources = [{"n": i, "type": "disclosure", "id": f"2026112{i}000001", "title": t,
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo=2026112{i}000001",
                "timestamp": f"2026-11-2{i}T23:59:59+09:00"} for i, t in enumerate(titles, 1)]
    return {"date": date, "ticker": ticker, "signal_type": "rank", "summary": "템플릿 요약", "sources": sources,
            "disclaimer": io.DISCLAIMER}


def test_record_keys_match_common_io():
    assert s.EVIDENCE_KEYS == io.EVIDENCE_KEYS
    assert tuple(record("a")) == io.EVIDENCE_KEYS


def test_faithful_summary_passes():
    rec = record("주요사항보고서(자기주식취득결정)", "연결재무제표기준영업(잠정)실적(공정공시)", "[기재정정]단일판매ㆍ공급계약체결")
    text = "자기주식 취득 결정을 공시했다 [1]. 연결 기준 영업(잠정)실적을 공시했다 [2]. 단일판매·공급계약 체결 공시를 정정했다 [3]."
    assert s.check(text, rec) == []


@pytest.mark.parametrize("text, title, expected", [
    ("유상증자 결정을 공시했다 [1].", "무상증자결정", "제목에 없는 말"),
    ("자기주식 취득 결정을 공시했다 [1].", "자기주식처분결정", "제목에 없는 말"),
    ("단일판매·공급계약 해지를 공시했다 [1].", "단일판매ㆍ공급계약체결", "제목에 없는 말"),
    ("전환사채권 발행 결정을 공시했다 [1].", "신주인수권부사채권발행결정", "제목에 없는 말"),
    ("기억사항(자율공시)을 공시했다 [1].", "기타경영사항(자율공시)", "제목에 없는 말"),      # 시험에서 실제로 나온 오류
    ("배당 결정을 정정했다 [1].", "현금ㆍ현물배당결정", "없는 '정정'"),
    ("자기주식 취득으로 주가 상승이 기대된다 [1].", "자기주식취득결정", "금지 표현"),
    ("자기주식 100억원 취득 결정을 공시했다 [1].", "자기주식취득결정", "출처에 없는 숫자"),
    ("자기주식 취득 결정을 공시했다 [2].", "자기주식취득결정", "없는 출처 번호"),
    ("자기주식 취득 결정을 공시했다.", "자기주식취득결정", "출처 번호로 끝나지 않는"),
    ("자기주식취득결정 [1].", "자기주식취득결정", "평서문"),
    ("단일판매·공급계약 체결을 공시했다 [1].", "[기재정정]단일판매ㆍ공급계약체결", "정정이라고 쓰지 않은"),
])
def test_wrong_summaries_are_caught(text, title, expected):
    problems = s.check(text, record(title))
    assert any(expected in p for p in problems), problems


def test_every_source_cited_and_at_most_three_sentences():
    rec = record("기업설명회(IR)개최(안내공시)", "주주총회소집결의")
    assert any("인용하지 않은 출처 [2]" in p for p in s.check("기업설명회 개최를 안내했다 [1].", rec))
    four = "기업설명회 개최를 안내했다 [1]. 주주총회 소집을 결의했다 [2]. 기업설명회 개최를 안내했다 [1]. 주주총회 소집을 결의했다 [2]."
    assert any("3문장 초과" in p for p in s.check(four, rec))


def test_title_numbers_are_allowed():
    rec = record("매출액또는손익구조30%(대규모법인은15%)이상변경")
    assert s.check("매출액 또는 손익구조 30%(대규모법인은 15%) 이상 변경을 공시했다 [1].", rec) == []


def test_clean_merges_duplicate_sentences_and_strips_thinking():
    raw = "<think>생각</think>\n- 기업설명회 개최를 안내했다 [2].\n기업설명회 개최를 안내했다 [3]. 주주총회 소집을 결의했다 [1]."
    assert s.clean(raw) == "기업설명회 개최를 안내했다 [2][3]. 주주총회 소집을 결의했다 [1]."


def fake(*answers):
    calls = []

    def generate(messages):
        calls.append([m["content"] for m in messages])
        return answers[len(calls) - 1]
    return generate, calls


def test_summarize_records_pass_retry_fallback_and_no_sources():
    good = record("자기주식취득결정", ticker="000001")
    retry = record("자기주식취득결정", ticker="000002")
    bad = record("자기주식취득결정", ticker="000003")
    empty = {**record(ticker="000004"), "summary": "최근 20거래일 공시: 조회된 근거가 없습니다."}
    generate, calls = fake("자기주식 취득 결정을 공시했다 [1].",
                           "자기주식 처분 결정을 공시했다 [1].", "자기주식 취득 결정을 공시했다 [1].",
                           "주가 상승이 기대된다 [1].", "주가 상승이 기대된다 [1].")
    out, report = s.summarize_records([good, retry, bad, empty], generate)

    assert [r["method"] for r in report] == ["llm", "llm_retry", "template_fallback", "template_no_sources"]
    assert [r["summary"] for r in out] == ["자기주식 취득 결정을 공시했다 [1].", "자기주식 취득 결정을 공시했다 [1].",
                                           "템플릿 요약", empty["summary"]]
    assert all(tuple(r) == io.EVIDENCE_KEYS for r in out) and out[0]["sources"] == good["sources"]
    assert len(calls) == 5 and "다음 규칙 위반을 고쳐" in calls[2][-1] and "처분" in calls[2][-1]
    assert report[2]["attempts"][-1]["problems"]


def test_m6_to_summary_input_uses_title_seen_before_cutoff(inputs, client_factory):  # noqa: F811
    """C #71 리뷰: 같은 공시의 제목이 컷오프 전·후에 다르면, M6 근거와 요약기 입력에는 컷오프 전 제목만 들어간다.

    컷오프 뒤·기록 마감 전에 받은 수정 제목('주식소각결정')은 출처에 없으므로, 그 말을 쓴 요약은 검사에서 막혀
    템플릿으로 돌아간다.
    """
    received = "20260216"                                  # t−19~t 창 안의 거래일
    archive_filings(client_factory, [item(day=received, stock_code=ticker(1), report_nm="자기주식취득결정")])
    archive_filings(client_factory, [item(day=received, stock_code=ticker(1), report_nm="주식소각결정")],
                    day=DAY + pd.Timedelta(days=1), observed_at=AFTER_CUTOFF)

    evidence.m6_step(StepContext(T))

    rec, = s.read_records(io.ROOT / io.EVIDENCE_FOLDER / f"{T}.jsonl")
    assert [source["title"] for source in rec["sources"]] == ["자기주식취득결정"]
    prompt = s.user_prompt(rec)
    assert "자기주식취득결정" in prompt and "소각" not in prompt

    generate, calls = fake("주식 소각 결정을 공시했다 [1].", "주식 소각 결정을 공시했다 [1].")
    out, report = s.summarize_records([rec], generate)
    assert report[0]["method"] == "template_fallback" and len(calls) == 2
    assert any("제목에 없는 말 ['소각']" in p for p in report[0]["attempts"][0]["problems"])
    assert out[0] == rec                                   # 템플릿 근거 그대로

    generate, _ = fake("자기주식 취득 결정을 공시했다 [1].")
    out, report = s.summarize_records([rec], generate)
    assert report[0]["method"] == "llm" and out[0]["summary"] == "자기주식 취득 결정을 공시했다 [1]."


def test_main_writes_separate_llm_file_and_report(tmp_path, monkeypatch):
    src = tmp_path / "2026-11-27.jsonl"
    rows = [record("자기주식취득결정", ticker="000001"), record("주식소각결정", ticker="000002")]
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    generate, _ = fake("자기주식 취득 결정을 공시했다 [1].", "주식 소각 결정을 공시했다 [1].")
    monkeypatch.setattr(s, "load_generator", lambda model_dir, extra_path=None: generate)

    assert s.main(["--date", "2026-11-27", "--model-dir", "m", "--input", str(src), "--output-dir", str(tmp_path / "llm")]) == 0

    written = [json.loads(line) for line in (tmp_path / "llm" / "2026-11-27.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["summary"] for r in written] == ["자기주식 취득 결정을 공시했다 [1].", "주식 소각 결정을 공시했다 [1]."]
    io._validate_evidence(written, "2026-11-27")
    report = json.loads((tmp_path / "llm" / "2026-11-27.report.json").read_text(encoding="utf-8"))
    assert report["counts"] == {"llm": 2} and report["revision"] == s.MODEL_REVISION
    assert src.read_text(encoding="utf-8").count("템플릿 요약") == 2          # 공식 근거 파일은 그대로


def test_read_records_rejects_other_fields(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"date": "2026-11-27", "ticker": "000001"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="필드"):
        s.read_records(path)
