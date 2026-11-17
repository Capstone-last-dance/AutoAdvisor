"""M6 LLM 근거 요약 (명세 6.7절, PM-11). PM 소유.

M6 템플릿 근거(`evidence/{t}.jsonl`)의 출처 공시만 입력으로 오픈소스 LLM(Q-10, Qwen3-1.7B)이 3문장 이하 요약을
쓰고, 규칙 검사를 통과한 요약만 쓴다. 통과하지 못하면 템플릿 요약을 그대로 둔다(6.7절 축소안).

- 입력 출처는 M6가 이미 t−19~t 거래일·t 15:30 KST 전 공시로 거른 것이다. 검사가 출처 밖 낱말·숫자를 막으므로
  요약에 t 15:30 이후 정보가 들어갈 수 없다.
- 검사: 3문장 이하 / 모든 문장이 실제 출처 번호로 끝나는 평서문 / 모든 출처 인용 / 문장의 낱말(조사·서술어 제외)이
  모두 인용 제목에 있음 / 출처에 없는 숫자 없음 / 금지 표현(가격 전망·매수·매도 권유·평가) 없음 /
  인용 제목에 없는 '정정·철회·취소·연장' 없음 / 정정 공시를 인용하면 문장에 '정정'을 씀.
- 위반이면 이유를 알려 한 번 다시 쓰게 하고, 그래도 위반이면 템플릿을 쓴다.
- 레코드 필드는 바꾸지 않는다(`io.EVIDENCE_KEYS`). 결과는 PM-13 채점 전까지 공식 근거와 따로
  `evidence/llm/{t}.jsonl`에 쓰고, 레코드별 방법·위반 이유는 `evidence/llm/{t}.report.json`에 남긴다.
- 이 모듈은 표준 라이브러리만 쓴다. torch·transformers는 `load_generator`에서만 읽는다(GPU 환경용).

사용 (저장소 루트, CUDA 환경):
    python -B -m src.rag.summarize --date YYYY-MM-DD --model-dir <Qwen3-1.7B 폴더> [--extra-path <패키지 폴더>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"   # 부록 3 Q-10
MAX_NEW_TOKENS = 160
EVIDENCE_KEYS = ("date", "ticker", "signal_type", "summary", "sources", "disclaimer")   # io.EVIDENCE_KEYS와 같다

SYSTEM = (
    "너는 한국 상장사 공시 목록을 요약하는 도우미다. 다음 규칙을 반드시 지켜라.\n"
    "1. 아래 번호가 붙은 공시 제목과 날짜에 적힌 사실만 쓴다. 제목에 없는 이유·금액·결과·해석을 만들지 않는다.\n"
    "2. 한국어 평서문 1~3문장으로 쓴다. 같은 종류의 공시는 한 문장으로 묶어도 된다.\n"
    "3. 모든 문장은 근거 공시 번호로 끝낸다. 예: '자기주식 취득 결정을 공시했다 [1].'\n"
    "4. 주가·가격 전망, 매수·매도 권유, 투자 의견, 좋다·나쁘다 같은 평가를 쓰지 않는다.\n"
    "5. 모든 공시 번호를 한 번 이상 인용한다. 제목을 그대로 옮기지 말고 '~을 공시했다' 형태의 문장으로 쓴다.\n"
    "6. '[기재정정]'·'[첨부정정]'은 앞서 낸 공시를 정정했다는 뜻이다.\n"
    "7. 날짜와 숫자는 쓰지 않는다. 날짜는 출처에 따로 표시된다.\n"
    "8. 요약만 출력한다. 머리말, 목록 기호, 설명을 붙이지 않는다.\n\n"
    "예시 입력:\n[1] 2024-05-10 주식소각결정\n[2] 2024-05-08 [기재정정]유형자산양수결정\n[3] 2024-05-08 자기주식처분결과보고서\n"
    "예시 출력:\n주식 소각 결정을 공시했다 [1]. 유형자산 양수 결정 공시를 정정했다 [2]. 자기주식 처분 결과를 보고했다 [3].\n"
    "제목에 '정정'이 없는 공시를 정정했다고 쓰지 마라.\n"
    "(예시의 내용은 다른 요약에 쓰지 마라. 실제 입력의 공시만 요약한다.)"
)

_BANNED = re.compile(
    r"매수|매도|추천|전망|목표\s*주가|목표가|상승|하락|급등|급락|호재|악재|투자\s*(?:의견|판단|매력)|"
    r"기대|우려|긍정|부정적|주목|관심|유망|리스크|위험|수혜"
)
_CITE = re.compile(r"\[(\d+)\]")
_SENTENCE_END = re.compile(r"((?:\s*\[\d+\])+)\s*[.。]?")
_FUNCTION_WORDS = {"공시했다", "안내했다", "정정했다", "보고했다", "제출했다", "신고했다", "발표했다", "공시하고",
                   "안내하고", "이를", "공시를", "공시", "했다", "및", "등", "관련", "사항", "대한", "또는", "각각",
                   "을", "를", "이", "가", "은", "는", "의", "에", "와", "과", "로", "으로", "도"}
_PARTICLE = re.compile(r"(?:으로|에서|에게|을|를|이|가|은|는|의|에|와|과|로|도)$")
_VERB_TAIL = re.compile(r"(?:공시|안내|정정|보고|제출|신고|발표)+$")


# ---------------------------------------------------------------------------
# 프롬프트와 검사
# ---------------------------------------------------------------------------

def user_prompt(record: dict) -> str:
    lines = [f"신호일: {record['date']}", "공시 목록:"]
    lines += [f"[{s['n']}] {s['timestamp'][:10]} {s['title']}" for s in record["sources"]]
    lines.append("위 공시를 규칙에 따라 요약하라.")
    return "\n".join(lines)


def sentences(text: str) -> list[str]:
    """출처 번호 묶음으로 끝나는 단위로 자른다. 번호 뒤에 남는 꼬리 글자는 마지막 원소로 남긴다."""
    out, pos = [], 0
    for m in _SENTENCE_END.finditer(text):
        out.append(text[pos:m.end()].strip())
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def _norm(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text.replace("ㆍ", "").replace("·", "").replace("・", "")).lower()


def _in_title(token: str, title: str, max_gap: int = 4) -> bool:
    """token이 제목에 그대로 있거나, 순서대로 있되 사이에 낀 글자가 max_gap 이하면 참 (영업실적 ⊂ 영업(잠정)실적)."""
    if token in title:
        return True
    for start in (i for i, ch in enumerate(title) if ch == token[0]):
        pos, gaps = start, 0
        for ch in token[1:]:
            nxt = title.find(ch, pos + 1)
            if nxt < 0:
                break
            gaps += nxt - pos - 1
            pos = nxt
        else:
            if gaps <= max_gap:
                return True
    return False


def unsupported_tokens(sentence: str, titles: str) -> list[str]:
    """문장에서 조사·서술어를 뺀 낱말 중 인용 제목에 없는 것. 비어 있으면 모든 낱말이 제목에 근거가 있다."""
    title = _norm(titles)
    missing = []
    for raw in re.findall(r"[0-9A-Za-z가-힣ㆍ·・]+", _CITE.sub(" ", sentence)):
        word = _norm(raw)
        if not word or word in _FUNCTION_WORDS or _in_title(word, title):
            continue
        if word.endswith(("했다", "하고")):                 # 안내공시했다·첨부정정했다: 앞 말만 제목에서 찾는다
            core = _VERB_TAIL.sub("", word[:-2])
            if not core or core in _FUNCTION_WORDS or _in_title(core, title):
                continue
        stem = _PARTICLE.sub("", word)
        if stem and (stem in _FUNCTION_WORDS or _in_title(stem, title)):
            continue
        missing.append(raw)
    return missing


def check(summary: str, record: dict) -> list[str]:
    """요약이 규칙을 어긴 이유 목록. 빈 목록이면 통과."""
    text = summary.strip()
    if not text:
        return ["빈 요약"]
    problems = []
    if "<think>" in text or "</think>" in text:
        problems.append("생각 과정 출력")
    parts = sentences(text)
    if len(parts) > 3:
        problems.append(f"{len(parts)}문장 (3문장 초과)")
    by_n = {int(s["n"]): s["title"] for s in record["sources"]}
    cited = set()
    for p in parts:
        cites = {int(c) for c in _CITE.findall(p)}
        cited |= cites
        if not cites or not re.search(r"\[\d+\]\s*[.。]?$", p):
            problems.append(f"출처 번호로 끝나지 않는 문장: {p[:40]}")
            continue
        if not cites <= set(by_n):
            problems.append(f"없는 출처 번호 {sorted(cites - set(by_n))}")
            continue
        if not re.search(r"다\s*(?:\[\d+\]\s*)+[.。]?$", p):
            problems.append(f"평서문(~다)으로 끝나지 않는 문장: {p[:40]}")
        titles = " ".join(by_n[n] for n in sorted(cites))
        missing = unsupported_tokens(p, titles)
        if missing:
            problems.append(f"인용 공시 제목에 없는 말 {missing}: {p[:40]}")
        for act in ("정정", "철회", "취소", "연장"):
            if act in p and act not in titles:
                problems.append(f"인용 공시에 없는 '{act}': {p[:40]}")
        if all("정정" in by_n[n] for n in cites) and "정정" not in p:
            problems.append(f"정정 공시를 정정이라고 쓰지 않은 문장: {p[:40]}")
    if set(by_n) - cited:
        problems.append(f"인용하지 않은 출처 {sorted(set(by_n) - cited)}")
    allowed = " ".join(f"{s['title']} {s['timestamp'][:10]}" for s in record["sources"])
    for word in set(_BANNED.findall(text)):
        if word not in allowed:
            problems.append(f"금지 표현 '{word}'")
    numbers = set(re.findall(r"\d+", allowed))
    numbers |= {str(int(x)) for x in numbers}
    for num in set(re.findall(r"\d+", _CITE.sub(" ", text))):
        if num not in numbers and str(int(num)) not in numbers:
            problems.append(f"출처에 없는 숫자 '{num}'")
    return problems


def clean(raw: str) -> str:
    """생각 과정·머리말·목록 기호를 떼고 한 줄로 만든 뒤, 내용이 같은 문장은 출처 번호를 모아 하나로 합친다."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    text = re.sub(r"^\s*(?:요약\s*[:：]|[-*•]\s*)", "", text, flags=re.M)
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    merged: dict[str, list[str]] = {}
    for p in sentences(text):
        body = _CITE.sub("", p).strip().rstrip(".。").strip()
        if not _CITE.search(p):                            # 번호 없는 꼬리는 그대로 두어 검사에서 걸리게 한다
            merged.setdefault(p, [])
            continue
        cites = merged.setdefault(body, [])
        cites += [c for c in _CITE.findall(p) if c not in cites]
    return " ".join(f"{body} {''.join(f'[{c}]' for c in cites)}." if cites else body for body, cites in merged.items())


# ---------------------------------------------------------------------------
# 요약 실행
# ---------------------------------------------------------------------------

Generator = Callable[[list[dict]], str]


def summarize_records(records: list[dict], generate: Generator, retries: int = 1) -> tuple[list[dict], list[dict]]:
    """레코드마다 LLM 요약을 시도해 (새 레코드 목록, 레코드별 기록)을 돌려준다. 새 레코드는 summary만 다르다."""
    out, report = [], []
    for rec in records:
        row = {"date": rec["date"], "ticker": rec["ticker"], "n_sources": len(rec["sources"]), "attempts": []}
        if not rec["sources"]:
            out.append(dict(rec))
            report.append({**row, "method": "template_no_sources"})
            continue
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(rec)}]
        for _ in range(retries + 1):
            start = time.time()
            raw = generate(messages)
            text = clean(raw)
            problems = check(text, rec)
            row["attempts"].append({"raw": raw, "summary": text, "problems": problems,
                                    "sec": round(time.time() - start, 2)})
            if not problems:
                break
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": "다음 규칙 위반을 고쳐 다시 써라: " + "; ".join(problems)}]
        last = row["attempts"][-1]
        ok = not last["problems"]
        method = ("llm" if len(row["attempts"]) == 1 else "llm_retry") if ok else "template_fallback"
        out.append({**rec, "summary": last["summary"] if ok else rec["summary"]})
        report.append({**row, "method": method})
    return out, report


def load_generator(model_dir: str | Path, extra_path: str | Path | None = None) -> Generator:
    """Qwen3-1.7B를 CUDA bf16으로 읽는다. non-thinking, 탐욕 디코딩(결정적)."""
    if extra_path:
        sys.path.append(str(extra_path))              # 환경에 없는 패키지만 뒤쪽 경로에서 읽는다
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 쓸 수 없다. LLM 요약은 GPU 환경에서 실행한다")
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16)
    model = model.to("cuda").eval()

    @torch.no_grad()
    def generate(messages: list[dict]) -> str:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt").to("cuda")
        ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
        return tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def read_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for rec in records:
        if tuple(rec) != EVIDENCE_KEYS:
            raise ValueError(f"{path}: 근거 레코드 필드가 {EVIDENCE_KEYS}와 다르다: {tuple(rec)}")
    return records


def write_outputs(date: str, records: list[dict], report: list[dict], meta: dict, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{date}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    tmp.replace(path)
    counts = {m: sum(r["method"] == m for r in report) for m in sorted({r["method"] for r in report})}
    (folder / f"{date}.report.json").write_text(
        json.dumps({**meta, "counts": counts, "records": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="M6 LLM 근거 요약 (PM-11, 6.7절)")
    ap.add_argument("--date", required=True, help="신호일 YYYY-MM-DD (evidence/{date}.jsonl을 읽는다)")
    ap.add_argument("--model-dir", required=True, help=f"{MODEL_ID}@{MODEL_REVISION[:7]} 로컬 폴더")
    ap.add_argument("--extra-path", help="환경에 없는 패키지(transformers 등)를 둔 폴더")
    ap.add_argument("--input", type=Path, help="기본 evidence/{date}.jsonl")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "evidence" / "llm")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        ap.error("--date는 YYYY-MM-DD")
    source = args.input or ROOT / "evidence" / f"{args.date}.jsonl"
    records = read_records(source)
    generate = load_generator(args.model_dir, args.extra_path)
    start = time.time()
    out, report = summarize_records(records, generate)
    meta = {"date": args.date, "input": str(source), "model": MODEL_ID, "revision": MODEL_REVISION,
            "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy, enable_thinking=False",
            "seconds": round(time.time() - start, 1)}
    path = write_outputs(args.date, out, report, meta, args.output_dir)
    counts = {m: sum(r["method"] == m for r in report) for m in sorted({r["method"] for r in report})}
    print(f"{args.date}: {len(out)}건 {counts} → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
