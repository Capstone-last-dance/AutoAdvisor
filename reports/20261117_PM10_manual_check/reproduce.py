"""Rebuild the review CSV from manual findings and local source files. No network access."""

import csv
import hashlib
import json
import re
import zipfile
from bisect import bisect_left
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pyarrow.parquet as pq
import yaml

from fetch_documents import LEDGER, ROOT, SAMPLE, STORE, count_attempts


OUT = ROOT / "reports/20261117_PM10_manual_check"
PROCESSED = ROOT / "data/processed"
ARCHIVE = ROOT / "data/raw/2026-10-29/20261029_PM_disclosures_full/opendart_list"
DART = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo="
KST = timezone(timedelta(hours=9))

DESIGNATIONS = {"20190215900345": "2019-02-18", "20190220900765": "2019-02-21"}
RELATED = {"20151207900545", "20151215900882", "20200625000298", "20220330902526"}

# The basis is the filing text at each receipt number. The list, ticker and dates are checked below.
FINDINGS = {
    "20190215900345": ("경보", "거래소 소속부변경 원문에서 중견기업부 → 관리종목(소속부 없음), 신규 지정 효력일 2019-02-18 확인."),
    "20190220900765": ("경보", "거래소 원문의 당시 KJ프리텍이 중견기업부 → 관리종목(소속부 없음)으로 변경. 신규 지정 효력일 2019-02-21."),
    "20210302901016": ("경보 아님", "기존 관리종목의 자본전액잠식 관련 상장폐지 가능성과 이미 추가된 사유를 안내. 신규 지정 및 효력일 없음."),
    "20210531901002": ("경보 아님", "2021년 1분기보고서 미제출은 기존 관리종목의 추가 사유. 2021-05-11 상장폐지 확정과 정리매매 보류를 안내."),
    "20220223900185": ("경보 아님", "당시 샘코는 이미 관리종목. 내부결산상 전액잠식 및 감사보고서에 따라 지정 사유가 추가될 가능성을 안내."),
    "20170323901140": ("정답", "거래소가 감사의견 거절에 따른 상장폐지 사유 발생을 신양오라컴디스플레이의 정지기간 변경 사유로 기재."),
    "20200714900329": ("정답", "한국제6호기업인수목적의 상장예비심사 청구서 미제출 사유가 관리종목 지정 후 1개월 내 해소되지 않아 상장폐지 사유로 공시."),
    "20210104900778": ("정답", "한국제7호기업인수목적의 상장예비심사 청구서 미제출 사유가 관리종목 지정 후 1개월 내 해소되지 않아 상장폐지 사유로 공시."),
    "20210421900592": ("정답", "거래소가 이큐셀을 상장적격성 실질심사 대상으로 결정."),
    "20220117900490": ("오답", "감사 전 내부결산에서 4개 사업연도 연속 영업손실에 따른 관리종목 지정 가능성만 공시. T2 상장폐지·실질심사 사유가 본문에 없음."),
    "20220214900665": ("정답", "내부결산에서 2021·2020년 법인세차감전계속사업손실률 409.1%·64.3%를 해당 상장폐지 사유로 명시. 감사 확인을 전제로 한 6.10절의 T2 자율공시."),
    "20220829900562": ("정답", "당시 녹원씨엔아이의 상장폐지 사유 발생을 거래소가 정지기간 변경 사유로 명시. 이의신청 또는 상장폐지 여부 결정일까지로 변경."),
    "20240314900706": ("정답", "당시 에스비더블유생명과학에 대해 실질심사 대상 사유가 발생해 거래정지를 대상 여부 결정일까지 변경."),
    "20240321902880": ("정답", "셀피글로벌의 기존 정지 사유에 2023사업연도 감사의견으로 인한 상장폐지 사유를 추가."),
    "20250911800037": ("정답", "거래소가 일양약품의 회계처리기준 위반 검찰 고발을 실질심사 사유로 공시. 심의 대상 여부는 결정 예정."),
    "20151207900024": ("경보 아님", "당시 매일유업 조회 요구의 12-07 미확정·12-15 부인 답변이 모두 12-07 08:57 요구를 명시. 전 부회장 횡령 기소는 별도 회사 건이라고 확인. 직원 개인비리(배임수재) 기소도 별도 사실로 기재."),
    "20170725900197": ("정답", "거래소가 당시 엠피그룹 전 회장의 횡령·배임 공소장을 확인. 07-05 조회 요구에 대한 확인 사항."),
    "20180810900731": ("정답", "지디가 전 대표 업무상 횡령·배임 혐의로 공시 당일 서울동부지검에 고소장을 제출."),
    "20220211801212": ("오답", "대우건설 전 임원 배임 등 사건의 항소심이 원심을 파기하고 무죄를 선고. 사실확인금액 공란, 앞선 사건의 후속 공시."),
    "20220629900585": ("정답", "당시 유테크가 전임 대표 등 횡령·배임 혐의로 공시 당일 서울중앙지검에 고소장을 제출."),
    "20240429900861": ("정답", "거래소가 디에이테크놀로지 전 대표 횡령·배임 공소장을 확인하고 04-11 조회 요구에 대한 확정답변을 갈음."),
    "20240802800572": ("정답", "남양유업이 전직 임직원의 횡령 등 혐의로 고소장을 접수했다고 공시."),
    "20250508800637": ("정답", "신풍제약 전직 임원 횡령·배임의 대법원 상고 기각과 원심 사실확인금액 9,768,189,844원 확인. 같은 사건의 후속 판결로 연결할 근거가 있음."),
    "20250623900427": ("정답", "소프트캠프 재무담당 직원의 업무상 횡령 정황과 혐의금액 927,439,286원을 확인. 고소장 제출은 예정."),
    "20251223800674": ("오답", "한일홀딩스 전현직 임직원 배임 혐의 2심 무죄, 사실확인금액 0원. 2021-11-25 혐의발생 공시의 후속."),
    "20160418900562": ("정답", "거래소가 당시 나노스의 회생절차개시신청에 따른 거래정지와 개시 결정일까지의 정지기간을 공시."),
    "20170906000036": ("정답", "엔에스브이 채권자가 2017-09-04 부산지법에 회생절차개시를 신청하고 보전처분 등도 접수."),
    "20200623000306": ("오답", "당시 지코의 06-23 원문은 06-25 신청 예정이며 관할법원도 미정. 06-25 정정 원문은 신청 예정일을 07-01로 변경."),
    "20210120000617": ("정답", "이엠네트웍스가 2021-01-20 서울회생법원에 회생절차개시를 신청. DART 목록 접수일은 01-21."),
    "20210426000580": ("정답", "당시 엔지스테크널러지가 2021-04-26 서울회생법원에 신청 접수. 사건번호 2021회합100050."),
    "20210917000574": ("정답", "매직마이크로가 2021-09-17 수원지법에 신청 접수. 사건번호 2021회합151."),
    "20220330001706": ("정답", "당시 유씨아이의 원문에 제출 계획·접수 문구가 함께 있음. 같은 날 거래소가 회생절차개시신청을 정지기간 변경 사유로 확인."),
    "20220404002389": ("정답", "소리바다가 2022-04-04 서울회생법원에 회생절차개시 및 보전처분을 신청했다고 공시."),
    "20240502000580": ("정답", "시스웍 채권자가 2024-04-26 서울회생법원에 신청. 사건번호 2024회합100045."),
    "20250326000938": ("정답", "제넨바이오가 2025-03-26 수원회생법원에 신청서를 접수. 사건번호 2025회합129."),
    "20150507000001": ("정답", "당시 티에스엠텍 만기어음 3,516,301,423원 미결제로 2015-05-06 최종부도 및 당좌거래정지."),
    "20191107000327": ("정답", "당시 데코앤이 전자어음 440,558,786원 결제 실패로 2019-11-07 최종부도."),
    "20210503000565": ("정답", "맥스로텍 만기어음 1,273,393,114원 예금 부족으로 2021-05-03 최종부도."),
    "20210901000478": ("정답", "자안바이오 전자어음 1,286,809,300원 미결제로 2021-09-01 최종부도."),
    "20230927000802": ("정답", "당시 대유플러스 김제지점 어음 8,778,000원이 보전명령에 따른 결제 제한으로 최종부도 처리."),
    "20231107000224": ("정답", "당시 대유플러스 김제지점의 별도 만기어음 101,585,000원이 같은 법적 지급제한으로 부도 처리. 최종부도에 따른 정지 대상은 아님."),
    "20240226006299": ("정답", "태영건설 CP 6,000,000,000원이 워크아웃 상환유예로 부도 처리. 최종부도에 따른 정지 대상은 아님."),
    "20240502000583": ("정답", "디에이테크놀로지 어음 10,434,196,568원이 보전명령에 따른 법적 지급제한으로 2024-05-02 부도 처리."),
    "20250508000336": ("정답", "동성제약 어음 103,480,000원 예금 부족으로 2025-05-08 1차 부도가 확인됐고 같은 날 입금."),
    "20250611000115": ("정답", "동성제약의 다른 만기어음 383,200,000원이 법적 지급제한으로 2025-06-11 부도 처리."),
}

# Manual transcription of the issuer, method and same-version terms in §1, §5, §6.
# None denotes an unverified per-share price or a missing applicable ticker.
T6_TERMS = {
    "20150212000377": ("코오롱웰케어", "", "주주배정증자", "주주배정", 400_000, 5_000, "기재가", "동일 발행인 종목코드와 전일 시가총액을 확인할 수 없음."),
    "20150423000326": ("넥솔론", "110570", "제3자배정증자", "제3자배정", 676_610, 1_000, "기재가", "회생채권 출자전환."),
    "20160610000459": ("태양씨앤엘", "072520", "제3자배정증자", "제3자배정", 990_099, 1_515, "기재가", "표본의 제넨바이오와 같은 종목코드의 당시 상호."),
    "20170404001574": ("위지트", "036090", "제3자배정증자", "제3자배정", 3_613_369, 1_107, "기재가", ""),
    "20180704900558": ("케이박스", "", "제3자배정증자", "제3자배정", 1_600_000, None, "", "JTC 종속회사 발행. 발행가액 칸의 160억원이 운영자금 총액과 같아 주당가로 사용하지 않음."),
    "20190823000185": ("코스모신소재", "005070", "주주배정후 실권주 일반공모", "주주배정", 9_500_000, 8_770, "예정가", "최초 접수 버전. 확정가는 공란."),
    "20200408001955": ("다이노나", "", "주주우선공모증자", None, 7_888_878, 2_850, "예정가", "원문에서 별도 증자방식으로 기재. DART 목록의 코넥스 086080은 가격표 범위 밖이어서 전일 시가총액 없음."),
    "20201028000468": ("데일리블록체인", "139050", "제3자배정증자", "제3자배정", 7_122_507, 702, "기재가", "표본의 비에프랩스와 같은 종목코드의 당시 상호."),
    "20230816900206": ("오픈엣지스퀘어", "", "주주배정증자", "주주배정", 359_980, 50_000, "확정가", "오픈엣지테크놀로지가 100% 보유한 신설 종속회사의 현물출자. 발행인 전일 시가총액 없음."),
    "20231005900137": ("티에네스", "", "제3자배정증자", "제3자배정", 326_087, 11_500, "기재가", "덴티스의 비상장 종속회사로, 종류주식(RCPS) 발행."),
}

SUPPLEMENT = {
    "20151207900024": [DART + "20151207900545", DART + "20151215900882"],
    "20200623000306": [DART + "20200625000298"],
    "20220330001706": [DART + "20220330902526"],
    "20190823000185": ["https://kind.krx.co.kr/external/2019/08/23/000375/20190823000971/10001.htm"],
    "20200408001955": ["https://kind.krx.co.kr/external/2019/11/25/000391/20191125000988/10601.htm"],
}


def get_receipt(row):
    return re.search(r"rcpNo=(\d{14})$", row["url"])[1]


def from_dart_date(value):
    return datetime.strptime(value[:8], "%Y%m%d").date()


def list_observations(receipts, dates):
    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    observations = {}
    for name, window in manifest["windows"].items():
        if window["state"] != "complete":
            continue
        if not any(window["bgn"] <= day <= window["end"] for day in dates):
            continue
        for attempt in window["accepted_attempts"]:
            folder = ARCHIVE / "windows" / name / f"attempt_{attempt:02}"
            for path in sorted(folder.glob("page_*.json")):
                if not re.fullmatch(r"page_\d+\.json", path.name):
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                for item in data.get("list", []):
                    receipt = item["rcept_no"]
                    if receipt in receipts:
                        previous = observations.get(receipt)
                        if previous:
                            assert all(previous[key] == item[key] for key in ("corp_code", "stock_code", "rcept_dt"))
                        observations[receipt] = dict(item, archive=str(path.relative_to(ARCHIVE)))
    assert set(observations) == receipts, sorted(receipts - observations.keys())
    return observations


def date_only(value):
    return value.date() if isinstance(value, datetime) else value


def listing_on(security, day):
    start = date_only(security["listing_date"])
    end = date_only(security["delisting_date"])
    return start and start <= day and (end is None or day <= end)


def verify_document(receipt, digests):
    folder = STORE / "documents"
    zip_path = folder / f"{receipt}.zip"
    assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == digests[receipt], receipt
    with zipfile.ZipFile(zip_path) as archive:
        member = next(name for name in archive.namelist() if name.lstrip("/") == f"{receipt}.xml")
        assert (folder / f"{receipt}.xml").read_bytes() == archive.read(member), receipt


def evaluate_t6(receipt, filing_day, row, observation, days, rule):
    issuer, ticker, method, normalized, shares, price, price_stage, note = T6_TERMS[receipt]
    mcap_day = days[bisect_left(days, filing_day) - 1]
    assert mcap_day < filing_day
    mcap = None
    if ticker:
        assert ticker == row["ticker"] == observation["stock_code"], receipt
        values = pq.read_table(PROCESSED / "prices.parquet", columns=["mcap"],
                               filters=[("date", "=", mcap_day), ("ticker", "=", ticker)])["mcap"].to_pylist()
        if values:
            assert len(values) == 1 and values[0] > 0, (receipt, values)
            mcap = values[0]
    elif observation["stock_code"] == "086080":
        assert pq.read_table(PROCESSED / "prices.parquet", columns=["ticker"],
                             filters=[("date", "=", mcap_day), ("ticker", "=", "086080")]).num_rows == 0
    amount = shares * price if price else None
    ratio = Decimal(amount) / Decimal(mcap) if amount is not None and mcap is not None else None
    if normalized is None:
        label = "미판정"
    elif normalized not in rule["methods"]:
        label = "경보 아님"
    elif ratio is None:
        label = "미판정"
    else:
        label = "경보" if ratio >= Decimal(str(rule["min_amount_to_mcap"])) else "경보 아님"
    evidence = f"{issuer} {method}. {shares:,}주"
    if price is not None:
        evidence += f" × {price_stage} {price:,}원 = {amount:,}원."
    else:
        evidence += ", 주당 발행가 미검증."
    if ratio is not None:
        evidence += f" {mcap_day} MC {mcap:,}원 대비 {ratio:.4%}."
    if note:
        evidence += " " + note
    return label, evidence, dict(rcept_no=receipt, issuer=issuer, sample_ticker=row["ticker"], issuer_ticker=ticker,
                                rcept_date=filing_day, prev_krx_day=mcap_day, method=method, price_stage=price_stage,
                                shares=shares, issue_price=price, amount_won=amount, mcap_won=mcap,
                                ratio=ratio, manual_label=label)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with SAMPLE.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    receipts = {get_receipt(row) for row in rows}
    assert len(rows) == len(receipts) == 55 and all(not row["manual_check"] for row in rows)
    assert set(FINDINGS).isdisjoint(T6_TERMS) and receipts == set(FINDINGS) | set(T6_TERMS)
    dates = {row["rcept_date"] for row in rows} | {from_dart_date(receipt).isoformat() for receipt in RELATED}
    observations = list_observations(receipts | RELATED, dates)
    for reply in ("20151207900545", "20151215900882"):
        assert observations[reply]["corp_code"] == observations["20151207900024"]["corp_code"]
        assert observations[reply]["stock_code"] == "005990"
    ledger = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()]
    digests = {event["receipt"]: event["sha256"] for event in ledger if "sha256" in event}
    for receipt in sorted(receipts | RELATED):
        verify_document(receipt, digests)

    tickers = {row["ticker"]: row for row in pq.read_table(PROCESSED / "tickers.parquet").to_pylist()}
    calendar = pq.read_table(PROCESSED / "index.parquet", columns=["date"],
                             filters=[("index_code", "=", "KOSPI200")])["date"].to_pylist()
    days = sorted({date_only(day) for day in calendar})
    rule = yaml.safe_load((ROOT / "configs/prereg.yaml").read_text(encoding="utf-8"))["events"]["triggers"]["T6"]

    results, t6_audit, listing_audit = [], [], []
    for row in rows:
        receipt = get_receipt(row)
        obs = observations[receipt]
        received = date.fromisoformat(row["rcept_date"])
        assert received == from_dart_date(obs["rcept_dt"]), receipt
        available_date = max(received, from_dart_date(receipt))
        ticker = row["ticker"]
        effective = date.fromisoformat(DESIGNATIONS[receipt]) if receipt in DESIGNATIONS else None
        security = tickers.get(ticker)
        if ticker:
            assert ticker == obs["stock_code"] and security and listing_on(security, received), receipt
            if effective:
                assert listing_on(security, effective), receipt

        if row["trigger_type"] == "T6":
            label, evidence, calculation = evaluate_t6(receipt, available_date, row, obs, days, rule)
            t6_audit.append(calculation)
        else:
            label, evidence = FINDINGS[receipt]
            evidence += f" DART 목록 {ticker} 일치"
            if effective:
                start = date_only(security["listing_date"])
                end = date_only(security["delisting_date"])
                evidence += f", 접수·효력일 상장 중({start}~{end or '상장폐지일 공란'})"
            evidence += "."
        links = " ".join([row["url"], *SUPPLEMENT.get(receipt, [])])
        results.append(dict(row, manual_label=label, evidence=evidence + " " + links, checker="Codex"))
        listing_audit.append(dict(rcept_no=receipt, corp_code=obs["corp_code"], sample_ticker=ticker,
                                  observed_stock_code=obs["stock_code"], rcept_date=received,
                                  designation_effective=effective,
                                  available_at=datetime.combine(available_date, time(23, 59, 59), KST).isoformat(),
                                  listing_date=security and date_only(security["listing_date"]),
                                  delisting_date=security and date_only(security["delisting_date"]),
                                  archive_page=obs["archive"]))

    write_csv(OUT / "manual_check_result.csv", results)
    write_csv(STORE / "t6_calculations.csv", t6_audit)
    write_csv(STORE / "listing_and_timing.csv", listing_audit)
    counts = Counter((row["trigger_type"], row["final"], row["manual_label"]) for row in results)
    for key, count in sorted(counts.items()):
        print(*key, count)
    print(f"OpenDART document attempts: {count_attempts()}. Checked original documents: {len(receipts)} sample + {len(RELATED)} related.")


if __name__ == "__main__":
    main()
