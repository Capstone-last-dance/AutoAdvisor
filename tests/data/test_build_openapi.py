import json
from datetime import date

import pandas as pd
import pytest

from src.data.build_openapi import build_all, require_complete_files
from src.data.collect_openapi import raw_path, write_raw
from src.data.schemas import validate

DAYS = [date(2025, 1, d) for d in (6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
HOLIDAY = date(2025, 1, 10)         # 금요일 휴장: 지수 응답이 비어 있다
SIG = [date(2025, 1, 9), date(2025, 1, 17)]


def _px(code, name, mkt, close, ret, vol=1000):
    return {"BAS_DD": "", "ISU_CD": code, "ISU_NM": name, "MKT_NM": mkt, "SECT_TP_NM": "", "TDD_CLSPRC": str(close),
            "CMPPREVDD_PRC": "0", "FLUC_RT": str(ret), "TDD_OPNPRC": str(close), "TDD_HGPRC": str(close), "TDD_LWPRC": str(close),
            "ACC_TRDVOL": str(vol), "ACC_TRDVAL": str(close * vol), "MKTCAP": str(close * 1_000_000), "LIST_SHRS": "1000000"}


def _base(code, name, mkt, secu, kind, listed="19900101"):
    return {"ISU_CD": "KR7" + code, "ISU_SRT_CD": code, "ISU_NM": name, "ISU_ABBRV": name, "ISU_ENG_NM": name, "LIST_DD": listed,
            "MKT_TP_NM": mkt, "SECUGRP_NM": secu, "SECT_TP_NM": "", "KIND_STKCERT_TP_NM": kind, "PARVAL": "100", "LIST_SHRS": "1000000"}


def _make(raw, drop=None, patch=None):
    for d in DAYS:
        idx = [] if d == HOLIDAY else [{"BAS_DD": f"{d:%Y%m%d}", "IDX_NM": "코스피", "CLSPRC_IDX": "1.0"},
                                       {"BAS_DD": f"{d:%Y%m%d}", "IDX_NM": "코스피 200", "CLSPRC_IDX": "300.5"}]
        idx_kq = [] if d == HOLIDAY else [{"BAS_DD": f"{d:%Y%m%d}", "IDX_NM": "코스닥", "CLSPRC_IDX": "1.0"},
                                          {"BAS_DD": f"{d:%Y%m%d}", "IDX_NM": "코스닥 150", "CLSPRC_IDX": "800.25"}]
        kospi, kosdaq = [], []
        if d != HOLIDAY:
            kospi = [_px("000001", "가", "KOSPI", 1000, 0.0), _px("000002", "가우", "KOSPI", 500, 0.0),
                     _px("000006", "무", "KOSPI", 300, 0.0), _px("000005", "정", "KOSPI", 200, 0.0, vol=0 if d == SIG[1] else 1000)]
            kosdaq = [_px("000003", "나", "KOSDAQ", 1000 if d < date(2025, 1, 14) else 500, 0.0)]    # 1월 14일 2:1 분할(등락률 0, 기준가 500)
            if d <= date(2025, 1, 13):
                kospi.append(_px("000004", "폐", "KOSPI", 100, 0.0))                              # 1월 13일이 마지막 시세
        for r in kospi + kosdaq:
            r["BAS_DD"] = f"{d:%Y%m%d}"
        for svc, rows in (("kospi_dd_trd", idx), ("kosdaq_dd_trd", idx_kq), ("stk_bydd_trd", kospi), ("ksq_bydd_trd", kosdaq)):
            write_raw(raw_path(raw, svc, d), json.dumps({"OutBlock_1": rows}, ensure_ascii=False))
    for d in SIG:
        stk = [_base("000001", "가", "KOSPI", "주권", "보통주"), _base("000002", "가우", "KOSPI", "주권", "구형우선주"),
               _base("000005", "정", "KOSPI", "주권", "보통주")]
        if d == SIG[0]:
            stk.append(_base("000004", "폐", "KOSPI", "주권", "보통주"))
        ksq = [_base("000003", "나", "KOSDAQ GLOBAL", "주권", "보통주")]
        write_raw(raw_path(raw, "stk_isu_base_info", d), json.dumps({"OutBlock_1": stk}, ensure_ascii=False))
        write_raw(raw_path(raw, "ksq_isu_base_info", d), json.dumps({"OutBlock_1": ksq}, ensure_ascii=False))
    if patch:
        patch(raw)


@pytest.fixture()
def built(tmp_path):
    _make(tmp_path)
    return build_all(tmp_path, DAYS[0], DAYS[-1], first_signal_year=2025)


def test_all_tables_pass_the_9_3_schema(built):
    for name in ("prices", "index", "tickers", "status", "universe", "delisted"):
        validate(name, built[name])


def test_trading_days_exclude_holiday_and_index_is_kospi200_only(built):
    """코스닥150은 쓰는 곳이 없어 index에서 뺐다 (PM 2026-11-04). io.py도 KOSPI200만 허용한다."""
    assert HOLIDAY not in [d.date() for d in built["_days"]] and len(built["_days"]) == 9
    idx = built["index"].set_index("index_code")["close"]
    assert set(built["index"]["index_code"]) == {"KOSPI200"}
    assert idx.loc["KOSPI200"].eq(300.5).all()


def test_split_is_adjusted_and_raw_close_is_kept(built):
    p = built["prices"].set_index(["ticker", "date"])
    assert p.loc[("000003", pd.Timestamp("2025-01-13")), "close_raw"] == 1000
    assert p.loc[("000003", pd.Timestamp("2025-01-13")), "close"] == pytest.approx(500)
    assert p.loc[("000003", pd.Timestamp("2025-01-14")), "close"] == 500
    assert len(built["_events"]) == 1 and built["_events"]["ticker"].iloc[0] == "000003"


def test_market_names_and_ticker_format(built):
    p = built["prices"]
    assert set(p["market"]) == {"KOSPI", "KOSDAQ"} and p["ticker"].str.fullmatch(r"[0-9A-Z]{6}").all()
    assert built["tickers"].set_index("ticker").loc["000003", "market"] == "KOSDAQ"        # KOSDAQ GLOBAL은 KOSDAQ으로


def test_delisted_ticker_last_price_and_date(built):
    d = built["delisted"].set_index("ticker")
    assert list(d.index) == ["000004"]
    assert d.loc["000004", "delist_date"] == pd.Timestamp("2025-01-14")        # 마지막 시세일(1/13)의 다음 거래일
    assert d.loc["000004", "last_price"] == 100
    t = built["tickers"].set_index("ticker")
    assert pd.isna(t.loc["000001", "delisting_date"]) and t.loc["000004", "delisting_date"] == pd.Timestamp("2025-01-14")


def test_security_types_and_unknown(built):
    t = built["tickers"].set_index("ticker")["security_type"]
    assert t["000001"] == "common" and t["000002"] == "preferred" and t["000006"] == "unknown"   # 기본정보에 없던 종목은 unknown


def test_status_is_daily_with_missing_admin_and_halt_from_zero_volume(built):
    s = built["status"]
    assert set(s["date"]) == {pd.Timestamp(d) for d in built["_days"]}        # 신호일만이 아니라 전 거래일
    assert s["is_admin"].isna().all()
    h = s[(s["ticker"] == "000005") & (s["date"] == pd.Timestamp(SIG[1]))]["is_halted"].iloc[0]
    assert bool(h) and not s[s["ticker"] == "000001"]["is_halted"].any()


def test_incomplete_collection_is_rejected(tmp_path):
    _make(tmp_path)
    raw_path(tmp_path, "stk_bydd_trd", date(2025, 1, 15)).unlink()
    with pytest.raises(RuntimeError, match="받지 못한 날 1일"):
        build_all(tmp_path, DAYS[0], DAYS[-1])
    _make(tmp_path)
    raw_path(tmp_path, "kospi_dd_trd", date(2025, 1, 7)).unlink()
    with pytest.raises(Exception, match="받지 못한 날"):
        build_all(tmp_path, DAYS[0], DAYS[-1])


def _break(field, value):
    def patch(raw):
        f = raw_path(raw, "stk_bydd_trd", date(2025, 1, 8))
        j = json.loads(f.read_text(encoding="utf-8"))
        j["OutBlock_1"][0][field] = value
        f.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    return patch


@pytest.mark.parametrize("field,value,msg", [("ISU_CD", "12345", "6자리가 아닌"), ("TDD_CLSPRC", "-", "숫자로 읽히지"),
                                            ("MKT_NM", "KOSDAQ", "시장 구분"), ("BAS_DD", "20250101", "기준일이 파일 날짜와")])
def test_bad_raw_values_stop_the_build(tmp_path, field, value, msg):
    _make(tmp_path, patch=_break(field, value))
    with pytest.raises(RuntimeError, match=msg):
        build_all(tmp_path, DAYS[0], DAYS[-1])


def test_ret_pct_missing_is_allowed(tmp_path):
    _make(tmp_path, patch=_break("FLUC_RT", "-"))                          # 등락률만 결측이면 이벤트로 보지 않고 진행한다
    out = build_all(tmp_path, DAYS[0], DAYS[-1], first_signal_year=2025)
    validate("prices", out["prices"])


def test_require_complete_files_message(tmp_path):
    with pytest.raises(RuntimeError, match="받지 못한 날 1일"):
        require_complete_files(tmp_path, "stk_bydd_trd", [date(2025, 1, 6)])
