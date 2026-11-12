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
    """공용 index.parquet는 KOSPI200만, 코스닥150은 공용 스키마 밖 index_kosdaq로 따로 둔다 (DEC-27)."""
    assert HOLIDAY not in [d.date() for d in built["_days"]] and len(built["_days"]) == 9
    assert set(built["index"]["index_code"]) == {"KOSPI200"}
    assert built["index"].set_index("index_code")["close"].loc["KOSPI200"].eq(300.5).all()
    assert set(built["index_kosdaq"]["index_code"]) == {"KOSDAQ150"}
    assert built["index_kosdaq"].set_index("index_code")["close"].loc["KOSDAQ150"].eq(800.25).all()


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


def test_delisted_reused_supplement_for_known_tickers():
    """DEC-28: 재사용 종목코드 2건(101970, 036220)의 옛 회사 폐지 보완 데이터.
    마지막 거래일 다음 거래일이 delist_date, 그날 원종가가 last_price다 (delisted.parquet와 같은 정의)."""
    from src.data.build_openapi import REUSED_TICKER_OLD_LAST_DAY, build_delisted_reused
    days = [pd.Timestamp(d) for d in
            ["2015-03-13", "2015-03-16", "2015-03-17",            # 101970 마지막 거래일(03-16) 다음이 03-17
             "2016-05-03", "2016-05-04", "2016-05-09"]]            # 036220 마지막 거래일(05-04) 다음이 05-09
                                                                    # (05-05 어린이날, 05-06 임시공휴일, 05-07~08 주말)
    p = pd.DataFrame([
        {"ticker": "101970", "date": pd.Timestamp("2015-03-16"), "close_raw": 830.0},
        {"ticker": "036220", "date": pd.Timestamp("2016-05-04"), "close_raw": 3500.0},
    ])
    out = build_delisted_reused(p, days)
    assert set(out["ticker"]) == set(REUSED_TICKER_OLD_LAST_DAY)
    row = out.set_index("ticker")
    assert row.loc["101970", "last_price"] == 830.0 and row.loc["101970", "delist_date"] == pd.Timestamp("2015-03-17")
    assert row.loc["036220", "last_price"] == 3500.0 and row.loc["036220", "delist_date"] == pd.Timestamp("2016-05-09")


def test_delisted_reused_raises_if_price_missing_in_covered_range():
    from src.data.build_openapi import build_delisted_reused
    days = [pd.Timestamp("2015-03-16"), pd.Timestamp("2015-03-17")]   # 기준일이 구간 안인데 시세가 없다
    with pytest.raises(RuntimeError, match="101970"):
        build_delisted_reused(pd.DataFrame(columns=["ticker", "date", "close_raw"]), days)


def test_delisted_reused_skips_ticker_outside_build_range():
    """부분 기간 재구축 등 기준일이 이번 빌드 구간 밖이면 조용히 건너뛴다(예외 아님)."""
    from src.data.build_openapi import build_delisted_reused
    days = [pd.Timestamp(d) for d in pd.bdate_range("2020-01-01", "2020-01-10")]
    out = build_delisted_reused(pd.DataFrame(columns=["ticker", "date", "close_raw"]), days)
    assert out.empty


# ---------------------------------------------------------------------------
# run_weekly 1단계 (m1_step)
# ---------------------------------------------------------------------------

class _FakeCtx:
    """D의 StepContext 중 1단계가 쓰는 부분만 흉내 낸다 (test_signals.py의 m3_step 테스트와 같은 요령)."""

    def __init__(self, date, tables=()):
        self.date = date
        self._tables = dict(tables)
        self.written = {}

    def read_table(self, name, date=None):
        return self._tables[name]

    def write_table(self, name, df, date=None):
        self.written[name] = df
        return f"{name}.parquet"


def _canned_out():
    a_tables = ("prices", "index", "tickers", "status", "universe", "delisted", "delisted_reused")
    return {name: pd.DataFrame({"name": [name]}) for name in a_tables}


def test_m1_step_writes_a_tables_empty_pm_tables_and_passes_sector_through(monkeypatch):
    from src.data import build_openapi

    out = _canned_out()
    monkeypatch.setattr(build_openapi, "build_all", lambda raw, start, end: out)
    sector = pd.DataFrame({"name": ["sector"]})
    ctx = _FakeCtx("2026-11-13", {"sector": sector})

    build_openapi.m1_step(ctx)

    for name, df in out.items():
        pd.testing.assert_frame_equal(ctx.written[name], df)
    pd.testing.assert_frame_equal(ctx.written["sector"], sector)   # 다시 받지 않고 그대로 옮김 (DEC-23)
    for name in build_openapi._PM_TABLES:
        assert ctx.written[name].empty
        from src.common import io
        io.validate(name, ctx.written[name])   # 9.3절 스키마를 만족하는 0행
