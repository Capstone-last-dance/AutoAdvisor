"""M1 정제, KRX Open API 원문 경로 (A-02): data/raw/{수집일}/openapi/ 의 JSON에서 9.3절 테이블을 만든다. 네트워크와 로그인이 필요 없다.

사용: python3 -m src.data.build_openapi --snapshot 2026-11-02 --start 2025-09-01 --end 2026-10-30 [--first-signal-year 2011]

만드는 것: prices, index(코스피200), index_kosdaq(코스닥150, 공용 스키마 밖), tickers, status(일별, is_halted 포함), universe(스텁), delisted.
만들지 못하는 것 (Open API에 원천이 없다, 보고서의 [결정 필요]): investor_flows(투자자별 순매수), sector(업종; 코스피 종목의 SECT_TP_NM이 비어 있다),
status.is_admin(관리종목, Q-06). 이 값은 결측으로 두고 그 값이 필요한 universe는 스텁으로만 쓴다 (R-08).
in_kospi200·in_kosdaq150(구성종목)은 쓰는 곳이 없어 status 스키마에서 뺐다 (PM 2026-11-03).
공용 index.parquet에는 KOSPI200만 넣는다. 코스닥150은 공용 스키마 밖의 index_kosdaq.parquet로 따로 둔다.
소비처가 생기면 PM이 기록한 뒤 D가 공용 스키마에 추가한다 (DEC-27).

가정 (명세에 정의가 없어 여기서 정했다):
- 거래일 = 지수 서비스 응답에서 `코스피 200` 종가가 있는 날 (Q-14).
- 시세는 비수정이다. 수정종가는 등락률로 역산한 기준가로 만든다 (adjust.py).
- 상장폐지 종목은 마지막 시세일 이후 시세 서비스에서 사라진 종목이다. delisting_date = 마지막 시세일의 다음 거래일, last_price = 마지막 시세일의
  비수정 종가. 종목코드가 바뀐 경우(합병·코드 변경)도 폐지로 보이므로 품질 리포트에서 건수를 확인한다.
- 종목 유형(보통주 등)은 신호일 종목기본정보 중 가장 최근 것으로 판별한다. 한 번도 기본정보에 나오지 않은 종목은 'unknown'이다(유니버스 제외).
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.data import io_shim
from src.data.adjust import adjust_events, adjusted_close
from src.data.build_processed import build_universe, listed_days
from src.data.calendar import signal_dates
from src.data.collect_openapi import calendar_days, raw_path, weekdays
from src.data.krx_client_paths import ROOT
from src.data.security_type import classify_frame

PRICE_SERVICES = (("stk_bydd_trd", "KOSPI"), ("ksq_bydd_trd", "KOSDAQ"))
BASE_SERVICES = ("stk_isu_base_info", "ksq_isu_base_info")
CALENDAR_SERVICE, INDEX_NAME = "kospi_dd_trd", "코스피 200"
INDEX_SERIES = ((CALENDAR_SERVICE, INDEX_NAME, "KOSPI200"), ("kosdaq_dd_trd", "코스닥 150", "KOSDAQ150"))
TICKER_RE = r"[0-9A-Z]{6}"
MARKET_MAP = {"KOSDAQ GLOBAL": "KOSDAQ"}
NUMERIC = {"TDD_CLSPRC": "close_raw", "FLUC_RT": "ret_pct", "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value", "MKTCAP": "mcap"}


def _rows(raw: Path, service: str, d) -> list:
    return json.loads(raw_path(raw, service, d).read_text(encoding="utf-8"))["OutBlock_1"]


def require_complete_files(raw: Path, service: str, days) -> None:
    """수집이 끝난 날에는 파일이 있다(응답이 비어도 빈 목록으로 저장). 없는 날은 받지 못한 것이므로 조용히 건너뛰지 않는다."""
    missing = [d for d in days if not raw_path(raw, service, d).exists()]
    if missing:
        raise RuntimeError(f"{service}: 받지 못한 날 {len(missing)}일 (예: {pd.Timestamp(missing[0]):%Y-%m-%d}). 수집을 이어받은 뒤 다시 실행한다")


def read_prices(raw: Path, days) -> pd.DataFrame:
    """거래일별 시세를 한 표로. 숫자 변환에 실패한 값(결측)은 ret_pct만 허용하고 나머지는 오류로 멈춘다."""
    parts = []
    for service, market in PRICE_SERVICES:
        for d in days:
            rows = _rows(raw, service, d)
            if not rows:
                continue
            x = pd.DataFrame(rows, columns=["BAS_DD", "ISU_CD", "MKT_NM", *NUMERIC])
            if (x["BAS_DD"] != f"{d:%Y%m%d}").any():
                raise RuntimeError(f"{service} {d}: 응답 기준일이 파일 날짜와 다르다")
            for c in NUMERIC:
                x[c] = pd.to_numeric(x[c], errors="coerce")
            x["date"] = pd.Timestamp(d)
            x["market"] = x["MKT_NM"].replace(MARKET_MAP)
            if not x["market"].eq(market).all():
                raise RuntimeError(f"{service} {d}: 시장 구분이 {market}이 아닌 행이 있다: {sorted(set(x['market']))}")
            parts.append(x.rename(columns={"ISU_CD": "ticker", **NUMERIC})[["date", "ticker", "market", "close_raw", "ret_pct", "volume", "value", "mcap"]])
    p = pd.concat(parts, ignore_index=True)
    bad_code = ~p["ticker"].str.fullmatch(TICKER_RE)
    if bad_code.any():
        raise RuntimeError(f"6자리가 아닌 종목코드 {int(bad_code.sum())}건: {p.loc[bad_code, 'ticker'].unique()[:5].tolist()}")
    nan = {c: int(p[c].isna().sum()) for c in ("close_raw", "volume", "value", "mcap") if p[c].isna().any()}
    if nan:
        raise RuntimeError(f"숫자로 읽히지 않은 값이 있다 {nan}. 원문을 확인하고 처리 규칙을 정한다 (R-08)")
    dup = p.duplicated(["date", "ticker"])
    if dup.any():
        raise RuntimeError(f"(date, ticker) 중복 {int(dup.sum())}건")
    return p.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_prices(p: pd.DataFrame, tickers_for_listing: pd.DataFrame, days: list) -> pd.DataFrame:
    p = p.copy()
    p["close"] = adjusted_close(p).astype("float64")
    p["close_raw"] = p["close_raw"].astype("float64")
    for c in ("volume", "value", "mcap"):
        p[c] = p[c].astype("int64")
    p["listed_days"] = listed_days(p, tickers_for_listing, days)
    return p[["date", "ticker", "market", "close", "close_raw", "volume", "value", "mcap", "listed_days"]]


def build_index(raw: Path, days) -> pd.DataFrame:
    out = []
    for service, idx_name, code in INDEX_SERIES:
        for d in days:
            r = [x for x in _rows(raw, service, d) if x["IDX_NM"] == idx_name]
            if not r:
                raise RuntimeError(f"{service} {d}: {idx_name} 종가가 없다")
            out.append((pd.Timestamp(d), code, float(r[0]["CLSPRC_IDX"])))
    return pd.DataFrame(out, columns=["date", "index_code", "close"])


def read_base_latest(raw: Path, sig_days) -> pd.DataFrame:
    """신호일 종목기본정보 중 종목별 가장 최근 행."""
    latest = {}
    for d in sig_days:
        for service in BASE_SERVICES:
            for r in _rows(raw, service, d):
                latest[r["ISU_SRT_CD"]] = (r["ISU_ABBRV"], r["MKT_TP_NM"], r["LIST_DD"], r["SECUGRP_NM"], r["KIND_STKCERT_TP_NM"])
    b = pd.DataFrame.from_dict(latest, orient="index", columns=["name", "market", "list_dd", "secu", "kind"]).rename_axis("ticker").reset_index()
    b["market"] = b["market"].replace(MARKET_MAP)
    b["listing_date"] = pd.to_datetime(b["list_dd"], format="%Y%m%d", errors="coerce")
    return b.drop(columns="list_dd")


def build_tickers(base: pd.DataFrame, p: pd.DataFrame, days: list) -> pd.DataFrame:
    seen = p.groupby("ticker")["date"].agg(first_seen="min", last_seen="max")
    t = base[base["ticker"].isin(seen.index)].copy()
    t = classify_frame(t, "secu", "kind", "name")
    unknown = sorted(set(seen.index) - set(t["ticker"]))     # 시세에는 있는데 신호일 기본정보에는 한 번도 안 나온 종목
    if unknown:
        t = pd.concat([t, pd.DataFrame({"ticker": unknown, "name": None, "market": None, "security_type": "unknown",
                                        "security_type_source": "none"})], ignore_index=True)
    last_day = pd.Timestamp(days[-1])
    nxt = pd.Series(pd.DatetimeIndex(days[1:] + [pd.NaT]), index=pd.DatetimeIndex(days))   # 마지막 시세일의 다음 거래일
    ls = t["ticker"].map(seen["last_seen"])
    t["delisting_date"] = pd.to_datetime(ls.map(nxt).where(ls < last_day))
    return t[["ticker", "name", "market", "listing_date", "delisting_date", "security_type", "security_type_source"]].reset_index(drop=True)


def build_status(p: pd.DataFrame) -> pd.DataFrame:
    """일별 상태(모든 거래일, 전종목). 관리종목(Q-06)은 원천이 없어 결측이다.
    거래정지는 '거래량 0'으로 근사한다(Q-06 확인 전 근사). 정지일의 close는 원천 응답이 그대로 직전 종가를
    반복하므로(2026-11-03 확인, 예: 000030 2019-01-09) 별도 이월 처리가 필요 없다."""
    s = p[["date", "ticker", "volume"]]
    na = pd.array([pd.NA] * len(s), dtype="boolean")
    return pd.DataFrame({"date": s["date"].values, "ticker": s["ticker"].values, "is_admin": na,
                         "is_halted": (s["volume"] == 0).values})


def build_delisted(tickers: pd.DataFrame, p: pd.DataFrame) -> pd.DataFrame:
    t = tickers[tickers["delisting_date"].notna()]
    last = p.sort_values("date").groupby("ticker").tail(1).set_index("ticker")["close_raw"]
    return pd.DataFrame({"ticker": t["ticker"].values, "delist_date": t["delisting_date"].values,
                         "last_price": t["ticker"].map(last).astype("float64").values})


# DEC-28: 재사용 종목코드 2건의 "옛 회사" 상장폐지 보완 데이터. 마지막 거래일·원종가는 수집한 prices 원본에서
# 직접 확인했고(tickers/delisted는 '가장 최근' 회사 기준이라 옛 회사 쪽은 담지 못한다), 날짜·사유는 뉴스로 교차 확인했다.
# 101970(우양에이치씨): 2015-03-17 최종부도 상장폐지, 정리매매 2015-03-06~16
#   https://edaily.co.kr/News/Read?mediaCodeNo=257&newsId=03827766609300040 (2015-03-04)
# 036220(인포피아): 2016-05-09 상장폐지(마지막 거래일 05-04 다음은 05-05 어린이날·05-06 임시공휴일·05-07~08
#   주말이 모두 끼어 05-09가 다음 거래일이다 — 한때 05-06으로 잘못 기록했었음, 2026-11-05 정정), 정리매매 2016-04-26~05-04
#   https://edaily.co.kr/News/Read?mediaCodeNo=257&newsId=03181606612618744
REUSED_TICKER_OLD_LAST_DAY = {"101970": "2015-03-16", "036220": "2016-05-04"}


def build_delisted_reused(p: pd.DataFrame, days: list) -> pd.DataFrame:
    """delisted.parquet(종목당 한 행) 스키마는 그대로 두고(상장 구간 형식 변경은 v2, D와 결정), 공용 스키마
    밖 별도 파일로 둔다. delist_date는 옛 회사 마지막 거래일의 다음 거래일(delisted.parquet와 같은 정의),
    last_price는 그날 원종가다. 기준일이 이번 빌드 구간(days) 밖이면(부분 기간 재구축 등) 그 종목은 건너뛴다 —
    구간 안인데 시세가 없으면 그게 진짜 이상이므로 예외를 낸다."""
    pos = {d: i for i, d in enumerate(days)}
    out = []
    for ticker, last_day in REUSED_TICKER_OLD_LAST_DAY.items():
        last_day = pd.Timestamp(last_day)
        if last_day not in pos:
            continue
        row = p[(p["ticker"] == ticker) & (p["date"] == last_day)]
        if row.empty:
            raise RuntimeError(f"{ticker}: 보완 데이터 기준일 {last_day:%Y-%m-%d}이 거래일인데 시세가 없다")
        out.append((ticker, days[pos[last_day] + 1], float(row["close_raw"].iloc[0])))
    return pd.DataFrame(out, columns=["ticker", "delist_date", "last_price"])


def build_all(raw: Path, start: date, end: date, first_signal_year: int = 2011) -> dict:
    weekdays_all = weekdays(start, end)
    require_complete_files(raw, CALENDAR_SERVICE, weekdays_all)
    days = [pd.Timestamp(d) for d in calendar_days(raw, CALENDAR_SERVICE, weekdays_all, INDEX_NAME)]
    sig = signal_dates(days)
    for service, _ in PRICE_SERVICES:
        require_complete_files(raw, service, days)
    for service in BASE_SERVICES:
        require_complete_files(raw, service, sig)
    require_complete_files(raw, "kosdaq_dd_trd", days)

    p = read_prices(raw, days)
    base = read_base_latest(raw, sig)
    tickers = build_tickers(base, p, days)
    prices = build_prices(p, tickers, days)
    status = build_status(prices)
    universe = build_universe(prices, tickers, status, sig, first_signal_year, provisional=True)
    idx = build_index(raw, days)
    return {"prices": prices, "index": idx[idx["index_code"] == "KOSPI200"].reset_index(drop=True),
            "index_kosdaq": idx[idx["index_code"] == "KOSDAQ150"].reset_index(drop=True),   # 공용 스키마 밖 (DEC-27)
            "tickers": tickers.drop(columns="security_type_source"), "status": status,
            "universe": universe, "delisted": build_delisted(tickers, p),
            "delisted_reused": build_delisted_reused(p, days),   # 공용 스키마 밖 (DEC-28)
            "_events": adjust_events(p), "_days": days}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--start", required=True, help="수집한 구간의 첫 평일 (누락된 날이 있으면 오류로 멈춘다)")
    ap.add_argument("--end", required=True)
    ap.add_argument("--first-signal-year", type=int, default=2011)
    a = ap.parse_args(argv)
    raw = ROOT / "data" / "raw" / a.snapshot
    out = build_all(raw, date.fromisoformat(a.start), date.fromisoformat(a.end), a.first_signal_year)

    interim = ROOT / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    out["_events"].to_parquet(interim / "adjust_events.parquet", index=False)       # 품질 리포트가 읽는 중간 산출물
    out["prices"].groupby("date").size().rename("n_rows").reset_index().to_parquet(interim / "rows_per_day.parquet", index=False)
    paths = {}
    for name in ("prices", "index", "tickers", "status", "delisted"):
        paths[name] = str(io_shim.write(name, out[name]))
    paths["universe"] = str(io_shim.write("universe", out["universe"], force_stub=True))   # is_admin 결측(Q-06)이라 스텁
    kosdaq_path = Path(paths["index"]).parent / "index_kosdaq.parquet"   # 공용 스키마 밖 (DEC-27), 검증 없이 직접 쓴다
    out["index_kosdaq"].to_parquet(kosdaq_path, index=False)
    paths["index_kosdaq"] = str(kosdaq_path)
    reused_path = Path(paths["delisted"]).parent / "delisted_reused.parquet"   # 공용 스키마 밖 (DEC-28)
    out["delisted_reused"].to_parquet(reused_path, index=False)
    paths["delisted_reused"] = str(reused_path)
    print(json.dumps({"io_available": io_shim.IO_AVAILABLE, "rows": {k: len(out[k]) for k in paths}, "paths": paths,
                      "trading_days": len(out["_days"]), "not_built": ["investor_flows", "sector"],
                      "unknown_security_type": int((out["tickers"]["security_type"] == "unknown").sum()),
                      "adjust_events": len(out["_events"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
