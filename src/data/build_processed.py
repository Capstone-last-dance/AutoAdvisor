"""M1 정제 (A-02): data/raw/{수집일}/ 스냅샷에서 9.3절 테이블을 만든다. 로그인이 필요 없다.

사용: python -m src.data.build_processed --snapshot 2026-10-30 [--foreign-excludes-other] [--first-signal-year 2011]
KRX에서 확인하지 못한 값(관리종목 이력, 부록 3 Q-06)은 비워 두고, 그 값이 필요한 universe는 스텁으로만 쓴다 (R-08).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import io_shim
from src.data.adjust import adjust_events, adjusted_close
from src.data.calendar import signal_dates
from src.data.krx_client_paths import ROOT
from src.data.security_type import classify_frame
from src.data.universe import avg_trading_value, universe_for_date

FIRST_DAY = pd.Timestamp("2010-01-04")


def _read_many(folder: Path, days, cols=None) -> pd.DataFrame:
    def one(d):
        f = folder / f"{d:%Y-%m-%d}.parquet"
        if not f.exists():
            return None
        x = pd.read_parquet(f, columns=cols)
        x["date"] = d
        return x
    with ThreadPoolExecutor(8) as ex:
        parts = [x for x in ex.map(one, days) if x is not None and len(x)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def require_complete(folder: Path, days) -> None:
    """수집이 끝난 날에는 파일이 있다(응답이 비어도 빈 파일로 남긴다). 파일이 없는 날은 받지 못한 것이므로 조용히 건너뛰지 않는다.
    없는 날을 건너뛰면 수정종가가 가짜 분할 이벤트를 만들고 순매수가 0으로 채워진다."""
    missing = [d for d in days if not (folder / f"{d:%Y-%m-%d}.parquet").exists()]
    if missing:
        raise RuntimeError(f"{folder.name}: 받지 못한 거래일 {len(missing)}일 (예: {missing[0]:%Y-%m-%d}). 수집을 이어받은 뒤 다시 실행한다")


def build_index(raw: Path) -> pd.DataFrame:
    x = pd.read_parquet(raw / "index_1028.parquet")
    return pd.DataFrame({"date": pd.to_datetime(x["날짜"]), "index_code": "KOSPI200", "close": x["종가"].astype("float64")})


def build_tickers(raw: Path, price_tickers: set) -> pd.DataFrame:
    cur = pd.read_parquet(raw / "listing_now.parquet")
    cur = pd.DataFrame({
        "ticker": cur["ISU_SRT_CD"], "name": cur["ISU_ABBRV"],
        "market": cur["MKT_TP_NM"].replace({"KOSDAQ GLOBAL": "KOSDAQ"}),
        "listing_date": pd.to_datetime(cur["LIST_DD"], format="%Y/%m/%d", errors="coerce"),
        "delisting_date": pd.NaT, "secu": cur["SECUGRP_NM"], "kind": cur["KIND_STKCERT_TP_NM"],
    })
    dl = pd.read_parquet(raw / "delisting.parquet")
    dl = dl[dl["Symbol"].astype(str).str.fullmatch(r"[0-9A-Z]{6}")]    # 8자리 코드는 수익증권·신주인수권 등 주식이 아닌 상품(0.4절은 6자리만 허용)
    dl = pd.DataFrame({
        "ticker": dl["Symbol"], "name": dl["Name"], "market": dl["Market"],
        "listing_date": pd.to_datetime(dl["ListingDate"]), "delisting_date": pd.to_datetime(dl["DelistingDate"]),
        "secu": dl["SecuGroup"], "kind": dl["Kind"],
    })
    both = set(cur["ticker"]) & set(dl["ticker"])
    if both:  # tickers.parquet의 키는 ticker 하나라서 같은 코드가 두 번 나오면 표현할 수 없다 → 결정 필요 (ISIN 열 추가 여부)
        raise RuntimeError(f"현재 상장 종목과 폐지 목록에 같은 코드가 있다: {len(both)}건 {sorted(both)[:5]}. [결정 필요] 9.3절 tickers 키")
    dup = dl[dl.duplicated("ticker", keep=False)]
    if len(dup):
        raise RuntimeError(f"폐지 목록에 같은 코드가 여러 번 있다: {dup['ticker'].nunique()}종목. [결정 필요] 9.3절 tickers 키")
    t = classify_frame(pd.concat([cur, dl], ignore_index=True), "secu", "kind", "name")
    unknown = sorted(price_tickers - set(t["ticker"]))
    if unknown:  # 현재 상장도 폐지 목록도 아닌데 시세에 있는 종목 (예: ETF·ETN, 목록 누락). 유형을 모르므로 유니버스에서 빠진다
        t = pd.concat([t, pd.DataFrame({"ticker": unknown, "name": None, "market": None, "security_type": "unknown",
                                        "security_type_source": "none"})], ignore_index=True)
    return t[["ticker", "name", "market", "listing_date", "delisting_date", "security_type", "security_type_source"]]


def build_prices(raw: Path, days: list, tickers: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for mk, ds in (("KOSPI", "ohlcv_kospi"), ("KOSDAQ", "ohlcv_kosdaq")):
        require_complete(raw / ds, days)
        x = _read_many(raw / ds, days)
        x["market"] = mk
        parts.append(x)
    p = pd.concat(parts, ignore_index=True).rename(columns={
        "티커": "ticker", "종가": "close_raw", "거래량": "volume", "거래대금": "value", "등락률": "ret_pct", "시가총액": "mcap"})
    p = p[["date", "ticker", "market", "close_raw", "ret_pct", "volume", "value", "mcap"]]
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    p["close"] = adjusted_close(p).astype("float64")
    interim = ROOT / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    ev = adjust_events(p)
    ev.to_parquet(interim / "adjust_events.parquet", index=False)         # 9.3절 파일이 아닌 A의 중간 산출물 (품질 리포트가 읽는다)
    p.groupby("date").size().rename("n_rows").reset_index().to_parquet(interim / "rows_per_day.parquet", index=False)
    p["close_raw"] = p["close_raw"].astype("float64")
    for c in ("volume", "value", "mcap"):
        p[c] = p[c].astype("int64")
    p["listed_days"] = listed_days(p, tickers, days)
    return p


RELISTING_GAP_TRADING_DAYS = 60   # 종목코드의 거래 공백이 이 거래일수를 넘으면 재상장(다른 회사가 같은 코드를 받음)으로
                                  # 보고 구간을 나눈다. 정상적인 거래정지보다 훨씬 길고, 공백 쪽 구간은 유니버스 60일
                                  # 기준에 어차피 못 미치므로 경계값이 며칠 달라져도 영향이 없다
                                  # (실례: 101970, 036220 — C 리뷰 2026-11-03, docs/appendix3_A_results.md 참고)


def _trading_episodes(p: pd.DataFrame, pos: pd.Series, gap: int = RELISTING_GAP_TRADING_DAYS) -> tuple:
    """종목코드별로 거래일 공백이 gap을 넘으면 구간을 나눈다. 반환: (구간 시작 거래일 위치, 구간 끝 거래일 위치),
    각각 p와 같은 인덱스. 재상장으로 같은 코드를 재사용해도 옛 회사·새 회사 구간이 분리된다."""
    d = p[["ticker", "date"]].sort_values(["ticker", "date"])
    p_pos = d["date"].map(pos)
    prev_pos = p_pos.groupby(d["ticker"]).shift(1)
    new_episode = prev_pos.isna() | ((p_pos - prev_pos) > gap)
    episode_id = new_episode.groupby(d["ticker"]).cumsum()
    key = [d["ticker"], episode_id]
    start = p_pos.groupby(key).transform("min").reindex(p.index)
    end = p_pos.groupby(key).transform("max").reindex(p.index)
    return start, end


def listed_days(p: pd.DataFrame, tickers: pd.DataFrame, days: list) -> pd.Series:
    """실제 상장일부터 센 거래일 수(상장일 당일이 1). 상장일이 days의 첫날보다 앞이면 그 앞 구간은 주중 일수로 센다
    (휴장일을 몰라 약간 크게 나오지만 60일 기준 판정에는 상장일이 수집 시작 직전인 소수 종목만 영향이 있다).

    종목코드는 재사용될 수 있다(폐지 후 몇 년 뒤 다른 회사가 같은 코드로 재상장). tickers의 listing_date는 그 코드의
    '가장 최근' 회사 기준이라서, 공백을 두고 거래 구간을 나눈 뒤(_trading_episodes) 공식 상장일이 설명하는 구간(그
    구간 끝에서 gap 거래일 이내)에서만 상장일-첫 시세일 보정을 적용한다. 그보다 이전 구간(옛 회사)은 그 구간 자체의
    첫 시세일부터 센다 — 새 회사의 상장일을 옛 회사 쪽에 적용하면 음수가 돼 1로 잘리고, 반대로 옛 회사의 첫 시세일을
    새 회사 쪽에 적용하면 신규 상장을 오래된 종목으로 잘못 본다."""
    first = pd.Timestamp(days[0])
    pos = pd.Series(np.arange(len(days)), index=pd.DatetimeIndex(days))
    lst = tickers.set_index("ticker")["listing_date"]
    L = p["ticker"].map(lst)
    official_pos = pd.Series(
        np.searchsorted(np.array(days, dtype="datetime64[ns]"), L.fillna(first).values.astype("datetime64[ns]")), index=p.index)

    ep_start, ep_end = _trading_episodes(p, pos)
    # 공식 상장일이 '이 구간' 안에 있거나(소급 재분류 등) 구간 시작 바로 앞(상장일이 약간 어긋난 정상적인 경우)이면
    # 이 구간이 공식 상장일이 설명하는 구간이다. 구간 끝과 비교하면 안 된다 — 2010년 이전부터 계속 거래된 정상
    # 종목도 상장일(예: 1995년, 수집 시작 전이라 위치 0)과 구간 끝(최근)이 멀어서 잘못 걸린다
    describes_this_episode = (official_pos >= ep_start - RELISTING_GAP_TRADING_DAYS) & (official_pos <= ep_end + RELISTING_GAP_TRADING_DAYS)
    start_pos = np.where(describes_this_episode.values, np.minimum(official_pos.values, ep_start.values), ep_start.values)
    start_pos = pd.Series(start_pos, index=p.index)

    pre = pd.Series(0, index=p.index)
    before = L.notna() & (L < first)
    pre[before] = np.busday_count(L[before].values.astype("datetime64[D]"), np.datetime64(first, "D"))
    out = p["date"].map(pos) - start_pos + 1 + pre
    out[L.isna()] = 10_000                       # 상장일을 모르면 충분히 오래된 것으로 본다 (리포트에서 건수를 센다)
    return out.clip(lower=1).astype("int32")


def build_flows(raw: Path, days: list, prices: pd.DataFrame, foreign_includes_other: bool) -> pd.DataFrame:
    cols = {}
    for ds in ("flow_retail", "flow_foreign", "flow_otherforeign", "flow_inst", "flow_othercorp"):
        require_complete(raw / ds, days)
        x = _read_many(raw / ds, days, cols=["티커", "순매수거래대금"])
        cols[ds] = x.set_index(["date", "티커"])["순매수거래대금"]
    base = pd.MultiIndex.from_frame(prices[["date", "ticker"]], names=["date", "티커"])
    g = pd.DataFrame({k: v.reindex(base).fillna(0) for k, v in cols.items()})   # 응답에 없는 종목 = 그 투자자 거래 없음 (합계 0 항등식으로 검증)
    foreign = g["flow_foreign"] + (g["flow_otherforeign"] if foreign_includes_other else 0)
    out = pd.DataFrame({
        "date": prices["date"].values, "ticker": prices["ticker"].values,
        "retail_net": g["flow_retail"].round().astype("int64").values,
        "foreign_net": foreign.round().astype("int64").values,
        "inst_net": g["flow_inst"].round().astype("int64").values,
    })
    return out


def build_sector(raw: Path, sig: list) -> pd.DataFrame:
    parts = [_read_many(raw / f"sector_{m}", sig, cols=["종목코드", "업종명"]) for m in ("kospi", "kosdaq")]
    s = pd.concat(parts, ignore_index=True).rename(columns={"종목코드": "ticker", "업종명": "sector"})
    s = s.sort_values(["ticker", "date"])
    run = (s["sector"] != s.groupby("ticker")["sector"].shift()).cumsum()
    g = s.groupby(["ticker", run], sort=False).agg(sector=("sector", "first"), valid_from=("date", "min"), last_seen=("date", "max")).reset_index(level=1, drop=True).reset_index()
    nxt = g.groupby("ticker")["valid_from"].shift(-1)
    last_sig = max(sig)
    g["valid_to"] = nxt - pd.DateOffset(days=1)
    is_last = nxt.isna()
    g.loc[is_last & (g["last_seen"] == last_sig), "valid_to"] = pd.NaT      # 현재도 유효
    g.loc[is_last & (g["last_seen"] != last_sig), "valid_to"] = g["last_seen"]
    return g[["ticker", "sector", "valid_from", "valid_to"]].reset_index(drop=True)


def build_status(raw: Path, sig: list, prices: pd.DataFrame) -> pd.DataFrame:
    """신호일별 상태. 관리종목 이력은 확인된 원천이 없어(Q-06) 결측으로 둔다. 거래정지는 '거래량 0'으로 근사한다(4.3절 '체결일에 거래가 없으면'과 같은 기준, Q-06 확인 전 근사)."""
    s = prices[prices["date"].isin(set(sig))][["date", "ticker", "volume"]].copy()
    out = pd.DataFrame({"date": s["date"].values, "ticker": s["ticker"].values,
                        "is_admin": pd.array([pd.NA] * len(s), dtype="boolean"),
                        "is_halted": (s["volume"] == 0).values})
    for col, ds in (("in_kospi200", "kospi200"), ("in_kosdaq150", "kosdaq150")):
        m = {d: set(pd.read_parquet(raw / ds / f"{d:%Y-%m-%d}.parquet")["티커"]) if (raw / ds / f"{d:%Y-%m-%d}.parquet").exists() else None for d in sig}
        avail = out["date"].map(lambda d: bool(m.get(d)))
        member = [t in m[d] if m.get(d) else False for d, t in zip(out["date"], out["ticker"])]
        out[col] = pd.array([x if a else pd.NA for x, a in zip(member, avail)], dtype="boolean")
    return out


def build_delisted(tickers: pd.DataFrame, prices: pd.DataFrame, raw: Path) -> pd.DataFrame:
    dl = pd.read_parquet(raw / "delisting.parquet")
    last_trade = dl.set_index("Symbol")["ArrantEndDate"].fillna(dl.set_index("Symbol")["DelistingDate"])
    t = tickers[tickers["delisting_date"].notna() & (tickers["delisting_date"] >= FIRST_DAY)]
    last = prices.sort_values("date").groupby("ticker").tail(1).set_index("ticker")[["date", "close_raw"]]
    out = []
    for r in t.itertuples():
        lp = np.nan
        if r.ticker in last.index:
            d, c = last.loc[r.ticker, "date"], last.loc[r.ticker, "close_raw"]
            ref = pd.Timestamp(last_trade.get(r.ticker, r.delisting_date))
            if abs((ref - d).days) <= 7:          # 마지막 거래 기록이 정리매매 종료일 근처일 때만 마지막 가격으로 쓴다
                lp = float(c)
        out.append((r.ticker, r.delisting_date, lp))
    return pd.DataFrame(out, columns=["ticker", "delist_date", "last_price"]).astype({"last_price": "float64"})


def build_universe(prices: pd.DataFrame, tickers: pd.DataFrame, status: pd.DataFrame, sig: list, first_year: int, provisional: bool):
    vp = prices.pivot(index="date", columns="ticker", values="value")
    avg = avg_trading_value(vp)
    info = tickers.set_index("ticker")[["name", "security_type"]]
    use = {t for t in sig if t.year >= first_year}
    # 신호일 행과 상태를 한 번에 합친다 (종목마다 조회하면 신호일 수 × 종목 수만큼 느려진다)
    ps = prices[prices["date"].isin(use)][["date", "ticker", "mcap", "listed_days"]]
    ps = ps.merge(status[["date", "ticker", "is_halted", "is_admin"]], on=["date", "ticker"], how="left")
    ps["is_halted"] = ps["is_halted"].fillna(False).astype(bool)
    if ps["is_admin"].isna().any() and not provisional:
        raise RuntimeError("is_admin 결측: 관리종목 이력 원천이 확정되지 않았다(Q-06). 4.4절 3단계를 실행할 수 없다 (R-08)")
    ps["is_admin"] = ps["is_admin"].fillna(False).astype(bool)
    ps = ps.join(info, on="ticker")
    out = []
    for t, day in ps.groupby("date", sort=True):
        u = universe_for_date(day, avg.loc[t])
        u.insert(0, "date", t)
        out.append(u)
    u = pd.concat(out, ignore_index=True)
    u["date"] = pd.to_datetime(u["date"])
    return u.astype({"mcap_rank": "int32", "in_universe": "bool"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--first-signal-year", type=int, default=2011)
    ap.add_argument("--foreign-excludes-other", action="store_true", help="foreign_net을 '외국인'만으로 (기본은 기타외국인 포함 = 외국인 합계)")
    ap.add_argument("--provisional", action="store_true", help="is_admin이 비어 있어도 universe를 스텁으로 만든다 (실제 판정에 쓰지 않는다)")
    a = ap.parse_args()
    raw = ROOT / "data" / "raw" / a.snapshot

    index = build_index(raw)
    days = list(index["date"])
    sig = signal_dates(days)
    # 시세 종목 집합은 prices를 만든 뒤 알 수 있으므로, tickers는 두 번 만든다 (1차: 상장일 매핑용, 2차: unknown 반영)
    t0 = build_tickers(raw, set())
    prices = build_prices(raw, days, t0)
    tickers = build_tickers(raw, set(prices["ticker"]))   # prices.listed_days는 t0로 이미 계산됨(시세에만 있는 종목은 t0에서도 상장일 결측이라 결과가 같다)

    flows = build_flows(raw, days, prices, foreign_includes_other=not a.foreign_excludes_other)
    sector = build_sector(raw, sig)
    status = build_status(raw, sig, prices)
    delisted = build_delisted(tickers, prices, raw)
    universe = build_universe(prices, tickers, status, sig, a.first_signal_year, a.provisional)

    admin_missing = bool(status["is_admin"].isna().any())
    final = {"prices": prices[["date", "ticker", "market", "close", "close_raw", "volume", "value", "mcap", "listed_days"]],
             "index": index, "tickers": tickers.drop(columns="security_type_source"), "status": status,
             "delisted": delisted, "sector": sector, "investor_flows": flows}
    paths = {}
    for name, df in final.items():
        paths[name] = str(io_shim.write(name, df))
    paths["universe"] = str(io_shim.write("universe", universe, force_stub=admin_missing))   # is_admin 결측이면 스텁
    tickers.to_parquet(ROOT / "_stub" / "tickers_with_source.parquet", index=False) if (ROOT / "_stub").exists() else None
    print(json.dumps({"io_available": io_shim.IO_AVAILABLE, "universe_stub": admin_missing,
                      "rows": {k: len(v) for k, v in final.items()} | {"universe": len(universe)}, "paths": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
