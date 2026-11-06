"""sector v1 (DEC-23): KIND(kind.krx.co.kr) 상장법인목록의 업종으로 만든다. KRX Open API가 아니라
사람이 그 페이지의 '다운로드' 버튼으로 받는 것과 같은 단일 정적 요청이다(로그인, 반복 호출 없음).

업종의 시점별 이력은 원천이 없어(Q-04) 최신 분류를 전 기간에 쓴다. 현재 상장 종목만 KIND로 채우고,
상장폐지 종목과 KIND 목록에 없는 종목(우선주 등, 회사당 보통주 기준 한 줄이라 안 나온다)은
unknown_sector_label('미분류')이다. 보통주 업종을 우선주에 유추해 채우지 않는다(R-08: 확인 못한 값을 가정하지 않는다).

사용: python3 -m src.data.sector_kind --snapshot 2026-11-02
"""
import argparse
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

from src.data import io_shim
from src.data.krx_client_paths import ROOT

KIND_URL = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
UNKNOWN_SECTOR_LABEL = "미분류"
TICKER_RE = r"[0-9A-Z]{6}"


def fetch_corp_list(timeout: float = 30.0) -> bytes:
    """KIND 상장법인목록 원문 바이트(EUC-KR, HTML 표)를 그대로 받는다."""
    with urllib.request.urlopen(KIND_URL, timeout=timeout) as resp:
        return resp.read()


def save_raw(raw_bytes: bytes, snapshot: str) -> Path:
    path = ROOT / "data" / "raw" / snapshot / "kind" / "corp_list.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)
    return path


def parse_corp_list(raw_bytes: bytes) -> pd.DataFrame:
    """회사명, 종목코드(6자리), 업종만 남긴다. 형식이 기대와 다르면 멈춘다(R-08)."""
    df = pd.read_html(io.BytesIO(raw_bytes), encoding="euc-kr")[0]
    df = df.rename(columns={"회사명": "name", "종목코드": "ticker", "업종": "sector"})
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    bad = ~df["ticker"].str.fullmatch(TICKER_RE)
    if bad.any():
        raise RuntimeError(f"KIND 종목코드가 6자리 형식이 아닌 행 {int(bad.sum())}건: {df.loc[bad, 'ticker'].tolist()[:5]}")
    blank = df["sector"].isna() | (df["sector"].astype(str).str.strip() == "")
    if blank.any():
        raise RuntimeError(f"KIND 응답에 업종이 빈 행 {int(blank.sum())}건")
    dup = df["ticker"].duplicated()
    if dup.any():
        raise RuntimeError(f"KIND 목록에 종목코드 중복 {int(dup.sum())}건")
    return df[["ticker", "name", "sector"]]


def build_sector(tickers: pd.DataFrame, corp_list: pd.DataFrame) -> pd.DataFrame:
    """v1: 현재 상장 종목(delisting_date 결측)은 KIND 업종, 그 외는 '미분류'.
    valid_from=상장일, valid_to=폐지일(현재 상장이면 결측)."""
    mapped = tickers["ticker"].map(corp_list.set_index("ticker")["sector"])
    current = tickers["delisting_date"].isna()
    sector = mapped.where(current, UNKNOWN_SECTOR_LABEL).fillna(UNKNOWN_SECTOR_LABEL)
    return pd.DataFrame({"ticker": tickers["ticker"].values, "sector": sector.values,
                         "valid_from": tickers["listing_date"].values, "valid_to": tickers["delisting_date"].values})


def unclassified_coverage(universe: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    """신호일별 U_t(in_universe) 중 '미분류' 비중 (DEC-23: PM 보고용)."""
    u = universe.loc[universe["in_universe"], ["date", "ticker"]].merge(
        sector[["ticker", "sector"]], on="ticker", how="left")
    u["sector"] = u["sector"].fillna(UNKNOWN_SECTOR_LABEL)
    g = u.groupby("date")["sector"].agg(n_total="size", n_unclassified=lambda s: int((s == UNKNOWN_SECTOR_LABEL).sum()))
    g["unclassified_ratio"] = g["n_unclassified"] / g["n_total"]
    return g.reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="원문을 저장할 data/raw/{snapshot}/kind/ (이어받기는 같은 값을 쓴다)")
    a = ap.parse_args(argv)

    raw_path = ROOT / "data" / "raw" / a.snapshot / "kind" / "corp_list.html"
    if raw_path.exists():
        raw_bytes = raw_path.read_bytes()
    else:
        raw_bytes = fetch_corp_list()
        save_raw(raw_bytes, a.snapshot)
    corp_list = parse_corp_list(raw_bytes)

    tickers = io_shim.read("tickers")
    universe = io_shim.read("universe", force_stub=True)   # universe가 아직 스텁이면 그걸로 커버리지를 본다
    sector = build_sector(tickers, corp_list)
    coverage = unclassified_coverage(universe, sector)

    path = io_shim.write("sector", sector)
    print(json.dumps({
        "io_available": io_shim.IO_AVAILABLE,
        "rows": len(sector),
        "path": str(path),
        "kind_companies": len(corp_list),
        "classified": int((sector["sector"] != UNKNOWN_SECTOR_LABEL).sum()),
        "unclassified": int((sector["sector"] == UNKNOWN_SECTOR_LABEL).sum()),
        "unclassified_ratio_in_universe": {
            "mean": float(coverage["unclassified_ratio"].mean()),
            "max": float(coverage["unclassified_ratio"].max()),
            "max_date": coverage.loc[coverage["unclassified_ratio"].idxmax(), "date"].strftime("%Y-%m-%d"),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
