"""M1 원천 수집 (A-02). 응답을 data/raw/{수집일}/ 에 그대로 저장한다. 이미 있는 파일은 덮어쓰지 않는다(재개 가능).

사용: python -m src.data.collect_raw --allow-bulk-pykrx --start 2010-01-04 --end 2026-10-30 [--snapshot 2026-10-30] [--workers 4]

2026-10-30 이 경로의 대량 호출로 data.krx.co.kr가 IP를 1일 차단했다(약관 제10조 제2호, 자동화 수집 금지). 수집은 KRX Open API
(collect_openapi)로 하고, 이 모듈은 사용자가 명시적으로 허용했을 때만 돌린다. 연속 실패가 이어지면 요청을 멈춘다.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.data.calendar import signal_dates
from src.data.krx_client import call, stock
from src.data.krx_client_paths import ROOT

INVESTORS = {  # 데이터셋 이름 -> KRX 투자자 구분. 5개 합계가 종목별로 정확히 0이다(2012, 2024 검증).
    "flow_retail": "개인",
    "flow_foreign": "외국인",
    "flow_otherforeign": "기타외국인",
    "flow_inst": "기관합계",
    "flow_othercorp": "기타법인",
}
INDEX_KOSPI200, INDEX_KOSDAQ150 = "1028", "2203"
CHUNK = 40
MAX_CONSECUTIVE_FAILURES = 5


def _ymd(d):
    return pd.Timestamp(d).strftime("%Y%m%d")


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.reset_index().to_parquet(tmp, index=False)
    tmp.replace(path)


def _fetch(dataset: str, d: pd.Timestamp) -> pd.DataFrame:
    s = _ymd(d)
    if dataset == "ohlcv_kospi":
        return call(stock.get_market_ohlcv_by_ticker, s, market="KOSPI")
    if dataset == "ohlcv_kosdaq":
        return call(stock.get_market_ohlcv_by_ticker, s, market="KOSDAQ")
    if dataset in INVESTORS:  # 해당 투자자가 거래하지 않은 종목은 응답에 없다(=0). 통째로 비는 날은 드물지만 허용
        return call(stock.get_market_net_purchases_of_equities, s, s, "ALL", INVESTORS[dataset], expect_rows=False)
    if dataset == "sector_kospi":
        return call(stock.get_market_sector_classifications, s, "KOSPI")
    if dataset == "sector_kosdaq":
        return call(stock.get_market_sector_classifications, s, "KOSDAQ")
    if dataset in ("kospi200", "kosdaq150"):  # 구성종목. 원천에 이력이 없는 구간은 빈 목록(빈 파일로 기록)
        code = INDEX_KOSPI200 if dataset == "kospi200" else INDEX_KOSDAQ150
        tk = call(stock.get_index_portfolio_deposit_file, code, s, expect_rows=False) or []
        return pd.DataFrame({"티커": list(tk)}).set_index("티커")
    raise ValueError(dataset)


DAILY = ["ohlcv_kospi", "ohlcv_kosdaq", *INVESTORS]
WEEKLY = ["sector_kospi", "sector_kosdaq", "kospi200", "kosdaq150"]


def collect_index(out: Path, start, end):
    path = out / "index_1028.parquet"
    if path.exists():
        return pd.read_parquet(path)
    parts = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        a = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01"))
        b = min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31"))
        parts.append(call(stock.get_index_ohlcv_by_date, _ymd(a), _ymd(b), INDEX_KOSPI200))
    df = pd.concat(parts).sort_index()
    df.index.name = "날짜"
    _write(df, path)
    return df.reset_index()


def collect_static(out: Path):
    """날짜 인자가 없는 현재 시점 목록. 수집일 스냅샷으로만 의미가 있다."""
    import FinanceDataReader as fdr
    from pykrx.website.krx.market.core import 전종목기본정보

    static = {
        "listing_now": lambda: 전종목기본정보().fetch(mktId="ALL", segTpCd="ALL"),
        "delisting": lambda: fdr.StockListing("KRX-DELISTING"),
        "admin_now": lambda: fdr.StockListing("KRX-ADMINISTRATIVE"),
    }
    for name, fn in static.items():
        path = out / f"{name}.parquet"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            fn().to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-04")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--snapshot", default=str(date.today()), help="data/raw/{수집일}. 이어받기는 같은 값을 쓴다")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="특정 데이터셋만 (디버그용)")
    ap.add_argument("--allow-bulk-pykrx", action="store_true", help="약관상 자동화 수집이 허용되는지 확인했을 때만 쓴다")
    a = ap.parse_args()
    if not a.allow_bulk_pykrx:
        raise SystemExit("중단: pykrx 대량 수집은 KRX 약관 위반으로 IP 차단을 받았다. KRX Open API(src.data.collect_openapi)를 쓰거나, 허용을 확인했다면 --allow-bulk-pykrx를 붙인다")

    out = ROOT / "data" / "raw" / a.snapshot
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    idx = collect_index(out, a.start, a.end)
    days = list(pd.to_datetime(idx["날짜"]))
    sig = set(signal_dates(days))
    collect_static(out)

    tasks = [(ds, d) for d in days for ds in DAILY] + [(ds, d) for d in sorted(sig) for ds in WEEKLY]
    if a.only:
        tasks = [t for t in tasks if t[0] in a.only]
    tasks = [(ds, d) for ds, d in tasks if not (out / ds / f"{d:%Y-%m-%d}.parquet").exists()]
    print(f"거래일 {len(days)}, 신호일 {len(sig)}, 남은 호출 {len(tasks)}", flush=True)

    failures, done = [], 0

    def run(task):
        ds, d = task
        try:
            _write(_fetch(ds, d), out / ds / f"{d:%Y-%m-%d}.parquet")
            return None
        except Exception as e:
            return {"dataset": ds, "date": f"{d:%Y-%m-%d}", "error": f"{type(e).__name__}: {str(e)[:100]}"}

    streak, aborted = 0, False
    with ThreadPoolExecutor(a.workers) as ex:
        for i in range(0, len(tasks), CHUNK):           # 묶음마다 연속 실패를 보고, 이어지면 더 요청하지 않는다
            for r in ex.map(run, tasks[i:i + CHUNK]):
                done += 1
                streak = streak + 1 if r else 0
                if r:
                    failures.append(r)
            if streak >= MAX_CONSECUTIVE_FAILURES:
                aborted = True
                print(f"연속 {streak}회 실패로 중단 ({done}/{len(tasks)}). 차단·세션 만료일 수 있으니 확인 전에는 다시 돌리지 않는다", flush=True)
                break
            if done % 1000 < CHUNK:
                print(f"{done}/{len(tasks)} 실패 {len(failures)} ({time.time() - t0:.0f}s)", flush=True)

    import pykrx
    meta = {
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "pykrx": pykrx.__version__, "start": a.start, "end": a.end,
        "n_trading_days": len(days), "n_signal_days": len(sig),
        "kosdaq150_index_name": call(stock.get_index_ticker_name, INDEX_KOSDAQ150),
        "failures": failures, "aborted_on_consecutive_failures": aborted,
    }
    (out / f"meta_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: 실패 {len(failures)}건, {time.time() - t0:.0f}초")
    raise SystemExit(1 if failures or aborted else 0)


if __name__ == "__main__":
    main()
