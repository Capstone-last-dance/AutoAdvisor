"""A-03 데이터 품질 리포트 (6.1절 완료 기준(A) (1)~(5)). reports/{run_id}/ 에 metrics.json, figures/, report.md를 쓴다.

사용: python -m src.data.quality_report --snapshot 2026-10-30 [--processed _stub/data/processed] [--no-krx]
--no-krx를 빼면 pykrx로 (1) 거래일 교차검증과 계산한 수정종가 대조를 추가로 한다 (로그인 필요).
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.krx_client_paths import ROOT  # noqa: E402

import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["AppleGothic", "Noto Sans CJK KR", "NanumGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
SPLIT_CASES = [("005930", "삼성전자", "2018-05-04", "50:1"), ("035420", "NAVER", "2018-10-12", "5:1"), ("035720", "카카오", "2021-04-15", "5:1")]


def check_calendar(index: pd.DataFrame, use_krx: bool) -> dict:
    """완료 기준 (1): 코스피 200 종가 기준의 거래일 캘린더. index.parquet엔 코스닥 150도 섞여 있어 그것만 본다."""
    kospi200 = index[index["index_code"] == "KOSPI200"]
    days = pd.DatetimeIndex(kospi200["date"])
    r = {"n_days": int(len(days)), "first": str(days.min().date()), "last": str(days.max().date()),
         "duplicates": int(days.duplicated().sum()), "close_nonpositive_or_nan": int((~(kospi200["close"] > 0)).sum())}
    if use_krx:
        from src.data.krx_client import call, stock
        bd = set()
        for y in range(days.min().year, days.max().year + 1):
            for m in range(1, 13):
                if (y, m) > (days.max().year, days.max().month):
                    break
                bd |= {pd.Timestamp(x).normalize() for x in call(stock.get_business_days, y, m, expect_rows=False) or []}
        bd = {d for d in bd if days.min() <= d <= days.max()}
        r["pykrx_business_days"] = len(bd)
        r["index_only"] = sorted(str(d.date()) for d in set(days) - bd)[:20]
        r["pykrx_only"] = sorted(str(d.date()) for d in bd - set(days))[:20]
    return r


def check_delisted(tickers, prices, delisted) -> dict:
    t = tickers[tickers["delisting_date"].notna() & (tickers["delisting_date"] >= "2010-01-04")]
    in_prices = set(prices["ticker"].unique())
    out = {"delisted_since_2010_total": int(len(t)), "in_prices": int(t["ticker"].isin(in_prices).sum()),
           "by_type": t.groupby("security_type").agg(n=("ticker", "size")).assign(in_prices=t.assign(p=t["ticker"].isin(in_prices)).groupby("security_type")["p"].sum()).astype(int).to_dict("index")}
    c = t[t["security_type"] == "common"]
    out["common_delisted"] = int(len(c))
    out["common_delisted_in_prices"] = int(c["ticker"].isin(in_prices).sum())
    d = delisted.set_index("ticker").reindex(c["ticker"])
    out["common_delisted_with_last_price"] = int(d["last_price"].notna().sum())
    return out


def check_splits(prices) -> tuple:
    rows, res = [], []
    for tk, name, day, ratio in SPLIT_CASES:
        p = prices[prices["ticker"] == tk].sort_values("date").reset_index(drop=True)
        i = p.index[p["date"] == pd.Timestamp(day)]
        if len(i) == 0:
            res.append({"ticker": tk, "name": name, "date": day, "ratio": ratio, "status": "시세 없음"})
            continue
        i = int(i[0])
        w = p.iloc[max(0, i - 2): i + 3][["date", "close_raw", "close"]].copy()
        w.insert(0, "ticker", tk)
        rows.append(w)
        step = (p["close"].pct_change().iloc[max(1, i - 2): i + 3]).abs().max()
        raw_step = (p["close_raw"].pct_change().iloc[max(1, i - 2): i + 3]).abs().max()
        res.append({"ticker": tk, "name": name, "date": day, "ratio": ratio, "max_abs_daily_change_adjusted": float(step),
                    "max_abs_daily_change_raw": float(raw_step), "status": "연속" if step < 0.31 else "끊김"})
    return res, (pd.concat(rows) if rows else pd.DataFrame())


def check_flow_identity(raw: Path, days) -> dict:
    """5개 투자자 구분의 종목별 순매수대금 합계가 0인지 전 거래일에 대해 검사한다.
    investor_flows를 만들지 않기로 한 뒤(DEC-22)에는 원천 자체가 없으므로 해당 없음으로 표시한다."""
    names = ["flow_retail", "flow_foreign", "flow_otherforeign", "flow_inst", "flow_othercorp"]
    if not any((raw / n).exists() for n in names):
        return {"applicable": False, "reason": "investor_flows를 만들지 않음 (DEC-22, 원천 없음)"}
    bad_days, n_checked, max_abs = [], 0, 0.0
    for d in days:
        parts = []
        for ds in names:
            f = raw / ds / f"{d:%Y-%m-%d}.parquet"
            if f.exists():
                x = pd.read_parquet(f, columns=["티커", "순매수거래대금"]).set_index("티커")["순매수거래대금"]
                parts.append(x.rename(ds))
        if not parts:
            bad_days.append(str(d.date())); continue
        s = pd.concat(parts, axis=1).fillna(0).sum(axis=1).abs()
        n_checked += 1
        max_abs = max(max_abs, float(s.max()))
        if (s > 1).any():
            bad_days.append(str(d.date()))
    return {"days_checked": n_checked, "days_with_nonzero_sum": len(bad_days), "examples": bad_days[:10], "max_abs_sum_krw": max_abs}


def check_adjusted_vs_pykrx(prices, n=30, seed=42) -> dict:
    from src.data.krx_client import call, stock
    rng = np.random.default_rng(seed)
    cand = prices[prices["date"] >= "2016-01-01"].groupby("ticker").size()
    cand = cand[cand > 1000].index.to_numpy()
    pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
    errs, per = [], []
    for tk in pick:
        p = prices[(prices["ticker"] == tk) & (prices["date"] >= "2016-01-01")].set_index("date")["close"]
        rels = []
        for y in range(2016, p.index.max().year + 1):
            r = call(stock.get_market_ohlcv_by_date, f"{y}0101", f"{y}1231", tk, expect_rows=False)
            if r is None or not len(r):
                continue
            j = pd.concat([p.rename("mine"), r["종가"].rename("pykrx")], axis=1, join="inner")
            if not len(j):
                continue
            # pykrx 수정주가는 조회 구간 끝 기준으로 조정되어 해마다 단위가 다르다. 연도 조각 안에서만 비율을 비교한다
            # (조각 안의 이벤트는 양쪽에 모두 반영되고, 조각 밖의 이벤트는 비율이 상수가 되어 상쇄된다)
            rels.append((j["mine"] / j["pykrx"]).pipe(lambda x: x / x.iloc[-1] - 1).abs())
        if not rels:
            continue
        rel = pd.concat(rels)
        errs.append(rel.max()); per.append({"ticker": tk, "max_rel_err": float(rel.max()), "n": int(len(rel))})
    e = np.array(errs)
    return {"n_tickers": int(len(e)), "max_rel_err": float(e.max()) if len(e) else None,
            "p95_rel_err": float(np.quantile(e, 0.95)) if len(e) else None, "n_above_1e-3": int((e > 1e-3).sum()), "worst": sorted(per, key=lambda r: -r["max_rel_err"])[:5]}


def dividend_check(prices, events: pd.DataFrame) -> dict:
    """Q-05: 조정 이벤트가 연말 배당락 시기(12월 24~30일)에 몰리는지 본다. 현금배당이 기준가에 반영되면 이 시기 이벤트 비율이 눈에 띄게 높다.
    비교 기준은 같은 달력 구간이 아닌 전체 종목-일 대비 비율이다."""
    md = prices["date"].dt.strftime("%m-%d")
    in_win = md.between("12-24", "12-30")
    ev_md = events["date"].dt.strftime("%m-%d")
    ev_win = ev_md.between("12-24", "12-30")
    rate_all = len(events) / len(prices)
    rate_win = ev_win.sum() / max(int(in_win.sum()), 1)
    return {"events_total": int(len(events)), "rows_total": int(len(prices)), "event_rate_all_days": float(rate_all),
            "events_dec24_30": int(ev_win.sum()), "rows_dec24_30": int(in_win.sum()), "event_rate_dec24_30": float(rate_win),
            "ratio_year_end_to_all": float(rate_win / rate_all) if rate_all else None,
            "events_ratio_below_1_pct_of_all": float(((events["ratio"] < 1) & (events["ratio"] > 0.9)).mean()) if len(events) else None,
            "reading": "비율이 1에 가까우면 배당락 조정이 없고, 수 배 이상이면 현금배당이 반영된 것"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--processed", default=str(ROOT / "_stub" / "data" / "processed"))
    ap.add_argument("--no-krx", action="store_true")
    ap.add_argument("--tag", default="dq_v1")
    a = ap.parse_args()
    raw, proc = ROOT / "data" / "raw" / a.snapshot, Path(a.processed)
    run_id = f"{date.today():%Y%m%d}_A_{a.tag}"
    out = ROOT / "reports" / run_id
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rd = lambda n, **k: pd.read_parquet(proc / f"{n}.parquet", **k)
    index, tickers, prices = rd("index"), rd("tickers"), rd("prices")
    delisted, universe, status = rd("delisted"), rd("universe"), rd("status")
    # ret_pct는 processed에 없으므로 원본에서 다시 읽지 않고 등락률 기반 점검은 수정종가 이벤트 수로 대신한다
    days = list(pd.DatetimeIndex(index.loc[index["index_code"] == "KOSPI200", "date"]))

    m = {"run_id": run_id, "snapshot": a.snapshot, "processed_dir": str(proc)}
    m["c1_calendar"] = check_calendar(index, use_krx=not a.no_krx)
    m["c2_delisted"] = check_delisted(tickers, prices, delisted)
    m["c3_ticker_format"] = {t: bool(df["ticker"].dropna().str.fullmatch(r"[0-9A-Z]{6}").all()) for t, df in
                             [("prices", prices), ("tickers", tickers), ("status", status), ("universe", universe), ("delisted", delisted)]}
    splits, split_rows = check_splits(prices)
    m["c4_splits"] = splits
    split_rows.to_csv(out / "split_cases.csv", index=False, encoding="utf-8-sig")

    nt = universe[universe["in_universe"]].groupby("date").size()
    m["c5_nt"] = {"n_signal_days": int(len(nt)), "min": int(nt.min()) if len(nt) else None, "median": float(nt.median()) if len(nt) else None,
                  "max": int(nt.max()) if len(nt) else None,
                  "universe_is_stub": not (ROOT / "data" / "processed" / "universe.parquet").exists()}
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(nt.index, nt.values, lw=1)
    ax.set(title="신호일별 유니버스 종목 수 N_t" + (" (스텁: 관리종목 이력 미반영)" if m["c5_nt"]["universe_is_stub"] else ""), ylabel="N_t")
    ax.axhline(500, color="gray", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(out / "figures" / "n_t.png", dpi=150); plt.close(fig)

    m["extra_flow_identity"] = check_flow_identity(raw, days)
    m["extra_security_types"] = tickers["security_type"].value_counts().to_dict()
    m["extra_unknown_tickers_in_prices"] = int((tickers["security_type"] == "unknown").sum())
    first = prices.groupby("ticker")["date"].min()
    lst = tickers.set_index("ticker")["listing_date"].reindex(first.index)
    gap = (first - lst).dt.days
    new = first > days[0] + pd.Timedelta(days=7)    # 수집 시작 이후 처음 나타난 종목만 비교 가능
    m["extra_listing_vs_first_price"] = {"compared": int(new.sum()), "first_price_more_than_7d_after_listing": int((gap[new] > 7).sum()),
                                         "first_price_before_listing": int((gap[new] < 0).sum()), "listing_date_missing": int(lst.isna().sum())}
    if not a.no_krx:
        m["extra_adjusted_vs_pykrx"] = check_adjusted_vs_pykrx(prices)
    ev_path = ROOT / "data" / "interim" / "adjust_events.parquet"
    m["extra_cash_dividend_q05"] = dividend_check(prices, pd.read_parquet(ev_path)) if ev_path.exists() else "adjust_events.parquet 없음"
    (out / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "git_commit.txt").write_text("not a git repository (저장소 연결 전)\n", encoding="utf-8")
    (out / "config.yaml").write_text(f"snapshot: {a.snapshot}\nprocessed: {proc}\nno_krx: {a.no_krx}\ncommand: {' '.join(sys.argv)}\n", encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
