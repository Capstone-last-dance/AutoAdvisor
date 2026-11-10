"""A-03 데이터 품질 리포트 (6.1절 완료 기준(A) (1)~(5)). reports/{run_id}/에 metrics.json·figures·csv를 쓴다.

    python3 -m src.data.quality_report --run-id 20261110_A_dq_v1

표는 공용 I/O로만 읽는다 (R-18). 예전 env 판에 있던 pykrx 교차검증(거래일·수정종가 대조)은 넣지 않았다 —
pykrx 자동화 수집은 약관 위반으로 폐기했기 때문이다(`src/data/build_openapi.py` 참고). 거래일은 코스피 200
종가가 있는 날로 정의하고(Q-14), 수정종가는 adjust 이벤트 분포로 간접 점검한다(Q-05).

같은 run_id 폴더가 있으면 덮어쓰지 않고 멈춘다 (R-11).
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.common import io

RUN_ID_RE = re.compile(r"^\d{8}_[A-Za-z]+_[A-Za-z0-9_]+$")
# 6.1절 완료 기준 (4): 액면분할 종목의 수정종가가 끊기지 않는지 본다
SPLIT_CASES = (("005930", "삼성전자", "2018-05-04", "50:1"),
               ("035420", "NAVER", "2018-10-12", "5:1"),
               ("035720", "카카오", "2021-04-15", "5:1"))
SPLIT_CONTINUOUS_MAX = 0.31   # 분할 전후 수정종가 일변화가 이보다 작으면 '연속'으로 본다 (상·하한가 30%)
TICKER_RE = r"[0-9A-Z]{6}"


def check_calendar(index: pd.DataFrame) -> dict:
    """완료 기준 (1): 거래일 캘린더. index에는 코스닥 150도 섞일 수 있어 코스피 200만 본다 (Q-14)."""
    kospi200 = index[index["index_code"] == "KOSPI200"]
    days = pd.DatetimeIndex(kospi200["date"])
    return {"n_days": int(len(days)), "first": str(days.min().date()), "last": str(days.max().date()),
            "duplicates": int(days.duplicated().sum()),
            "close_nonpositive_or_nan": int((~(kospi200["close"] > 0)).sum())}


def check_delisted(tickers: pd.DataFrame, prices: pd.DataFrame, delisted: pd.DataFrame,
                   delisted_reused: pd.DataFrame | None = None) -> dict:
    """완료 기준 (2): 2010년 이후 폐지 종목이 시세에 남아 있는지 (생존 편향 점검).

    `tickers.delisting_date`는 **그 종목코드의 가장 최근 회사** 기준이다. 폐지된 코드를 몇 년 뒤 다른
    회사가 다시 쓴 경우(DEC-28: 101970·036220) `tickers`에는 신회사 상장 정보만 남고 구회사 폐지는
    `delisted_reused`에만 있다. 그래서 두 집계를 분리하고, `delisted_reused`가 없으면 "전부 확인"으로
    보고하지 않는다 (C 리뷰 #45)."""
    t = tickers[tickers["delisting_date"].notna() & (tickers["delisting_date"] >= "2010-01-04")]
    in_prices = set(prices["ticker"].unique())
    by_type = t.assign(present=t["ticker"].isin(in_prices)).groupby("security_type")["present"].agg(["size", "sum"])
    out = {"scope": "tickers.delisting_date 기준 (종목코드별 최근 회사). 재사용 코드의 구회사는 reused 항목에서 따로 본다",
           "delisted_since_2010_total": int(len(t)), "in_prices": int(t["ticker"].isin(in_prices).sum()),
           "by_type": {k: {"n": int(v["size"]), "in_prices": int(v["sum"])} for k, v in by_type.iterrows()}}
    common = t[t["security_type"] == "common"]
    out["common_delisted"] = int(len(common))
    out["common_delisted_in_prices"] = int(common["ticker"].isin(in_prices).sum())
    out["common_delisted_with_last_price"] = int(
        delisted.set_index("ticker").reindex(common["ticker"])["last_price"].notna().sum())
    out["reused"] = check_reused_delisted(prices, delisted_reused)
    out["complete"] = bool(out["reused"]["checked"])
    return out


def check_reused_delisted(prices: pd.DataFrame, delisted_reused: pd.DataFrame | None) -> dict:
    """재사용 종목코드의 **구회사** 폐지 이력이 시세에 남아 있는지 (DEC-28, C 리뷰 #45).

    구회사 구간은 폐지일 이전 시세다. 같은 코드에 신회사 시세가 이어 붙어 있으므로 날짜로 나눠 본다."""
    if delisted_reused is None or not len(delisted_reused):
        return {"checked": False,
                "reason": "delisted_reused 입력이 없다 — 재사용 코드의 구회사 폐지 이력은 점검하지 못했다 (DEC-28)"}
    rows = []
    for r in delisted_reused.itertuples():
        before = prices[(prices["ticker"] == r.ticker) & (prices["date"] < r.delist_date)].sort_values("date")
        after = prices[(prices["ticker"] == r.ticker) & (prices["date"] >= r.delist_date)]
        last_raw = float(before["close_raw"].iloc[-1]) if len(before) else None
        rows.append({"ticker": r.ticker, "delist_date": str(pd.Timestamp(r.delist_date).date()),
                     "rows_before_delist": int(len(before)), "rows_after_delist": int(len(after)),
                     "last_close_raw_before_delist": last_raw, "last_price": float(r.last_price),
                     "last_price_matches_close_raw": last_raw is not None and abs(last_raw - float(r.last_price)) < 1e-9})
    return {"checked": True, "n": len(rows),
            "all_old_company_rows_present": all(x["rows_before_delist"] > 0 for x in rows),
            "all_last_price_matches": all(x["last_price_matches_close_raw"] for x in rows), "detail": rows}


def check_ticker_format(tables: dict) -> dict:
    """완료 기준 (3): 모든 표의 종목코드가 6자리 형식인지."""
    return {name: bool(df["ticker"].dropna().astype(str).str.fullmatch(TICKER_RE).all()) for name, df in tables.items()}


def check_splits(prices: pd.DataFrame) -> tuple:
    """완료 기준 (4): 액면분할일 전후 수정종가가 끊기지 않는지. (판정 목록, 원자료 행)을 돌려준다."""
    rows, result = [], []
    for ticker, name, day, ratio in SPLIT_CASES:
        p = prices[prices["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        position = p.index[p["date"] == pd.Timestamp(day)]
        if len(position) == 0:
            result.append({"ticker": ticker, "name": name, "date": day, "ratio": ratio, "status": "시세 없음"})
            continue
        i = int(position[0])
        window = p.iloc[max(0, i - 2): i + 3][["date", "close_raw", "close"]].copy()
        window.insert(0, "ticker", ticker)
        rows.append(window)
        step = p["close"].pct_change().iloc[max(1, i - 2): i + 3].abs().max()
        raw_step = p["close_raw"].pct_change().iloc[max(1, i - 2): i + 3].abs().max()
        result.append({"ticker": ticker, "name": name, "date": day, "ratio": ratio,
                       "max_abs_daily_change_adjusted": float(step), "max_abs_daily_change_raw": float(raw_step),
                       "status": "연속" if step < SPLIT_CONTINUOUS_MAX else "끊김"})
    return result, (pd.concat(rows) if rows else pd.DataFrame())


def check_universe_size(universe: pd.DataFrame, status: pd.DataFrame) -> tuple:
    """완료 기준 (5): 신호일별 유니버스 종목 수 N_t.

    DEC-24 (4): 관리종목 제외(4.4절 3단계)만 빼고 적용한 것이 universe v1의 정의다. 스텁이 아니라 정식
    산출물이지만 어느 단계가 빠졌는지는 리포트에 남긴다 (PM 2026-11-06). 파일 경로가 아니라 is_admin
    실제 값으로 판정한다."""
    nt = universe[universe["in_universe"]].groupby("date").size()
    applied = not bool(status["is_admin"].isna().all())
    return nt, {"n_signal_days": int(len(nt)), "min": int(nt.min()) if len(nt) else None,
                "median": float(nt.median()) if len(nt) else None, "max": int(nt.max()) if len(nt) else None,
                "admin_step_applied": applied,
                "universe_note": "4.4절 1~5단계 전부 적용" if applied else
                "DEC-24 (4) v1: 4.4절 3단계(관리종목 제외) 미적용 — is_admin 원천 미확정(Q-06)"}


def dividend_check(prices: pd.DataFrame, events: pd.DataFrame) -> dict:
    """Q-05: 조정 이벤트가 연말 배당락(12/24~30)에 몰리는지. 현금배당이 기준가에 반영되면 이 시기 비율이 높다."""
    in_window = prices["date"].dt.strftime("%m-%d").between("12-24", "12-30")
    event_window = events["date"].dt.strftime("%m-%d").between("12-24", "12-30")
    rate_all = len(events) / len(prices) if len(prices) else 0.0
    rate_window = event_window.sum() / max(int(in_window.sum()), 1)
    return {"events_total": int(len(events)), "rows_total": int(len(prices)), "event_rate_all_days": float(rate_all),
            "events_dec24_30": int(event_window.sum()), "rows_dec24_30": int(in_window.sum()),
            "event_rate_dec24_30": float(rate_window),
            "ratio_year_end_to_all": float(rate_window / rate_all) if rate_all else None,
            "reading": "비율이 1에 가까우면 배당락 조정이 없고, 수 배 이상이면 현금배당이 반영된 것"}


def listing_vs_first_price(prices: pd.DataFrame, tickers: pd.DataFrame, first_day: pd.Timestamp) -> dict:
    """상장일과 첫 시세일의 간격. 수집 시작 이후 상장한 종목만 비교할 수 있다."""
    first = prices.groupby("ticker")["date"].min()
    listed = tickers.set_index("ticker")["listing_date"].reindex(first.index)
    gap = (first - listed).dt.days
    new = first > first_day + pd.Timedelta(days=7)
    return {"compared": int(new.sum()), "first_price_more_than_7d_after_listing": int((gap[new] > 7).sum()),
            "first_price_before_listing": int((gap[new] < 0).sum()), "listing_date_missing": int(listed.isna().sum())}


def _n_t_figure(nt: pd.Series, path: Path, admin_step_applied: bool) -> bool:
    try:
        import matplotlib
        import matplotlib.font_manager
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    for family in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(family == f.name for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = family
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(nt.index, nt.to_numpy(), lw=1)
    ax.set(title="signal-date universe size N_t" + ("" if admin_step_applied else " (v1: no admin-issue filter)"),
           ylabel="N_t")
    ax.axhline(500, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=io.ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(run_id: str, out_root: Path | None = None) -> dict:
    """리포트를 만들어 reports/{run_id}/에 쓰고 metrics를 돌려준다."""
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"run_id는 '{{YYYYMMDD}}_{{역할ID}}_{{짧은설명}}' 형식이어야 한다: {run_id!r}")
    out = Path(out_root or (io.ROOT / "reports")) / run_id
    if out.exists():
        raise FileExistsError(f"{out}가 이미 있다. 덮어쓰지 않는다 (R-11)")

    index, tickers, prices = io.read_table("index"), io.read_table("tickers"), io.read_table("prices")
    delisted, universe, status = io.read_table("delisted"), io.read_table("universe"), io.read_table("status")
    try:
        reused = io.read_table("delisted_reused")       # DEC-28. 없으면 (2)를 "전부 확인"으로 보고하지 않는다
    except FileNotFoundError:
        reused = None

    metrics = {"run_id": run_id, "source_commit": _git_commit()}
    metrics["c1_calendar"] = check_calendar(index)
    metrics["c2_delisted"] = check_delisted(tickers, prices, delisted, reused)
    metrics["c3_ticker_format"] = check_ticker_format(
        {"prices": prices, "tickers": tickers, "status": status, "universe": universe, "delisted": delisted})
    splits, split_rows = check_splits(prices)
    metrics["c4_splits"] = splits
    nt, metrics["c5_nt"] = check_universe_size(universe, status)

    metrics["extra_security_types"] = {str(k): int(v) for k, v in tickers["security_type"].value_counts().items()}
    metrics["extra_listing_vs_first_price"] = listing_vs_first_price(
        prices, tickers, pd.Timestamp(metrics["c1_calendar"]["first"]))
    metrics["extra_investor_flows"] = {"applicable": False, "reason": "investor_flows를 만들지 않는다 (DEC-22, 원천 없음)"}
    events_path = io.ROOT / "data" / "interim" / "adjust_events.parquet"
    metrics["extra_cash_dividend_q05"] = (dividend_check(prices, pd.read_parquet(events_path))
                                          if events_path.is_file() else "adjust_events.parquet 없음")

    (out / "figures").mkdir(parents=True, exist_ok=True)
    split_rows.to_csv(out / "split_cases.csv", index=False, encoding="utf-8-sig")
    metrics["figure_written"] = _n_t_figure(nt, out / "figures" / "n_t.png", metrics["c5_nt"]["admin_step_applied"])
    nt.rename("n_t").to_csv(out / "figures" / "n_t.csv", encoding="utf-8")
    (out / "config.yaml").write_text(f"run_id: {run_id}\ncommand: {' '.join(sys.argv)}\n", encoding="utf-8")
    (out / "git_commit.txt").write_text(metrics["source_commit"] + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description="A-03 데이터 품질 리포트")
    ap.add_argument("--run-id", required=True, help="{YYYYMMDD}_A_{짧은설명}")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.run_id), ensure_ascii=False, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
