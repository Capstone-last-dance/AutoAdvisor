"""M6 템플릿 근거 요약 (명세 6.7절)."""
from __future__ import annotations

import re

import pandas as pd

from src.common import io
from src.data.calendar import signal_dates
from src.data.disclosures import build_disclosures
from src.events import alerts as m9

_CLOSE = pd.Timedelta(hours=15, minutes=30)
_INPUTS = {"signals", "index", "tickers", "alerts"}
_DART_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo="
_DISCLOSURE_TYPES = {"B": "주요사항보고", "I": "거래소공시"}
_NO_EVIDENCE = "최근 20거래일 공시: 조회된 근거가 없습니다."
_RESTRICTED_WORDS = re.compile(r"매수|매도|추천|(?:가격|주가)\s*전망|목표\s*주가")


def m6_step(ctx) -> None:
    """신호일의 공시 근거를 6단계 출력으로 저장한다.

    직전 신호일은 공용 거래일 달력에서 t 이전 주의 마지막 거래일이다.
    M9 경보는 [직전 신호일 15:30, t 15:30) KST에 이용 가능해진 것을 쓴다.
    진입 후보 20개는 급락·쿨다운을 먼저 제외한 순위순으로 정한다(동률은 종목코드순).
    중복 종목은 crash > alert > rank 순으로 한 줄만 남긴다.
    공시는 t−19~t 거래일 접수분 중 t 15:30 전에 이용 가능한 최근 세 건을 인용한다.
    뉴스와 LLM 요약은 이 템플릿 범위 밖이다.
    """
    date = ctx.date
    stubbed = _INPUTS & ctx.stub_tables
    if stubbed:
        io.write_evidence(date, [], stub=True)
        ctx.stub_tables.add("evidence")
        ctx.log.warning("6단계 PM: 입력 표가 스텁이라(%s) 0행 evidence를 스텁으로 쓴다", ", ".join(sorted(stubbed)))
        return

    day = pd.Timestamp(date)
    trading_days = m9.load_trading_days()
    position = trading_days.get_indexer([day])[0]
    if position < 0:
        raise ValueError(f"{date}: 거래일 달력에 신호일이 없다")
    weekly = signal_dates(trading_days[:position + 1])
    if len(weekly) < 2:
        raise ValueError(f"{date}: 거래일 달력에 직전 신호일이 없다")
    previous = weekly[-2]
    cutoff = day.tz_localize(m9.KST) + _CLOSE
    receipt_days = trading_days[max(0, position - 19):position + 1]

    signals = ctx.read_table("signals", date=date)
    cooldown = m9.cooldown_at(date)
    recent_alerts = _recent_alert_tickers(previous, day, cutoff)
    eligible = signals.loc[signals["crash_flag"].eq(0) & ~signals["ticker"].isin(cooldown)]
    ranked = eligible.sort_values(["rank", "ticker"]).head(20)["ticker"]
    targets = dict.fromkeys(ranked, "rank")
    targets.update(dict.fromkeys(recent_alerts, "alert"))
    targets.update(dict.fromkeys(signals.loc[signals["crash_flag"].eq(1), "ticker"], "crash"))

    tickers = ctx.read_table("tickers")
    documents = _disclosures(day, receipt_days, cutoff, tickers)
    selected = documents[documents["ticker"].isin(targets)]
    ordered = selected.sort_values(["available_at", "rcept_no"], ascending=False)
    by_ticker = {ticker: group.head(3) for ticker, group in ordered.groupby("ticker")}
    records = [
        _evidence_record(date, ticker, signal_type, by_ticker.get(ticker))
        for ticker, signal_type in sorted(targets.items())
    ]
    io.write_evidence(date, records)


def _recent_alert_tickers(previous: pd.Timestamp, day: pd.Timestamp, cutoff: pd.Timestamp) -> set[str]:
    dates = pd.date_range(previous, day, freq="D").strftime("%Y-%m-%d").tolist()
    # M9의 일자별 읽기 계약을 재사용한다. 미수집을 경보 0건으로 처리하지 않는다.
    alerts = m9._read_dates(dates)
    start = previous.tz_localize(m9.KST) + _CLOSE
    in_window = alerts["available_at"].ge(start) & alerts["available_at"].lt(cutoff)
    return set(alerts.loc[in_window, "ticker"])


def _disclosures(day: pd.Timestamp, receipt_days: pd.DatetimeIndex, cutoff: pd.Timestamp,
                 tickers: pd.DataFrame) -> pd.DataFrame:
    raw = io.ROOT / "data" / "raw"
    runs = sorted(path.parent for path in raw.glob("*/*_PM_disclosures_*/opendart_list/manifest.json"))
    built = build_disclosures(runs, today=day.date(), tickers=tickers)
    frame = built.df
    frame = frame.loc[frame["rcept_date"].isin(receipt_days) & frame["available_at"].lt(cutoff)]
    observations = built.observations
    if not observations.empty:
        observations = observations[observations["rcept_no"].isin(frame["rcept_no"])]
    frame = m9.link_tickers(m9.attach_stock_codes(frame, observations), tickers)
    frame["disclosure_type"] = "공시"
    if not observations.empty:
        kinds = observations["pblntf_ty"].map(_DISCLOSURE_TYPES).dropna()
        labels = kinds.groupby(observations["rcept_no"]).agg(lambda values: "/".join(sorted(set(values))))
        frame["disclosure_type"] = frame["rcept_no"].map(labels).fillna("공시")
    return frame


def _evidence_record(date: str, ticker: str, signal_type: str, documents: pd.DataFrame | None) -> dict:
    lines = []
    sources = []
    if documents is not None:
        for n, row in enumerate(documents.itertuples(index=False), 1):
            title = _RESTRICTED_WORDS.sub("…", " ".join(row.report_nm.split()))
            lines.append(f"최근 20거래일 공시: {row.disclosure_type} {title} ({row.rcept_date:%Y-%m-%d}) [{n}]")
            sources.append({
                "n": n,
                "type": "disclosure",
                "id": row.rcept_no,
                "title": row.report_nm,
                "url": _DART_URL + row.rcept_no,
                "timestamp": row.available_at.tz_convert(m9.KST).isoformat(),
            })
    return {
        "date": date,
        "ticker": ticker,
        "signal_type": signal_type,
        "summary": "\n".join(lines) or _NO_EVIDENCE,
        "sources": sources,
        "disclaimer": io.DISCLAIMER,
    }
