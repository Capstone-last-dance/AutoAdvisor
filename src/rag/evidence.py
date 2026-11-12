"""M6 템플릿 근거 요약 (명세 6.7절)."""
from __future__ import annotations

import re

import pandas as pd

from src.common import io
from src.data.calendar import signal_dates
from src.data.disclosures import DATE_ONLY_AVAILABLE_TIME, build_disclosures
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
    공시 버전과 B·I 수집 완료 여부는 다음 거래일 09:00 KST 기록 마감 전 원본으로 고정한다.
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
    next_day = trading_days[m9._next_day_index(trading_days, day)]
    record_deadline = next_day.tz_localize(m9.KST) + m9.RECORD_DEADLINE
    documents = _disclosures(day, receipt_days, cutoff, record_deadline, tickers)
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
                 record_deadline: pd.Timestamp, tickers: pd.DataFrame) -> pd.DataFrame:
    raw = io.ROOT / "data" / "raw"
    runs = sorted(path.parent for path in raw.glob("*/*_PM_disclosures_*/opendart_list/manifest.json"))
    built = build_disclosures(runs, today=day.date(), tickers=tickers)
    windows = _windows_completed_before(built.windows, record_deadline)
    # DEC-09의 날짜 전용 원본에서 t일 공시는 이용 가능 시각을 만족할 수 없다.
    required = receipt_days[receipt_days < day].strftime("%Y-%m-%d").tolist()
    m9._require_windows_complete(windows, required)
    if built.observations.empty:
        return built.df.iloc[:0]

    observations = built.observations
    window_key = ["run_id", "window_id"]
    completed = pd.MultiIndex.from_frame(windows[window_key])
    observations = observations[pd.MultiIndex.from_frame(observations[window_key]).isin(completed)]
    fetched = pd.to_datetime(observations["fetched_at"], format="ISO8601", utc=True)
    observations = observations.assign(fetched_at=fetched).loc[fetched.lt(record_deadline)]

    # build_disclosures의 df는 전체 관측 중 최신 버전이다. 마감 전 관측에서
    # 제목과 종목코드가 같은 버전을 가리키도록 접수번호별로 다시 선택한다.
    latest = _latest_versions(observations, receipt_days)
    frame = latest.loc[latest["rcept_date"].isin(receipt_days) & latest["available_at"].lt(cutoff)]
    codes = frame[["rcept_no", "stock_code", "fetched_at"]]
    frame = m9.link_tickers(m9.attach_stock_codes(frame.drop(columns="stock_code"), codes), tickers)
    kinds = observations["pblntf_ty"].map(_DISCLOSURE_TYPES).dropna()
    labels = kinds.groupby(observations["rcept_no"]).agg(lambda values: "/".join(sorted(set(values))))
    frame["disclosure_type"] = frame["rcept_no"].map(labels).fillna("공시")
    return frame


def _windows_completed_before(windows: pd.DataFrame, deadline: pd.Timestamp) -> pd.DataFrame:
    if windows.empty:
        return windows
    complete = windows.loc[windows["state"].eq("complete")]
    keep = []
    for window in complete.itertuples(index=False):
        accepted = set(window.accepted_attempts)
        finished = {
            attempt["n"] for attempt in window.attempts
            if attempt["finished_at"] and pd.Timestamp(attempt["finished_at"]) < deadline
        }
        keep.append(bool(accepted) and accepted <= finished)
    return complete.loc[keep]


def _latest_versions(observations: pd.DataFrame, receipt_days: pd.DatetimeIndex) -> pd.DataFrame:
    # 창에 한 번이라도 접수된 번호는 창 밖으로 바뀐 버전까지 비교한다.
    candidates = observations.loc[observations["rcept_dt"].isin(receipt_days.strftime("%Y%m%d")), "rcept_no"]
    rows = observations.loc[observations["rcept_no"].isin(candidates)]
    receipt = pd.to_datetime(rows["rcept_dt"], format="%Y%m%d", errors="coerce")
    number_date = pd.to_datetime(rows["rcept_no"].str[:8], format="%Y%m%d", errors="coerce")
    valid = (
        rows["rcept_no"].str.fullmatch(r"\d{14}") & rows["rcept_dt"].str.fullmatch(r"\d{8}")
        & receipt.notna() & number_date.notna()
    )
    available_day = receipt.where(receipt >= number_date, number_date)
    available = available_day.dt.tz_localize(m9.KST) + pd.Timedelta(DATE_ONLY_AVAILABLE_TIME.isoformat())
    versions = rows.assign(rcept_date=receipt, available_at=available).loc[valid]
    return versions.sort_values(["fetched_at", "run_id", "window_id"], kind="stable").drop_duplicates(
        "rcept_no", keep="last"
    )


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
