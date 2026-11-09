"""M9 이벤트 경보 v1 (명세서 6.10절). PM 소유.

공시 목록에서 경보를 만들고, B·C가 쓰는 쿨다운과 이벤트 청산 일정을 계산한다.

- 청산: 경보의 `available_at`이 속한 KST 날짜의 다음 KRX 거래일 종가
- 쿨다운: 그 거래일부터 `events.cooldown_trading_days` 거래일 동안 신규 진입 금지
- 같은 종목·유형의 후보가 진행 중인 쿨다운 안에 또 나오면 v1은 같은 사건으로 보고 종료일을 늘리지 않는다
- 신호일 t에는 `available_at < t 15:30 KST`인 경보만 쓴다 (DEC-09)

종목 연결: OpenDART 목록의 종목코드가 A의 `tickers`에 있고, 접수일이 그 상장 구간 안이며,
입력 전체에서 회사코드와 종목코드가 1:1일 때만 붙인다 (DEC-09 (2)).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.common import io
from src.events.triggers import ALERT, classify

KST = "Asia/Seoul"
ALERT_COLUMNS = ["date", "ticker", "trigger_type", "rcept_no", "report_nm", "rcept_dt", "available_at"]
LINKED = "linked"
FINAL_ALERT = "alert"
SAME_EVENT = "same_event"
UNLINKED = "unlinked"


def attach_stock_codes(disclosures: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """`build_disclosures`의 원본 관측에서 접수번호별 최신 종목코드를 붙인다."""
    latest = observations.assign(_at=pd.to_datetime(observations["fetched_at"], utc=True)).sort_values("_at")
    codes = latest.drop_duplicates("rcept_no", keep="last").set_index("rcept_no")["stock_code"]
    out = disclosures.copy()
    out["stock_code"] = out["rcept_no"].map(codes).fillna("").astype(str)
    return out


def link_tickers(disclosures: pd.DataFrame, tickers: pd.DataFrame) -> pd.DataFrame:
    """`ticker`(연결되지 않으면 None)와 `link_status`를 더한다. 입력은 바꾸지 않는다."""
    out = disclosures.copy()
    code = out["stock_code"].fillna("").astype(str)
    coded = out.loc[code != "", ["corp_code"]].assign(code=code[code != ""])
    ambiguous_codes = set(coded.groupby("code")["corp_code"].nunique().loc[lambda s: s > 1].index)
    ambiguous_corps = set(coded.groupby("corp_code")["code"].nunique().loc[lambda s: s > 1].index)

    table = tickers.drop_duplicates("ticker").set_index("ticker")
    listing = code.map(table["listing_date"])
    delisting = code.map(table["delisting_date"])
    day = pd.to_datetime(out["rcept_date"])
    in_listing = (listing <= day) & (delisting.isna() | (day <= delisting))

    status = pd.Series("no_code", index=out.index, dtype=object)
    status[code != ""] = "not_in_tickers"
    status[(code != "") & listing.notna()] = "outside_listing"
    status[(code != "") & listing.notna() & in_listing] = LINKED
    ambiguous = code.isin(ambiguous_codes) | out["corp_code"].isin(ambiguous_corps)
    status[(code != "") & ambiguous] = "ambiguous"
    out["link_status"] = status
    out["ticker"] = code.where(status == LINKED, None).astype(object).where(status == LINKED, None)
    return out


@dataclass
class AlertBuild:
    alerts: pd.DataFrame     # 9.3절 alerts 열
    decisions: pd.DataFrame  # 접수번호·유형별 판정 진단


def build_alerts(disclosures: pd.DataFrame, tickers: pd.DataFrame, events: Mapping[str, Any],
                 trading_days: Iterable) -> AlertBuild:
    """공시 표로 경보를 만든다. `disclosures`에는 `stock_code` 열이 있어야 한다."""
    days = _day_index(trading_days)
    linked = link_tickers(disclosures, tickers)
    cache = {title: classify(title, events) for title in linked["report_nm_norm"].unique()}
    rows = []
    for row in linked.itertuples(index=False):
        for decision in cache[row.report_nm_norm]:
            rows.append({"rcept_no": row.rcept_no, "trigger_type": decision.trigger_type, "status": decision.status,
                         "reason": decision.reason, "link_status": row.link_status, "ticker": row.ticker,
                         "rcept_date": row.rcept_date, "rcept_dt": row.rcept_dt, "available_at": row.available_at,
                         "corp_code": row.corp_code, "corp_name": row.corp_name, "report_nm": row.report_nm})
    columns = ["rcept_no", "trigger_type", "status", "reason", "link_status", "ticker", "rcept_date", "rcept_dt",
               "available_at", "corp_code", "corp_name", "report_nm"]
    decisions = pd.DataFrame(rows, columns=columns)
    if decisions.empty:
        decisions["final"] = pd.Series(dtype=object)
        return AlertBuild(_alert_frame(decisions), decisions)
    decisions = decisions.sort_values(["available_at", "rcept_no", "trigger_type"]).reset_index(drop=True)
    decisions["final"] = _final_status(decisions, days, int(events["cooldown_trading_days"]))
    return AlertBuild(_alert_frame(decisions[decisions["final"] == FINAL_ALERT]), decisions)


def _final_status(decisions: pd.DataFrame, days: pd.DatetimeIndex, cooldown: int) -> list[str]:
    finals: list[str] = []
    active_end: dict[tuple[str, str], int] = {}
    for row in decisions.itertuples(index=False):
        if row.status != ALERT:
            finals.append(row.status)
        elif row.link_status != LINKED:
            finals.append(UNLINKED)
        else:
            start = _next_day_index(days, _kst_date(row.available_at))
            key = (row.ticker, row.trigger_type)
            if key in active_end and start <= active_end[key]:
                finals.append(SAME_EVENT)
            else:
                active_end[key] = start + cooldown - 1
                finals.append(FINAL_ALERT)
    return finals


def _alert_frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({column: rows[column] for column in ALERT_COLUMNS if column in rows.columns}).copy()
    available = _as_kst(out["available_at"])
    out["available_at"] = available
    out["rcept_dt"] = _as_kst(out["rcept_dt"])
    out["date"] = available.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    return out[ALERT_COLUMNS].reset_index(drop=True)


def _as_kst(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.tz_convert(KST).astype(pd.DatetimeTZDtype("ns", KST))


class AlertBook:
    """경보에서 쿨다운과 이벤트 청산 일정을 계산한다."""

    def __init__(self, alerts: pd.DataFrame, trading_days: Iterable, cooldown_days: int, cutoff_time: str = "15:30"):
        self.days = _day_index(trading_days)
        self.cooldown_days = int(cooldown_days)
        self.cutoff = pd.Timedelta(hours=int(cutoff_time[:2]), minutes=int(cutoff_time[3:]))
        rows = alerts.sort_values(["available_at", "rcept_no"]) if len(alerts) else alerts
        self.available = pd.DatetimeIndex(rows["available_at"]).tz_convert(KST) if len(rows) else None
        self.tickers = rows["ticker"].astype(str).to_numpy() if len(rows) else np.array([], dtype=object)
        self.start = np.array([_next_day_index(self.days, _kst_date(a)) for a in rows["available_at"]], dtype=int)
        self.end = self.start + self.cooldown_days - 1

    def _cutoff_at(self, t: str) -> pd.Timestamp:
        return pd.Timestamp(t).tz_localize(KST) + self.cutoff

    def _known(self, t: str) -> np.ndarray:
        if self.available is None:
            return np.zeros(0, dtype=bool)
        return np.asarray(self.available < self._cutoff_at(t))

    def cooldown_at(self, t: str) -> set[str]:
        """신호일 t의 다음 거래일에 신규 진입이 금지된 종목. t 15:30 전에 이용 가능한 경보만 쓴다."""
        execution = _next_day_index(self.days, pd.Timestamp(t))
        active = self._known(t) & (self.start <= execution) & (execution <= self.end)
        return set(self.tickers[active])

    def engine_alerts(self, first_signal: str, t: str) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
        """(cooldowns, event_liquidations). 첫 신호일 다음 날부터 t까지 거래일별 종목 집합이다."""
        first = _next_day_index(self.days, pd.Timestamp(first_signal))
        last = int(self.days.searchsorted(pd.Timestamp(t).normalize(), side="right")) - 1
        known = self._known(t)
        cooldowns: dict[str, set[str]] = {}
        liquidations: dict[str, set[str]] = {}
        for i, ticker in zip(np.flatnonzero(known), self.tickers[known]):
            start, end = int(self.start[i]), int(self.end[i])
            for position in range(max(start, first), min(end, last) + 1):
                cooldowns.setdefault(self._key(position), set()).add(ticker)
            if first <= start <= last:
                liquidations.setdefault(self._key(start), set()).add(ticker)
        return ({day: frozenset(names) for day, names in cooldowns.items()},
                {day: frozenset(names) for day, names in liquidations.items()})

    def _key(self, position: int) -> str:
        return self.days[position].strftime("%Y-%m-%d")


def write_alert_files(alerts: pd.DataFrame, stub: bool = False) -> list[str]:
    """`alerts/{d}.parquet`를 날짜별로 쓴다. 경보가 없는 날은 파일을 만들지 않는다."""
    written = []
    for day, part in alerts.groupby("date"):
        date = pd.Timestamp(day).strftime("%Y-%m-%d")
        io.write_table("alerts", part.reset_index(drop=True), date=date, stub=stub)
        written.append(date)
    return written


def read_alerts(run_id: str | None = None) -> pd.DataFrame:
    """백테스트는 `run_id` 파일을, forward는 날짜별 파일 전체를 읽는다. 파일이 없으면 빈 표다."""
    if run_id is not None:
        return io.read_table("alerts", run_id=run_id)
    folder = io.ROOT / "alerts"
    files = sorted(p.stem for p in folder.glob("*.parquet") if not p.stem.startswith("wf_")) if folder.exists() else []
    if not files:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    return pd.concat([io.read_table("alerts", date=day) for day in files], ignore_index=True)


def load_trading_days() -> pd.DatetimeIndex:
    """코스피 200 종가가 있는 날에, 그 뒤 공휴일 표로 아는 거래일을 이어 붙인다 (t+1이 아직 없는 forward용)."""
    from src.pipeline.forward_test import CALENDAR_END, KRX_CLOSED_DAYS

    known = _day_index(io.read_table("index")["date"])
    future = pd.bdate_range(known[-1] + pd.Timedelta(days=1), CALENDAR_END)
    return known.append(future[~future.isin(KRX_CLOSED_DAYS)])


def load_book(run_id: str | None = None) -> AlertBook:
    from src.portfolio.params import load_prereg

    cfg = load_prereg()
    return AlertBook(read_alerts(run_id), load_trading_days(), cfg["events"]["cooldown_trading_days"],
                     cfg["calendar"]["signal_cutoff_time"])


def cooldown_at(t: str) -> set[str]:
    """B의 CooldownFunction. 신호일 t → t+1 신규 진입 금지 종목."""
    return load_book().cooldown_at(t)


def engine_alerts(first_signal: str, t: str) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """B의 EngineAlertsFunction. (cooldowns, event_liquidations)."""
    return load_book().engine_alerts(first_signal, t)


def _day_index(values: Iterable | Sequence) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(pd.to_datetime(list(values))).normalize().unique().sort_values()
    if days.empty:
        raise ValueError("거래일 달력이 비어 있다")
    return days


def _kst_date(value) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert(KST).tz_localize(None).normalize()


def _next_day_index(days: pd.DatetimeIndex, day: pd.Timestamp) -> int:
    position = int(days.searchsorted(day.normalize(), side="right"))
    if position >= len(days):
        raise ValueError(f"{day:%Y-%m-%d} 다음 거래일을 달력에서 찾을 수 없다")
    return position
