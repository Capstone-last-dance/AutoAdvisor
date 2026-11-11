"""M9 이벤트 경보 v1 (명세서 6.10절). PM 소유.

공시 목록에서 경보를 만들고, B·C가 쓰는 쿨다운과 이벤트 청산 일정을 계산한다.

- 청산: 경보의 `available_at`이 속한 KST 날짜의 다음 KRX 거래일 종가
- 쿨다운: 그 거래일부터 `events.cooldown_trading_days` 거래일 동안 신규 진입 금지
- 같은 종목·유형의 후보가 진행 중인 쿨다운 안에 또 나오면 v1은 같은 사건으로 보고 종료일을 늘리지 않는다
- 신호일 t에는 `available_at < t 15:30 KST`인 경보만 쓴다 (DEC-09)
- 체결일 d의 쿨다운·청산에는 d 15:30 KST 전에 만들어진 경보(`generated_at`)만 쓴다 (DEC-32). 나중에 채운 경보가
  지난 체결 재생을 바꾸지 않게 하는 규칙이다. 늦게 만든 경보는 생성 뒤 첫 체결일에 청산하고 남은 쿨다운을 적용한다.
  백테스트 파일은 `generated_at = available_at`이라 영향이 없다

종목 연결: OpenDART 목록의 종목코드가 A의 `tickers`에 있고, 접수일이 그 상장 구간 안이며,
입력 전체에서 회사코드와 종목코드가 1:1일 때만 붙인다 (DEC-09 (2)). 일간 실행의 입력 전체는 그날 공시다.

날짜별 파일 (DEC-33): `alerts/{d}.parquet`는 달력일 d의 수집이 끝났다는 뜻이고 경보가 없으면 0행이다. 파일이 없으면
미수집이다. forward 계산은 필요한 날짜의 파일만 읽고, 그중 하나라도 없으면 멈춘다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common import io
from src.events.triggers import ALERT, classify

KST = "Asia/Seoul"
ALERT_COLUMNS = ["date", "ticker", "trigger_type", "rcept_no", "report_nm", "rcept_dt", "available_at", "generated_at"]
LINKED = "linked"
FINAL_ALERT = "alert"
SAME_EVENT = "same_event"
UNLINKED = "unlinked"
DISCLOSURE_TYPES = ("B", "I")
# 달력일 d 00:00부터 잰다. OpenDART 목록은 접수일만 줘서 d 다음 날 08:00 KST 전의 결과는 확정이 아니다 (DEC-33)
CONFIRM_AFTER = pd.Timedelta(days=1, hours=8)


class AlertsNotCollected(FileNotFoundError):
    """필요한 날짜의 경보를 아직 만들지 못했다. 경보 0건과 다르므로 호출한 쪽은 결과를 쓰지 않고 멈춘다."""


class AlertsNotStarted(AlertsNotCollected):
    """날짜별 경보 파일이 처음부터 하나도 없다. B는 이 경우만 M9 없이 기록한다 (PM 11/9 결정)."""


def attach_stock_codes(disclosures: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """`build_disclosures`의 원본 관측에서 접수번호별 최신 종목코드를 붙인다."""
    out = disclosures.copy()
    if observations.empty:  # 정상 수집 0건이면 원본 관측의 열 자체가 없다
        out["stock_code"] = ""
        return out
    latest = observations.assign(_at=pd.to_datetime(observations["fetched_at"], utc=True)).sort_values("_at")
    codes = latest.drop_duplicates("rcept_no", keep="last").set_index("rcept_no")["stock_code"]
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
                 trading_days: Iterable, *, history: pd.DataFrame | None = None,
                 generated_at: pd.Timestamp | None = None) -> AlertBuild:
    """공시 표로 경보를 만든다. `disclosures`에는 `stock_code` 열이 있어야 한다.

    `history`는 입력보다 앞선 경보로, 같은 사건 판정의 시작 상태다. `generated_at`을 주지 않으면 백테스트처럼
    `available_at`과 같은 값을 쓴다.
    """
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
        return AlertBuild(_alert_frame(decisions, generated_at), decisions)
    decisions = decisions.sort_values(["available_at", "rcept_no", "trigger_type"]).reset_index(drop=True)
    decisions["final"] = _final_status(decisions, days, int(events["cooldown_trading_days"]), history)
    return AlertBuild(_alert_frame(decisions[decisions["final"] == FINAL_ALERT], generated_at), decisions)


def _final_status(decisions: pd.DataFrame, days: pd.DatetimeIndex, cooldown: int,
                  history: pd.DataFrame | None = None) -> list[str]:
    finals: list[str] = []
    active_end = _active_ends(history, days, cooldown)
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


def _active_ends(history: pd.DataFrame | None, days: pd.DatetimeIndex, cooldown: int) -> dict[tuple[str, str], int]:
    """앞선 경보에서 (종목, 유형)별 진행 중인 쿨다운의 마지막 거래일 위치를 만든다."""
    ends: dict[tuple[str, str], int] = {}
    if history is None or history.empty:
        return ends
    for row in history.sort_values(["available_at", "rcept_no"]).itertuples(index=False):
        ends[(row.ticker, row.trigger_type)] = _next_day_index(days, _kst_date(row.available_at)) + cooldown - 1
    return ends


def _alert_frame(rows: pd.DataFrame, generated_at: pd.Timestamp | None = None) -> pd.DataFrame:
    out = pd.DataFrame({column: rows[column] for column in ALERT_COLUMNS if column in rows.columns}).copy()
    available = _as_kst(out["available_at"])
    out["available_at"] = available
    out["rcept_dt"] = _as_kst(out["rcept_dt"])
    out["date"] = available.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    out["generated_at"] = available if generated_at is None else _as_kst(pd.Series(generated_at, index=out.index))
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
        empty_times = pd.DatetimeIndex([], tz=KST)
        self.available = pd.DatetimeIndex(rows["available_at"]).tz_convert(KST) if len(rows) else empty_times
        self.generated = pd.DatetimeIndex(rows["generated_at"]).tz_convert(KST) if len(rows) else empty_times
        self.tickers = rows["ticker"].astype(str).to_numpy() if len(rows) else np.array([], dtype=object)
        self.start = np.array([_next_day_index(self.days, _kst_date(a)) for a in rows["available_at"]], dtype=int)
        self.end = self.start + self.cooldown_days - 1
        # 경보를 적용할 수 있는 첫 체결일은 생성 시각보다 마감이 늦은 첫 거래일이다 (DEC-32).
        closes = self.days.tz_localize(KST) + self.cutoff
        self.applied_from = np.asarray(closes.searchsorted(self.generated, side="right"))

    def _cutoff_at(self, t: str) -> pd.Timestamp:
        return pd.Timestamp(t).tz_localize(KST) + self.cutoff

    def cooldown_at(self, t: str) -> set[str]:
        """t의 15:30 전에 이용 가능하고 생성된 경보로 다음 거래일의 진입 금지 종목을 계산한다."""
        cutoff = self._cutoff_at(t)
        execution = _next_day_index(self.days, pd.Timestamp(t))
        known = (self.available < cutoff) & (self.generated < cutoff)
        active = known & (self.start <= execution) & (execution <= self.end)
        return set(self.tickers[active])

    def engine_alerts(self, first_signal: str, t: str) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
        """(cooldowns, event_liquidations). 첫 신호일 다음 날부터 t까지 거래일별 종목 집합이다.

        체결일 d에는 d 15:30 전에 만들어진 경보만 넣는다. 늦게 만들어진 경보는 생성 뒤 첫 체결일부터 적용하고,
        그날이 아직 쿨다운 기간이면 그날 청산한다. 이미 지난 체결일로 되돌아가지는 않는다 (DEC-32).
        """
        first = _next_day_index(self.days, pd.Timestamp(first_signal))
        last = int(self.days.searchsorted(pd.Timestamp(t).normalize(), side="right")) - 1
        known = self.available < self._cutoff_at(t)
        cooldowns: dict[str, set[str]] = {}
        liquidations: dict[str, set[str]] = {}
        for i, ticker in zip(np.flatnonzero(known), self.tickers[known]):
            start, end, applied = int(self.start[i]), int(self.end[i]), int(self.applied_from[i])
            for position in range(max(start, first, applied), min(end, last) + 1):
                cooldowns.setdefault(self._key(position), set()).add(ticker)
            exit_day = max(start, applied)
            if exit_day <= end and first <= exit_day <= last:
                liquidations.setdefault(self._key(exit_day), set()).add(ticker)
        return ({day: frozenset(names) for day, names in cooldowns.items()},
                {day: frozenset(names) for day, names in liquidations.items()})

    def _key(self, position: int) -> str:
        return self.days[position].strftime("%Y-%m-%d")


def write_alert_files(alerts: pd.DataFrame, stub: bool = False) -> list[str]:
    """`alerts`에 있는 날짜만 `alerts/{d}.parquet`로 쓴다.

    경보가 없는 날은 파일을 만들지 않으므로 forward 운영에는 쓰지 않는다. forward는 `daily_alerts`와
    `backfill_alert_files`로 달력일마다 파일을 남긴다 (DEC-33).
    """
    written = []
    for day, part in alerts.groupby("date"):
        date = pd.Timestamp(day).strftime("%Y-%m-%d")
        io.write_table("alerts", part.reset_index(drop=True), date=date, stub=stub)
        written.append(date)
    return written


def read_alerts(run_id: str | None = None) -> pd.DataFrame:
    """백테스트는 `run_id` 파일을, forward는 날짜별 파일 전체를 읽는다.

    날짜별 파일이 하나도 없으면 `AlertsNotStarted`를 낸다. M9를 실행하지 않은 상태를 경보 0건으로 읽지 않기 위해서다.
    """
    if run_id is not None:
        return io.read_table("alerts", run_id=run_id)
    _require_started()
    return _concat_alerts([io.read_table("alerts", date=day) for day in io.list_dates("alerts")])


def _require_started() -> None:
    if not io.list_dates("alerts"):
        raise AlertsNotStarted("날짜별 경보 파일이 없다. M9 경보 생성을 먼저 실행한다")


def _read_dates(dates: Sequence[str]) -> pd.DataFrame:
    """`dates`의 날짜별 파일만 읽는다. 하나라도 없으면 `AlertsNotCollected`를 낸다."""
    present = set(io.list_dates("alerts"))
    missing = [day for day in dates if day not in present]
    if missing:
        raise AlertsNotCollected(f"경보 파일이 없는 날짜가 {len(missing)}일 있다 ({missing[0]} ~ {missing[-1]}). "
                                 "일간 경보를 먼저 실행한다")
    return _concat_alerts([io.read_table("alerts", date=day) for day in dates])


def _concat_alerts(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    filled = [frame for frame in frames if len(frame)]
    return pd.concat(filled, ignore_index=True) if filled else frames[0]


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


def _forward_dates(first_signal: str, t: str, days: pd.DatetimeIndex, cooldown: int) -> list[str]:
    """`first_signal` 다음 체결일부터 `t`까지 쿨다운·청산을 계산할 때 필요한 달력일 전부."""
    first = _next_day_index(days, pd.Timestamp(first_signal))
    since = days[max(first - cooldown, 0)]
    return _calendar_dates(since, pd.Timestamp(t).normalize())


def _calendar_dates(first: pd.Timestamp, last: pd.Timestamp) -> list[str]:
    return [day.strftime("%Y-%m-%d") for day in pd.date_range(first, last, freq="D")]


def _forward_book(first_signal: str, t: str) -> AlertBook:
    from src.portfolio.params import load_prereg

    _require_started()
    cfg = load_prereg()
    days = load_trading_days()
    cooldown = cfg["events"]["cooldown_trading_days"]
    alerts = _read_dates(_forward_dates(first_signal, t, days, int(cooldown)))
    return AlertBook(alerts, days, cooldown, cfg["calendar"]["signal_cutoff_time"])


def cooldown_at(t: str) -> set[str]:
    """B의 CooldownFunction. 신호일 t → t+1 신규 진입 금지 종목.

    날짜별 파일이 하나도 없으면 `AlertsNotStarted`, 필요한 날짜(t 포함)의 파일이 없으면 `AlertsNotCollected`를 낸다.
    """
    return _forward_book(t, t).cooldown_at(t)


def engine_alerts(first_signal: str, t: str) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """B의 EngineAlertsFunction. (cooldowns, event_liquidations). 예외는 `cooldown_at`과 같다."""
    return _forward_book(first_signal, t).engine_alerts(first_signal, t)


def daily_alerts(d: str) -> pd.DataFrame:
    """D-06 일간 실행기의 입력. 달력일 d에 이용 가능해진 공시의 경보를 9.3절 형식으로 돌려준다 (DEC-33).

    OpenDART에서 d의 공시를 받고, 앞선 쿨다운 기간의 경보 파일로 같은 사건을 판정한다. 파일은 쓰지 않는다.
    정상 수집 0건이면 0행이다. d 다음 날 08:00 KST 전이거나 수집이 끝나지 않았거나 앞선 기간의 파일이 없으면
    `AlertsNotCollected`를 낸다. 같은 날짜를 다시 만들면 이미 있던 경보는 처음 만든 시각을 유지한다.
    """
    from src.portfolio.params import load_prereg

    day = pd.Timestamp(d).normalize()
    now = _now_kst()
    _require_confirmable(day, now)
    events = load_prereg()["events"]
    days = load_trading_days()
    history = _history_before(day, days, int(events["cooldown_trading_days"]))
    alerts = build_alerts(_collected_disclosures(day, now), io.read_table("tickers"), events, days,
                          history=history).alerts
    alerts["generated_at"] = _now_kst()
    return _keep_first_generated(alerts, day.strftime("%Y-%m-%d"))


def backfill_alert_files(first: str, last: str, *, data_raw: Path | None = None) -> list[str]:
    """D-06 시작 전 달력일의 `alerts/{d}.parquet`를 보관된 공시 원본으로 채운다 (PM 운영용).

    공시 전체 이력으로 판정하므로 같은 사건 처리가 백테스트와 같다. 경보가 없는 날은 0행 파일을 쓰고,
    `generated_at`은 경보 생성이 끝난 시각이다. 수집이 끝나지 않은 날짜가 있거나 `last` 다음 날 08:00 KST 전이면
    파일을 하나도 쓰지 않는다.
    """
    from src.data.disclosures import build_disclosures
    from src.portfolio.params import load_prereg

    start, end = pd.Timestamp(first).normalize(), pd.Timestamp(last).normalize()
    if start > end:
        raise ValueError(f"시작일 {first}이 종료일 {last}보다 늦다")
    now = _now_kst()
    _require_confirmable(end, now)
    raw = Path(data_raw) if data_raw is not None else io.ROOT / "data" / "raw"
    tickers = io.read_table("tickers")
    built = build_disclosures([p.parent for p in raw.glob("*/*_PM_disclosures_*/opendart_list/manifest.json")],
                              today=now.date(), tickers=tickers)
    dates = _calendar_dates(start, end)
    _require_windows_complete(built.windows, dates)
    frame = attach_stock_codes(built.df, built.observations)
    frame = frame[_kst_days(frame["available_at"]) <= end].reset_index(drop=True)
    alerts = build_alerts(frame, tickers, load_prereg()["events"], load_trading_days()).alerts
    alerts["generated_at"] = _now_kst()
    written = []
    for date in dates:
        part = alerts[alerts["date"] == pd.Timestamp(date)].reset_index(drop=True)
        io.write_table("alerts", _keep_first_generated(part, date), date=date)
        written.append(date)
    return written


def _now_kst() -> pd.Timestamp:
    return pd.Timestamp.now(tz=KST)


def _require_confirmable(day: pd.Timestamp, now: pd.Timestamp) -> None:
    if now < day.tz_localize(KST) + CONFIRM_AFTER:
        raise AlertsNotCollected(f"{day:%Y-%m-%d} 경보는 다음 날 08:00 KST부터 확정할 수 있다")


def _history_before(day: pd.Timestamp, days: pd.DatetimeIndex, cooldown: int) -> pd.DataFrame | None:
    """day의 경보가 같은 사건인지 판정할 때 필요한 앞선 달력일의 경보."""
    dates = _forward_dates(day.strftime("%Y-%m-%d"), (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), days, cooldown)
    return _read_dates(dates) if dates else None


def _collected_disclosures(day: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    """day의 일간 창을 OpenDART에서 받고 `available_at`의 KST 날짜가 day인 공시만 돌려준다."""
    from src.data.dart_client import DartClient
    from src.data.disclosures import build_disclosures, collect_daily

    _load_env()
    result = collect_daily(day.date(), client=DartClient(), data_raw=io.ROOT / "data" / "raw")
    if result.status != "complete" or result.windows_by_state.get("provisional"):
        raise AlertsNotCollected(f"{day:%Y-%m-%d} 공시 수집이 끝나지 않았다 "
                                 f"(상태 {result.status}, 사유 {result.stop_reason}, 창 {result.windows_by_state})")
    built = build_disclosures([result.run_dir], today=now.date())
    frame = attach_stock_codes(built.df, built.observations)
    return frame[_kst_days(frame["available_at"]) == day].reset_index(drop=True)


def _load_env() -> None:
    """D-06 실행기로 불려도 저장소의 `.env`에서 OpenDART 키를 읽게 한다. 이미 설정된 환경 변수가 우선이다."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=io.ROOT / ".env", override=False)


def _require_windows_complete(windows: pd.DataFrame, dates: Sequence[str]) -> None:
    """모든 날짜가 완료된 수집 창에 들어 있어야 한다. 빠진 날을 0행으로 쓰면 미수집이 확정 0건이 된다."""
    gaps: set[str] = set(dates) if windows.empty else set()
    if not windows.empty:
        done = windows[windows["state"] == "complete"]
        for kind in DISCLOSURE_TYPES:
            spans = done[done["pblntf_ty"] == kind]
            gaps.update(day for day in dates if not ((spans["bgn"] <= day) & (day <= spans["end"])).any())
    if gaps:
        raise AlertsNotCollected(f"공시 수집이 끝나지 않은 날짜가 {len(gaps)}일 있다 ({min(gaps)} ~ {max(gaps)})")


def _keep_first_generated(alerts: pd.DataFrame, date: str) -> pd.DataFrame:
    """같은 날짜를 다시 만들어도 이미 있던 경보는 처음 만든 시각을 유지한다 (DEC-32)."""
    try:
        existing = io.read_table("alerts", date=date)
    except FileNotFoundError:
        return alerts
    if alerts.empty or existing.empty:
        return alerts
    earlier = existing[["rcept_no", "trigger_type", "generated_at"]].rename(columns={"generated_at": "_first"})
    merged = alerts.merge(earlier, how="left", on=["rcept_no", "trigger_type"])
    merged["generated_at"] = merged["_first"].fillna(merged["generated_at"])
    return merged[ALERT_COLUMNS]


def _kst_days(values: pd.Series) -> pd.Series:
    return values.dt.tz_convert(KST).dt.tz_localize(None).dt.normalize()


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
