"""대시보드 공통 함수 (M10, 명세서 6.11절). D 소유.

모든 페이지가 쓴다. 파일은 src/common/io.py로만 읽고, 쓰지 않는다.
DASHBOARD_DATA_ROOT를 지정하면 그 폴더의 데이터를 보여준다 (샘플 데이터 확인용).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.common import io  # noqa: E402

KST = "Asia/Seoul"
CANDIDATE_COUNT = 20
REPO_ROOT = io.ROOT


def _apply_data_root() -> None:
    override = os.environ.get("DASHBOARD_DATA_ROOT")
    io.ROOT = Path(override) if override else REPO_ROOT


def start_page(title: str) -> str | None:
    """페이지 상단을 그리고 최신 신호일을 돌려준다. 신호가 없으면 None."""
    _apply_data_root()
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    latest = latest_signal_date()
    if latest is None:
        st.caption("최신 신호일: 없음")
    else:
        st.caption(f"최신 신호일: {latest} · 파일 생성 시각: {_kst(io.file_mtime('signals', latest))} KST")
    return latest


def end_page() -> None:
    st.divider()
    st.caption(io.DISCLAIMER)


def latest_signal_date() -> str | None:
    dates = io.list_dates("signals")
    return dates[-1] if dates else None


def read(name: str, date: str | None = None) -> pd.DataFrame | None:
    """테이블을 읽는다. 파일이 없거나 스키마를 어기면 None."""
    mtime = io.file_mtime(name, date)
    if mtime is None:
        return None
    try:
        return _read_table(str(io.ROOT), name, date, mtime)
    except io.SchemaError as exc:
        st.error(f"{name} 파일이 스키마에 맞지 않아 표시하지 않습니다.\n\n{exc}")
        return None


def read_evidence(date: str) -> list[dict]:
    mtime = io.file_mtime("evidence", date)
    if mtime is None:
        return []
    try:
        return _read_evidence(str(io.ROOT), date, mtime)
    except io.SchemaError as exc:
        st.error(f"evidence 파일이 형식에 맞지 않아 표시하지 않습니다.\n\n{exc}")
        return []


# 파일 수정 시각을 인자로 받아, 파일이 바뀌면 다시 읽는다 (6.11절)
@st.cache_data(show_spinner=False)
def _read_table(root: str, name: str, date: str | None, mtime: float) -> pd.DataFrame:
    return io.read_table(name, date=date)


@st.cache_data(show_spinner=False)
def _read_evidence(root: str, date: str, mtime: float) -> list[dict]:
    return io.read_evidence(date)


@st.cache_data(show_spinner=False)
def trigger_names() -> dict[str, str]:
    """부록 1 events.triggers의 {T1: name}."""
    with open(REPO / "configs" / "prereg.yaml", encoding="utf-8") as f:
        triggers = yaml.safe_load(f)["events"]["triggers"]
    return {code: rule["name"] for code, rule in triggers.items()}


@st.cache_data(show_spinner=False)
def cooldown_days() -> int:
    with open(REPO / "configs" / "prereg.yaml", encoding="utf-8") as f:
        return int(yaml.safe_load(f)["events"]["cooldown_trading_days"])


@st.cache_data(show_spinner=False)
def _signal_cutoff_time() -> str:
    with open(REPO / "configs" / "prereg.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["calendar"]["signal_cutoff_time"]


def alert_window(count: int, until: str | None = None) -> list[str] | None:
    """until까지의 최근 count거래일. 거래일은 코스피 200 종가가 있는 날이다 (Q-14). index 파일이 없으면 None.

    until을 주지 않으면 거래일 자료와 경보 파일 중 가장 늦은 날이 기준이다.
    지수가 아직 갱신되지 않은 날의 경보 파일은 그날을 거래일로 본다.
    """
    index = read("index")
    if index is None or index.empty:
        return None
    days = sorted(index["date"].dt.strftime("%Y-%m-%d").unique())
    files = io.list_dates("alerts")
    end = until or max([days[-1], *files[-1:]])
    return ([d for d in days if d <= end] + [d for d in files if days[-1] < d <= end])[-count:]


def alerts_on(days: list[str]) -> pd.DataFrame | None:
    """days의 경보 파일을 합친다. 없으면 None."""
    frames = [df for d in days if (df := read("alerts", d)) is not None and not df.empty]
    return pd.concat(frames, ignore_index=True) if frames else None


def cooldown_tickers(days: list[str], signal_date: str) -> set[str]:
    """days의 경보 중 신호일 컷오프 전에 이용 가능했던 경보의 종목 (available_at < cutoff_at, 6.10절)."""
    alerts = alerts_on(days)
    if alerts is None:
        return set()
    cutoff = pd.Timestamp(f"{signal_date} {_signal_cutoff_time()}", tz=KST)
    return set(alerts.loc[alerts["available_at"] < cutoff, "ticker"])


def entry_candidates(signals: pd.DataFrame, cooldown: set[str]) -> pd.DataFrame:
    """crash_flag = 0이고 M9 쿨다운이 아닌 종목을 rank 오름차순으로 20개 (2.2절)."""
    keep = (signals["crash_flag"] == 0) & ~signals["ticker"].isin(cooldown)
    return signals[keep].sort_values("rank").head(CANDIDATE_COUNT)


def _kst(timestamp: float) -> str:
    return pd.Timestamp(timestamp, unit="s", tz="UTC").tz_convert(KST).strftime("%Y-%m-%d %H:%M")
