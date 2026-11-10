"""대시보드 공통 함수 (M10, 명세서 6.11절). D 소유.

모든 페이지가 쓴다. 파일은 src/common/io.py로만 읽고, 쓰지 않는다.
DASHBOARD_DATA_ROOT를 지정하면 그 폴더의 데이터를 보여준다 (샘플 데이터 확인용).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import streamlit as st
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.common import io  # noqa: E402
from src.events import alerts as m9  # noqa: E402

KST = "Asia/Seoul"
CANDIDATE_COUNT = 20
REPO_ROOT = io.ROOT


class AlertWindow(NamedTuple):
    base: str          # 기간 계산에 쓴 마지막 확정 거래일
    files: list[str]   # 기간 안의 경보 파일 날짜 (비거래일 포함)
    complete: bool     # 요청한 날까지 count거래일을 확정할 수 있으면 True
    later: int         # base 뒤의 경보 파일 수


def _apply_data_root() -> bool:
    """DASHBOARD_DATA_ROOT를 적용한다. 경로가 스텁 폴더 또는 그 하위면 적용하지 않고 False (6.11절)."""
    override = os.environ.get("DASHBOARD_DATA_ROOT")
    if override and any(part.lower() == io.STUB_DIR for part in Path(override).resolve().parts):
        return False
    io.ROOT = Path(override) if override else REPO_ROOT
    return True


def start_page(title: str) -> str | None:
    """페이지 상단을 그리고 최신 신호일을 돌려준다. 신호가 없으면 None."""
    allowed = _apply_data_root()
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)
    if not allowed:
        st.error("스텁 데이터는 대시보드에 표시하지 않습니다. DASHBOARD_DATA_ROOT를 확인하세요.")
        end_page()
        st.stop()
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
    """테이블을 읽는다. 파일이 없거나 읽기·스키마 검사에 실패하면 None (실패는 화면에 표시한다)."""
    mtime = io.file_mtime(name, date)
    if mtime is None:
        return None
    try:
        return _read_table(str(io.ROOT), name, date, mtime)
    except Exception as exc:  # 부분 파일 등 어떤 읽기 오류든 페이지를 멈추지 않는다
        st.error(f"{name} 파일을 읽지 못해 표시하지 않습니다.\n\n{exc}")
        return None


def read_evidence(date: str) -> list[dict]:
    mtime = io.file_mtime("evidence", date)
    if mtime is None:
        return []
    try:
        return _read_evidence(str(io.ROOT), date, mtime)
    except Exception as exc:  # 부분 파일 등 어떤 읽기 오류든 페이지를 멈추지 않는다
        st.error(f"evidence 파일을 읽지 못해 표시하지 않습니다.\n\n{exc}")
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


def alert_window(count: int, until: str | None = None) -> AlertWindow | None:
    """until까지 최근 count거래일 범위에 든 경보 파일을 찾는다. index를 읽지 못하면 None.

    거래일은 코스피 200 종가가 있는 날이다 (Q-14). 확정된 거래일로만 세고 추정하지 않는다.
    until을 주지 않으면 마지막 확정 거래일이 기준이다.
    """
    index = read("index")
    if index is None or index.empty:
        return None
    days = sorted(index["date"].dt.strftime("%Y-%m-%d").unique())
    files = io.list_dates("alerts")
    end = until or days[-1]
    known = [d for d in days if d <= end]
    base = known[-1] if known else days[0]
    later = sum(d > base for d in files)
    if end > days[-1] or len(known) < count:
        return AlertWindow(base, [], False, later)
    return AlertWindow(base, [d for d in files if known[-count] <= d <= end], True, later)


def alerts_on(days: list[str]) -> pd.DataFrame | None:
    """days의 경보 파일을 합친다. 없으면 None."""
    frames = [df for d in days if (df := read("alerts", d)) is not None and not df.empty]
    return pd.concat(frames, ignore_index=True) if frames else None


def m9_cooldown(signal_date: str) -> set[str] | None:
    """M9 `cooldown_at`: 신호일 t의 다음 거래일에 신규 진입이 금지된 종목 (6.10절).

    경보 파일이 없거나(M9 미실행) 읽지 못하면 오류를 표시하고 None을 돌려준다.
    """
    try:
        return m9.cooldown_at(signal_date)
    except FileNotFoundError:
        st.error("경보 파일이 아직 없습니다. M9 경보 생성을 먼저 실행해야 합니다.")
    except Exception as exc:  # 부분 파일 등 어떤 오류든 페이지를 멈추지 않는다
        st.error(f"M9 쿨다운을 계산하지 못했습니다.\n\n{exc}")
    return None


def entry_candidates(signals: pd.DataFrame, cooldown: set[str]) -> pd.DataFrame:
    """crash_flag = 0이고 M9 쿨다운이 아닌 종목을 rank 오름차순으로 20개 (2.2절)."""
    keep = (signals["crash_flag"] == 0) & ~signals["ticker"].isin(cooldown)
    return signals[keep].sort_values("rank").head(CANDIDATE_COUNT)


def _kst(timestamp: float) -> str:
    return pd.Timestamp(timestamp, unit="s", tz="UTC").tz_convert(KST).strftime("%Y-%m-%d %H:%M")
