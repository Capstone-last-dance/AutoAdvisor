"""성과 페이지. reports/final/의 확정 결과만 쓴다 (6.11절, D-09).

KPI 판정 표(kpi.csv), NAV 그림(nav.csv), 분석 구간 표(periods.csv)를 보여준다. 형식은 io.FINAL_TABLES.
파일이 하나도 없으면 '확정된 결과 없음'만 표시한다.
"""

import logging

import pandas as pd
import streamlit as st

import dashboard as ui
from src.common import io

_log = logging.getLogger("dashboard")


@st.cache_data(show_spinner=False)
def _read_final(root: str, name: str, mtime: float) -> pd.DataFrame:
    return io.read_final(name)


def read_final(name: str) -> pd.DataFrame | None:
    """확정 결과 표. 파일이 없거나 읽지 못하면 None (읽기 실패는 화면에 표시한다)."""
    mtime = io.final_mtime(name)
    if mtime is None:
        return None
    try:
        return _read_final(str(io.ROOT), name, mtime)
    except Exception as exc:  # 형식이 틀린 파일이 있어도 페이지를 멈추지 않는다
        _log.exception("reports/final/%s.csv 읽기 실패", name)
        st.error(f"{name}.csv를 읽지 못해 표시하지 않습니다 ({type(exc).__name__}).")
        return None


ui.start_page("성과")

files = io.list_final_reports()
if not files:
    st.info("확정된 결과 없음")
else:
    st.caption("C가 확정한 reports/final/의 결과만 보여줍니다. 실험 중인 결과는 표시하지 않습니다.")

    st.subheader("KPI 판정 (G0~G7)")
    kpi = read_final("kpi")
    if kpi is None:
        st.write("KPI 판정 표가 아직 없습니다.")
    else:
        st.dataframe(pd.DataFrame({
            "게이트": kpi["gate"], "지표": kpi["kpi"], "값": kpi["value"],
            "기준": kpi["threshold"], "판정": kpi["verdict"],
            **({"비고": kpi["note"]} if "note" in kpi.columns else {}),
        }), hide_index=True)

    st.subheader("NAV")
    nav = read_final("nav")
    if nav is None or nav.empty:
        st.write("NAV 자료가 아직 없습니다.")
    else:
        wide = nav.pivot_table(index="date", columns="series", values="nav", aggfunc="last").sort_index()
        st.line_chart(wide)
        st.caption(f"기간: {wide.index[0]:%Y-%m-%d} ~ {wide.index[-1]:%Y-%m-%d}")

    st.subheader("분석 구간")
    periods = read_final("periods")
    if periods is None:
        st.write("분석 구간 표가 아직 없습니다.")
    else:
        shown = periods.copy()
        shown["start"] = shown["start"].dt.strftime("%Y-%m-%d")
        shown["end"] = shown["end"].dt.strftime("%Y-%m-%d")
        st.dataframe(shown.rename(columns={"period": "구간", "start": "시작", "end": "끝", "description": "설명"}), hide_index=True)

    known = {f"{name}.csv" for name in io.FINAL_TABLES}
    others = [name for name in files if name not in known]
    if others:
        with st.expander("그 밖의 확정 결과 파일"):
            for name in others:
                st.write(f"- {name}")

ui.end_page()
