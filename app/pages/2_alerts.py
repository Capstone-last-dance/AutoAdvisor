"""주중 경보 페이지: 최근 20거래일 alerts (6.11절)."""

import pandas as pd
import streamlit as st

import dashboard as ui

ui.start_page("주중 경보")

window = ui.alert_window(20)
ready = window is not None and window.complete
alerts = ui.alerts_on(window.files) if ready else None
if window is not None:
    st.caption(f"경보 기준일: {window.base} (확정된 거래일 기준)")
    if window.later:
        st.warning(f"거래일 자료가 {window.base}까지만 있어 이후 경보 파일 {window.later}개는 아직 표시하지 않습니다.")
if not ready:
    st.warning("확정된 거래일 자료(index)가 부족해 경보를 표시하지 않습니다.")
elif alerts is None:
    st.info("최근 20거래일 경보가 없습니다.")
else:
    triggers = ui.trigger_names()
    alerts = alerts.sort_values(["date", "available_at"], ascending=False)
    st.dataframe(pd.DataFrame({
        "날짜": alerts["date"].dt.strftime("%Y-%m-%d"),
        "종목코드": alerts["ticker"],
        "경보 유형": alerts["trigger_type"].map(lambda t: triggers.get(t, t)),
        "공시명": alerts["report_nm"],
        "접수번호": alerts["rcept_no"],
        "이용 가능 시각": alerts["available_at"].dt.strftime("%Y-%m-%d %H:%M"),
    }), hide_index=True)

ui.end_page()
