"""주중 경보 페이지: 최근 20거래일 alerts (6.11절)."""

import pandas as pd
import streamlit as st

import dashboard as ui

ui.start_page("주중 경보")

window = ui.alert_window(20)
alerts = None if window is None else ui.alerts_on(window)
if window is None:
    st.warning("거래일 자료(index)가 없어 경보를 표시하지 않습니다.")
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
