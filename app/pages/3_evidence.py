"""근거 페이지: 종목을 고르면 evidence의 요약과 출처 링크를 보여준다 (6.11절)."""

import streamlit as st

import dashboard as ui
from src.common import io

SIGNAL_LABELS = {"crash": "진입 금지 경보", "rank": "진입 후보", "alert": "주중 경보"}

ui.start_page("근거")

dates = io.list_dates("evidence")
records = ui.read_evidence(dates[-1]) if dates else []
if not records:
    st.info("표시할 근거 요약이 없습니다.")
else:
    st.caption(f"근거 기준일: {dates[-1]}")
    tickers = sorted({rec["ticker"] for rec in records})
    ticker = st.selectbox("종목코드", tickers)
    for rec in records:
        if rec["ticker"] != ticker:
            continue
        st.subheader(SIGNAL_LABELS[rec["signal_type"]])
        st.write(rec["summary"])
        for src in rec["sources"]:
            url = src["url"] or ""
            label = f"[{src['n']}] {src['title']}"
            if url.startswith(("http://", "https://")):
                st.link_button(label, url)
            else:
                st.write(label)

ui.end_page()
