"""성과 페이지. reports/final/의 확정 결과만 쓴다 (6.11절). 표와 그림은 D-09에서 완성한다."""

import streamlit as st

import dashboard as ui
from src.common import io

ui.start_page("성과")

files = io.list_final_reports()
if not files:
    st.info("확정된 결과 없음")
else:
    st.write("확정 결과 파일")
    for name in files:
        st.write(f"- {name}")

ui.end_page()
