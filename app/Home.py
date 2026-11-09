"""대시보드 첫 화면: 개요와 데이터 기준 시각 (6.11절)."""

import streamlit as st

import dashboard as ui

latest = ui.start_page("개인 투자자 종목 진입 판단 보조 시스템")

st.write("주간 신호, 주중 공시 경보, 근거 요약과 성과 기록을 한곳에서 봅니다. 왼쪽 메뉴에서 페이지를 고르세요.")

signals = ui.read("signals", latest) if latest else None
alerts = ui.recent_alerts(20)
forward_log = ui.read("forward_log")

col1, col2, col3 = st.columns(3)
col1.metric("이번 신호일 진입 금지 경보", "-" if signals is None else f"{int((signals['crash_flag'] == 1).sum())}종목")
col2.metric("최근 20거래일 주중 경보", "0건" if alerts is None else f"{len(alerts)}건")
col3.metric("forward test 기록", "0건" if forward_log is None else f"{len(forward_log)}건")

if latest is None:
    st.info("아직 신호 파일이 없습니다.")

ui.end_page()
