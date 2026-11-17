"""forward test 페이지: forward_log.csv 표 (6.11절)."""

import pandas as pd
import streamlit as st

import dashboard as ui

ui.start_page("forward test")

log = ui.read("forward_log")
if log is None or log.empty:
    st.info("아직 forward test 기록이 없습니다.")
else:
    st.dataframe(pd.DataFrame({
        "신호일": log["signal_date"].dt.strftime("%Y-%m-%d"),
        "커밋 해시": log["commit_hash"],
        "신호 종류": log["signal_source"],
        "비중 기록": log["has_positions"].map({0: "없음", 1: "있음"}),
        "라벨 확정일": log["label_ready_date"].dt.strftime("%Y-%m-%d"),
        "상위 20 평균 ER": log["mean_er_top20"],
        "경보 종목 평균 ER": log["mean_er_flagged"],
        "포트폴리오 수익률": log["portfolio_return"],
        "BM 수익률": log["bm_return"],
    }), hide_index=True)
    st.caption("표본이 적어 성과 판정에는 쓰지 않고, 미래 정보 누수가 없다는 증거로만 씁니다.")

ui.end_page()
