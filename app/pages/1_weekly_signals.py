"""이번 주 신호 페이지 (6.11절)."""

import pandas as pd
import streamlit as st

import dashboard as ui

latest = ui.start_page("이번 주 신호")

if latest is None:
    st.info("아직 신호 파일이 없습니다.")
else:
    signals = ui.read("signals", latest)

    if signals is not None:
        st.subheader("진입 금지 경보")
        flagged = signals[signals["crash_flag"] == 1].sort_values("crash_prob", ascending=False)
        if flagged.empty:
            st.write("이번 신호일에는 진입 금지 경보가 없습니다.")
        else:
            st.dataframe(pd.DataFrame({
                "종목코드": flagged["ticker"],
                "crash_prob": flagged["crash_prob"],
            }), hide_index=True)

        st.subheader("진입 후보")
        days = ui.cooldown_days()
        window = ui.alert_window(days, until=latest)
        if window is None:
            st.warning("거래일 자료(index)가 없어 진입 후보를 계산하지 않습니다.")
        else:
            candidates = ui.entry_candidates(signals, ui.cooldown_tickers(window, latest))
            st.caption(f"진입 금지 경보 종목과, 신호 컷오프 전에 공개된 최근 {days}거래일 주중 경보 종목을 "
                       f"뺀 상위 {ui.CANDIDATE_COUNT}종목입니다.")
            st.dataframe(pd.DataFrame({
                "rank": candidates["rank"],
                "종목코드": candidates["ticker"],
                "rank_score": candidates["rank_score"],
            }), hide_index=True)

    st.subheader("모델 포트폴리오")
    positions = ui.read("positions", latest)
    if positions is None or positions.empty:
        st.info("이번 신호일의 모델 포트폴리오는 아직 없습니다.")
    else:
        stocks = positions[positions["ticker"] != "CASH"].sort_values("target_weight", ascending=False)
        col1, col2, col3 = st.columns(3)
        col1.metric("현금 비중", f"{1 - stocks['target_weight'].sum():.1%}")
        col2.metric("E_t", f"{positions['exposure'].iloc[0]:.2f}")
        col3.metric("regime_mc", int(positions["regime_mc"].iloc[0]))
        st.dataframe(pd.DataFrame({
            "종목코드": stocks["ticker"],
            "목표 비중": stocks["target_weight"],
        }), hide_index=True)

ui.end_page()
