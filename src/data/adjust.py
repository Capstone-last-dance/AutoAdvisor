"""수정종가 계산 (부록 3 Q-02, Q-05).

KRX 날짜별 시세의 등락률은 '조정된 기준가' 대비라서, 기준가 = 종가 / (1 + 등락률)이다.
기준가가 전일 종가와 다른 날이 분할·병합·권리락·현금배당 같은 조정 이벤트이고, 그 이후 가격의 비율을 과거 종가에 곱한다.
**현금배당도 기준가에 반영된다** (2026-11-03 A-03 전종목 검증으로 정정: 12월 말 배당락 시기 조정 이벤트 806건의
기준가/전일종가 비율 중앙값이 0.958로, 전형적인 배당수익률 범위와 일치한다. 삼성전자 1종목만 본 이전 결론은 틀렸다 -
docs/appendix3_A_results.md Q-05 참고).
"""
import numpy as np
import pandas as pd

EVENT_THRESHOLD = 1e-3  # 기준가/전일종가가 0.1% 넘게 다르면 이벤트. 등락률 반올림(소수 둘째 자리) 잡음은 약 5e-5


def _base_ratio(df: pd.DataFrame):
    """정렬된 d, 종가, 기준가, 전일 종가, 기준가/전일종가. 종가·기준가·전일 종가가 0 이하이거나 무한대인 행은 비율을 계산하지 않는다(결측).
    등락률 -100%나 종가 0 같은 이상 행 하나가 과거 전체 수정종가를 0이나 무한대로 만드는 것을 막는다."""
    d = df.sort_values(["ticker", "date"])
    close = d["close_raw"].astype("float64")
    base = close / (1 + d["ret_pct"].astype("float64") / 100)
    prev = close.groupby(d["ticker"]).shift(1)
    ok = (prev > 0) & (close > 0) & (base > 0) & np.isfinite(base)
    return d, close, base, prev, (base / prev).where(ok)


def adjusted_close(df: pd.DataFrame) -> pd.Series:
    """df: date, ticker, close_raw, ret_pct(등락률, %) 열을 가진 long 형식. 반환은 df와 같은 인덱스의 수정종가."""
    d, close, _, _, ratio = _base_ratio(df)
    ratio = ratio.where((ratio - 1).abs() > EVENT_THRESHOLD, 1.0).fillna(1.0)
    # 시점 t 이후에 일어난 이벤트 비율의 누적곱 (t 당일 이벤트는 t의 종가에 이미 반영돼 있으므로 제외)
    rev = ratio.iloc[::-1]
    cum = rev.groupby(d["ticker"].iloc[::-1]).cumprod().iloc[::-1]
    factor = cum.groupby(d["ticker"]).shift(-1).fillna(1.0)
    return (close * factor).reindex(df.index)


def adjust_events(df: pd.DataFrame) -> pd.DataFrame:
    """조정 이벤트 목록 (ticker, date, prev_close, base_price, ratio). ratio = 기준가 / 전일 종가."""
    d, _, base, prev, ratio = _base_ratio(df)
    ev = (ratio - 1).abs() > EVENT_THRESHOLD
    return pd.DataFrame({"ticker": d.loc[ev, "ticker"], "date": d.loc[ev, "date"], "prev_close": prev[ev],
                         "base_price": base[ev], "ratio": ratio[ev]}).reset_index(drop=True)
