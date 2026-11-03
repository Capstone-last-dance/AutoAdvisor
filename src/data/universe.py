"""유니버스 U_t (4.4절). 순수 함수로 두고 단위 테스트한다."""
import numpy as np
import pandas as pd

TOP_N = 500
MIN_LISTED_DAYS = 60
MIN_AVG_VALUE = 1_000_000_000
AVG_WINDOW = 20


def avg_trading_value(value_pivot: pd.DataFrame, window: int = AVG_WINDOW) -> pd.DataFrame:
    """value_pivot: 인덱스 거래일, 열 ticker, 값 거래대금. 시세 행이 없는 날은 거래 없음(0)으로 센다."""
    return value_pivot.fillna(0).rolling(window, min_periods=window).mean()


def universe_for_date(day: pd.DataFrame, avg_value: pd.Series) -> pd.DataFrame:
    """day: 신호일 t 한 날의 종목별 행. 열: ticker, name, security_type, listed_days, is_admin, is_halted, mcap.
    반환: date 없이 ticker, mcap_rank, in_universe. 1~3단계를 통과한 종목의 시총 순위가 TOP_N 이내인 종목을 담고,
    5단계(거래대금 하한)를 통과하면 in_universe=True. 그래서 in_universe인 종목은 항상 TOP_N 이하다."""
    for c in ("is_admin", "is_halted"):   # 결측을 False로 단정하지 않는다 (R-08). 호출자가 채우는 값을 정한다
        if day[c].isna().any():
            raise ValueError(f"{c}에 결측이 있다: 호출자가 값을 채워서 넘겨야 한다 (부록 3 Q-06 확인 전에는 관리종목 이력이 비어 있다)")
    ok = (
        (day["security_type"] == "common")
        & ~day["name"].fillna("").str.contains("스팩")
        & (day["listed_days"] >= MIN_LISTED_DAYS)
        & ~day["is_admin"].astype(bool)
        & ~day["is_halted"].astype(bool)
    )
    cand = day[ok].copy()
    cand = cand.sort_values(["mcap", "ticker"], ascending=[False, True])
    cand["mcap_rank"] = np.arange(1, len(cand) + 1, dtype="int32")
    top = cand[cand["mcap_rank"] <= TOP_N].copy()
    top["in_universe"] = top["ticker"].map(avg_value).fillna(0) >= MIN_AVG_VALUE
    return top[["ticker", "mcap_rank", "in_universe"]].reset_index(drop=True)
