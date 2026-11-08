"""거래일과 신호일 (4.1절). 거래일 원천은 코스피 200 지수 종가가 존재하는 날이다 (부록 3 Q-14)."""
import pandas as pd


def signal_dates(trading_days) -> list:
    """각 주(월~일)의 마지막 거래일."""
    s = pd.Series(pd.to_datetime(list(trading_days))).sort_values()
    iso = s.dt.isocalendar()
    return list(s.groupby([iso["year"], iso["week"]]).max().sort_values().dt.normalize())
