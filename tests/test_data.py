import numpy as np
import pandas as pd
import pytest

from src.data.adjust import adjusted_close
from src.data.calendar import signal_dates
from src.data.security_type import classify
from src.data.universe import avg_trading_value, universe_for_date


def _panel(rows):
    return pd.DataFrame(rows, columns=["date", "ticker", "close_raw", "ret_pct"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def test_adjusted_close_split_50_to_1():
    # 2650000 -> 액면분할 50:1 -> 기준가 53000, 종가 51900 (등락률 -2.0755%, 소수 둘째 자리 반올림 -2.08)
    p = _panel([
        ("2018-05-02", "005930", 2_650_000, 0.0),
        ("2018-05-03", "005930", 2_650_000, 0.0),
        ("2018-05-04", "005930", 51_900, -2.08),
        ("2018-05-08", "005930", 52_600, 1.35),
    ])
    adj = adjusted_close(p)
    assert adj.iloc[-1] == 52_600 and adj.iloc[2] == 51_900          # 이벤트 당일과 이후는 원래 값
    assert adj.iloc[1] == pytest.approx(53_000, rel=1e-3)             # 직전 종가가 기준가로 환산된다
    assert adj.iloc[0] == pytest.approx(53_000, rel=1e-3)


def test_adjusted_close_ignores_rounding_noise_and_cash_dividend():
    # 종가 10000 -> 10050, 등락률 0.5 (기준가=전일종가). 반올림 잡음과 배당락(조정 없음)은 이벤트가 아니다
    p = _panel([("2020-01-02", "000001", 10_000, 0.0), ("2020-01-03", "000001", 10_050, 0.5), ("2020-01-06", "000001", 9_900, -1.49)])
    assert (adjusted_close(p) == p["close_raw"]).all()


def test_adjusted_close_multiple_tickers_do_not_mix():
    p = _panel([
        ("2020-01-02", "A", 1000, 0.0), ("2020-01-03", "A", 500, 0.0),    # A: 2:1 분할 (기준가 500 = 종가/(1+0) → 이벤트)
        ("2020-01-02", "B", 2000, 0.0), ("2020-01-03", "B", 2020, 1.0),
    ])
    adj = adjusted_close(p)
    assert adj.tolist() == [500.0, 500.0, 2000.0, 2020.0]
    assert adj.index.tolist() == p.index.tolist()


def test_adjusted_close_first_row_and_new_listing():
    p = _panel([("2020-01-02", "N", 8000, 100.0), ("2020-01-03", "N", 8100, 1.25)])  # 신규상장 첫날은 비교 대상이 없다
    assert adjusted_close(p).tolist() == [8000.0, 8100.0]


def test_signal_dates_last_trading_day_of_week():
    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"])
    assert [d.strftime("%Y-%m-%d") for d in signal_dates(days)] == ["2024-01-05", "2024-01-11"]
    # 금요일 휴장 주: 그 주의 마지막 거래일
    days = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-15"])
    assert [d.strftime("%Y-%m-%d") for d in signal_dates(days)] == ["2024-01-11", "2024-01-15"]


def test_signal_dates_across_year_boundary():
    days = pd.to_datetime(["2020-12-29", "2020-12-30", "2021-01-04"])   # 2020-12-29/30은 ISO 2020년 53주, 2021-01-04는 2021년 1주
    assert [d.strftime("%Y-%m-%d") for d in signal_dates(days)] == ["2020-12-30", "2021-01-04"]


@pytest.mark.parametrize("group,kind,name,expected", [
    ("주권", "보통주", "삼성전자", ("common", "krx_field")),
    ("주권", "구형우선주", "삼성전자우", ("preferred", "krx_field")),
    ("주권", "신형우선주", "미래에셋증권2우B", ("preferred", "krx_field")),
    ("주권", "종류주권", "무슨종류", ("preferred", "krx_field")),
    ("주권", None, "한일시멘트우", ("preferred", "name_rule")),
    ("주권", float("nan"), "대한전선우", ("preferred", "name_rule")),
    ("주권", None, "경성전기", ("common", "name_rule")),
    ("주권", None, "현대차2우B", ("preferred", "name_rule")),
    ("부동산투자회사", "보통주", "롯데리츠", ("reit", "krx_field")),
    ("사회간접자본투융자회사", None, "맥쿼리인프라", ("infra_fund", "krx_field")),
    ("외국주권", "보통주", "외국회사", ("foreign_stock", "krx_field")),
])
def test_classify(group, kind, name, expected):
    assert classify(group, kind, name) == expected


def _day(rows):
    return pd.DataFrame(rows, columns=["ticker", "name", "security_type", "listed_days", "is_admin", "is_halted", "mcap"])


def test_universe_steps_and_rank():
    day = _day([
        ("000001", "대형", "common", 500, False, False, 1000),
        ("000002", "우선주", "preferred", 500, False, False, 900),       # 1단계 탈락
        ("000003", "OO스팩1호", "common", 500, False, False, 800),        # 1단계(이름) 탈락
        ("000004", "신규", "common", 59, False, False, 700),             # 2단계: 60일 미만
        ("000005", "관리", "common", 500, True, False, 600),             # 3단계
        ("000006", "정지", "common", 500, False, True, 500),             # 3단계
        ("000007", "중형", "common", 500, False, False, 400),
        ("000008", "저유동", "common", 500, False, False, 300),          # 5단계: 거래대금 미달
        ("000009", "경계", "common", 60, False, False, 200),             # 정확히 60일은 통과
    ])
    avg = pd.Series({"000001": 5e9, "000007": 1e9, "000008": 9.99e8, "000009": 2e9})
    u = universe_for_date(day, avg)
    assert u["ticker"].tolist() == ["000001", "000007", "000008", "000009"]
    assert u["mcap_rank"].tolist() == [1, 2, 3, 4]                       # 순위는 1~3단계 통과 종목 기준
    assert u["in_universe"].tolist() == [True, True, False, True]        # 하한 1e9는 포함, 미달은 제외


def test_universe_top_500_cutoff_before_liquidity_filter():
    n = 600
    day = _day([(f"{i:06d}", "x", "common", 500, False, False, 10_000 - i) for i in range(n)])
    avg = pd.Series({f"{i:06d}": 2e9 for i in range(n)})
    avg[f"{499:06d}"] = 0                                                # 500위가 거래대금 미달이어도 501위가 올라오지 않는다
    u = universe_for_date(day, avg)
    assert len(u) == 500 and u["in_universe"].sum() == 499 and u["mcap_rank"].max() == 500


def test_avg_trading_value_requires_full_window_and_counts_missing_as_zero():
    idx = pd.date_range("2024-01-01", periods=25, freq="B")
    v = pd.DataFrame({"A": np.full(25, 2e9), "B": np.r_[np.full(10, 2e9), np.full(15, np.nan)]}, index=idx)
    a = avg_trading_value(v)
    assert a["A"].iloc[18] != a["A"].iloc[18]                            # 19일째까지는 결측
    assert a["A"].iloc[19] == 2e9
    assert a["B"].iloc[24] == pytest.approx(2e9 * 5 / 20)                # 마지막 20일 중 거래 5일 = 평균 5억
