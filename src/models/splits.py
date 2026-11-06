"""4.7절 walk-forward 분할과 purge (A-06, A-07 공용).

폴드 Y (Y = 2016~2025): 학습 = 2011년 ~ (Y-3)년 신호일, 검증 = (Y-2), (Y-1)년, 테스트 = Y년.
최종 모델: 학습 2011~2023년, 검증 2024~2025년. 홀드아웃(2026-01-01~)은 purge 경계로만 쓰고 어떤 구간에도 넣지 않는다.
Purge: 다음 구간의 첫 신호일을 s라 할 때, t+1+h 거래일이 s 이상인 신호일 t를 앞 구간에서 뺀다 (라벨이 s 이후 가격을 쓰는 경우).
h는 라벨의 예측 기간이다. G6의 h = 1, 5 실험에서는 그 h로 부르고 signal_dates에 그 주기의 신호일을 넘긴다.
신호일의 연도는 신호일 자체의 달력 연도다.
"""
from dataclasses import dataclass

import pandas as pd

FIRST_TRAIN_YEAR = 2011
TEST_YEARS = tuple(range(2016, 2026))
HOLDOUT_START = pd.Timestamp("2026-01-01")
FINAL_TRAIN_LAST_YEAR = 2023
FINAL_VAL_YEARS = (2024, 2025)


@dataclass(frozen=True)
class Fold:
    test_year: int | None   # 최종 모델 분할은 None
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex


def _positions(days: pd.DatetimeIndex, dates) -> pd.Index:
    pos = days.get_indexer(pd.DatetimeIndex(dates))
    if (pos < 0).any():
        raise ValueError("거래일 캘린더에 없는 날짜가 있다")
    return pos


def purge(dates, next_start, trading_days, h: int) -> pd.DatetimeIndex:
    """t+1+h 거래일이 next_start 이상인 t를 뺀다. dates와 next_start는 모두 거래일이어야 한다(아니면 ValueError)."""
    days = pd.DatetimeIndex(trading_days)
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return dates
    s = _positions(days, [next_start])[0]
    keep = _positions(days, dates) + 1 + h < s
    return dates[keep]


def _years(sig: pd.DatetimeIndex, first: int, last: int) -> pd.DatetimeIndex:
    return sig[(sig.year >= first) & (sig.year <= last)]


def _need(name: str, idx: pd.DatetimeIndex, ctx: str):
    if len(idx) == 0:
        raise ValueError(f"{ctx}: {name} 구간에 신호일이 없다")


def make_folds(signal_dates, trading_days, h: int = 20, test_years=TEST_YEARS,
               first_train_year: int = FIRST_TRAIN_YEAR) -> list:
    """학습 끝과 검증 첫 신호일, 검증 끝과 테스트 첫 신호일 사이를 purge한다. 테스트는 purge하지 않는다."""
    sig = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(signal_dates))))
    days = pd.DatetimeIndex(trading_days)
    folds = []
    for y in test_years:
        ctx = f"폴드 {y}"
        train, val, test = _years(sig, first_train_year, y - 3), _years(sig, y - 2, y - 1), _years(sig, y, y)
        for name, idx in (("학습", train), ("검증", val), ("테스트", test)):
            _need(name, idx, ctx)
        train = purge(train, val[0], days, h)
        val = purge(val, test[0], days, h)
        folds.append(Fold(y, train, val, test))
    return folds


def final_split(signal_dates, trading_days, h: int = 20) -> Fold:
    """최종 모델: 학습 2011~2023, 검증 2024~2025. 학습↔검증, 검증↔홀드아웃(2026-01-01 이후 첫 신호일)을 purge한다."""
    sig = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(signal_dates))))
    days = pd.DatetimeIndex(trading_days)
    train = _years(sig, FIRST_TRAIN_YEAR, FINAL_TRAIN_LAST_YEAR)
    val = _years(sig, FINAL_VAL_YEARS[0], FINAL_VAL_YEARS[1])
    holdout = sig[sig >= HOLDOUT_START]
    for name, idx in (("학습", train), ("검증", val), ("홀드아웃 경계", holdout)):
        _need(name, idx, "최종 분할")
    return Fold(None, purge(train, val[0], days, h), purge(val, holdout[0], days, h), pd.DatetimeIndex([]))
