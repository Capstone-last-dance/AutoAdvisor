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
class CrossFit:
    """DEC-08 연도별 교차 적합 한 건. `year`를 뺀 나머지 학습 연도로 적합한 모델로 `year`를 예측한다."""

    test_year: int | None        # 폴드의 시험 연도. run_id에 들어간다 (DEC-08)
    year: int                    # 예측할 연도 (폴드 학습 구간 안의 한 해)
    train: pd.DatetimeIndex      # year를 빼고 양쪽을 purge한 학습 신호일
    predict: pd.DatetimeIndex    # year의 신호일


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


def purge_after(dates, prev_end, trading_days, h: int) -> pd.DatetimeIndex:
    """prev_end의 라벨이 끝나기 전에 시작하는 t를 뺀다. purge()의 반대 방향이다.

    purge()는 t의 라벨이 뒤 구간을 넘보는 경우를 막고, 이 함수는 앞 구간의 라벨이 t를 넘보는 경우를 막는다.
    가운데 구간을 빼고 적합하는 DEC-08 교차 적합에서는 두 방향이 모두 필요하다.
    """
    days = pd.DatetimeIndex(trading_days)
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return dates
    e = _positions(days, [prev_end])[0]
    keep = _positions(days, dates) > e + 1 + h
    return dates[keep]


def cross_fits(fold: Fold, trading_days, h: int = 20) -> list:
    """폴드 학습 구간을 연도로 나눠, 각 연도를 뺀 나머지로 적합할 분할을 만든다 (DEC-08).

    빠진 연도 y의 양쪽을 모두 purge한다. 앞쪽 학습 연도는 라벨이 y의 첫 신호일 전에 끝나야 하고(purge),
    뒤쪽 학습 연도는 y의 마지막 신호일 라벨이 끝난 뒤에 시작해야 한다(purge_after). 둘 다 실제 라벨
    종료일(t+1+h 거래일) 기준이다.

    하이퍼파라미터는 폴드에서 고른 값을 그대로 쓰므로 여기에는 검증 구간이 없다. 폴드 Y의 검증 연도와
    시험 연도는 폴드 Y 모델의 예측을 쓴다 (DEC-08).
    """
    days = pd.DatetimeIndex(trading_days)
    train = pd.DatetimeIndex(fold.train)
    _need("학습", train, f"폴드 {fold.test_year} 교차 적합")
    out = []
    for y in sorted(set(train.year)):
        predict = train[train.year == y]
        before = purge(train[train.year < y], predict[0], days, h)
        after = purge_after(train[train.year > y], predict[-1], days, h)
        rest = before.append(after)
        if len(rest) == 0:
            raise ValueError(f"폴드 {fold.test_year}의 {y}년: purge 뒤 남는 학습 신호일이 없다")
        out.append(CrossFit(fold.test_year, y, rest, predict))
    return out


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
