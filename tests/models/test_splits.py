import pandas as pd
import pytest

from src.data.calendar import signal_dates
from src.models.splits import HOLDOUT_START, final_split, make_folds, purge

DAYS = pd.bdate_range("2010-01-04", "2026-10-30")          # 휴장일 없이 평일 전부를 거래일로 본 합성 캘린더
SIG = pd.DatetimeIndex(signal_dates(DAYS))


def _pos(d):
    return DAYS.get_loc(d)


def test_ten_folds_with_expected_year_ranges():
    folds = make_folds(SIG, DAYS, h=20)
    assert [f.test_year for f in folds] == list(range(2016, 2026))
    for f in folds:
        y = f.test_year
        assert f.train.year.min() == 2011 and f.train.year.max() <= y - 3
        assert set(f.val.year) <= {y - 2, y - 1} and set(f.test.year) == {y}
        assert f.train.max() < f.val.min() and f.val.max() < f.test.min()
    assert folds[0].train.min() == SIG[SIG.year == 2011][0]            # 2011년 첫 신호일
    assert folds[0].val.min() == SIG[SIG.year == 2014][0]


@pytest.mark.parametrize("h", [1, 5, 20])
def test_purge_removes_exactly_the_leaking_signal_dates(h):
    for f in make_folds(SIG, DAYS, h=h):
        for before, after, kept_before in ((f.train, f.val, SIG[(SIG.year >= 2011) & (SIG.year <= f.test_year - 3)]),
                                            (f.val, f.test, SIG[(SIG.year >= f.test_year - 2) & (SIG.year <= f.test_year - 1)])):
            nxt = after.min()
            assert all(_pos(t) + 1 + h < _pos(nxt) for t in before)            # 남은 신호일은 다음 구간 가격을 쓰지 않는다
            removed = kept_before.difference(before)
            assert all(_pos(t) + 1 + h >= _pos(nxt) for t in removed)           # 뺀 신호일은 모두 누수 대상이다 (과도한 purge 없음)
            assert len(removed) >= (1 if h >= 5 else 0)                        # h=1은 주간 간격(5거래일)보다 짧아 뺄 것이 없다


def test_purge_counts_for_weekly_signals():
    # 주간 신호일 간격은 5거래일이고 경계 직전 신호일을 L, 다음 구간 첫 신호일을 s = L+5라 하자.
    # t+1+h >= s 인 t는 offset(t-L) >= 4-h 인 것: h=1이면 없음, h=5면 L 하나, h=20이면 L, L-5, L-10, L-15
    full = SIG[(SIG.year >= 2011) & (SIG.year <= 2013)]
    removed = {h: len(full) - len(make_folds(SIG, DAYS, h=h)[0].train) for h in (1, 5, 20)}
    assert removed == {1: 0, 5: 1, 20: 4}


def test_test_period_is_not_purged():
    for f in make_folds(SIG, DAYS, h=20):
        assert len(f.test) == (SIG.year == f.test_year).sum()


def test_nothing_after_2025_in_any_fold_and_final_split_excludes_holdout():
    for f in make_folds(SIG, DAYS, h=20):
        assert f.test.max() <= pd.Timestamp("2025-12-31")
    fin = final_split(SIG, DAYS, h=20)
    assert fin.train.year.min() == 2011 and fin.train.year.max() <= 2023
    assert set(fin.val.year) <= {2024, 2025} and fin.val.max() < HOLDOUT_START and len(fin.test) == 0
    h0 = SIG[SIG >= HOLDOUT_START][0]
    assert all(_pos(t) + 1 + 20 < _pos(h0) for t in fin.val)                   # 검증 ↔ 홀드아웃 purge
    assert all(_pos(t) + 1 + 20 < _pos(fin.val.min()) for t in fin.train)      # 학습 ↔ 검증 purge


def test_purge_ignores_dates_beyond_calendar_end_and_rejects_unknown_dates():
    days = DAYS[:100]
    assert list(purge([days[-1]], days[-1], days, h=1)) == []                  # 자기 자신이 경계면 뺀다
    with pytest.raises(ValueError, match="캘린더에 없는"):
        purge([pd.Timestamp("2010-01-02")], days[10], days, h=1)               # 토요일


def test_missing_period_raises():
    with pytest.raises(ValueError, match="신호일이 없다"):
        make_folds(SIG[SIG.year >= 2014], DAYS, h=20)                          # 2016 폴드의 학습(2011~2013) 구간이 비어 있다
    with pytest.raises(ValueError, match="홀드아웃 경계"):
        final_split(SIG[SIG < HOLDOUT_START], DAYS, h=20)


def test_h_5_and_20_signal_cadence_from_caller():
    sig20 = DAYS[DAYS >= "2011-01-03"][::20]                                    # G6: 20거래일 간격 신호일을 호출자가 만든다
    f = make_folds(sig20, DAYS, h=20)[0]
    assert all(_pos(t) + 21 < _pos(f.val.min()) for t in f.train)
