import pandas as pd
import pytest

from src.common.guards import purge_signal_dates
from src.data.calendar import signal_dates
from src.models.splits import (HOLDOUT_START, cross_fits, final_split, label_end_dates, make_folds, purge,
                               purge_after)

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


# DEC-08 연도별 교차 적합


def test_cross_fits_cover_every_train_year_and_total_is_75():
    total = 0
    for f in make_folds(SIG, DAYS, h=20):
        cfs = cross_fits(f, DAYS)
        assert [c.year for c in cfs] == sorted(set(pd.DatetimeIndex(f.train).year))
        assert all(c.test_year == f.test_year for c in cfs)
        total += len(cfs)
    assert total == 75                      # DEC-08 "추가 적합은 모델당 75회"


def test_cross_fit_predicts_exactly_the_left_out_year():
    for f in make_folds(SIG, DAYS, h=20):
        tr = pd.DatetimeIndex(f.train)
        for c in cross_fits(f, DAYS):
            assert list(c.predict) == list(tr[tr.year == c.year])
            assert c.year not in set(c.train.year)


@pytest.mark.parametrize("h", [1, 20])
def test_cross_fit_purges_both_sides_of_the_left_out_year(h):
    for f in make_folds(SIG, DAYS, h=h):
        tr = pd.DatetimeIndex(f.train)
        for c in cross_fits(f, DAYS, h=h):
            first, last = _pos(c.predict[0]), _pos(c.predict[-1])
            before, after = tr[tr.year < c.year], tr[tr.year > c.year]
            expected = (list(before[[_pos(d) + 1 + h < first for d in before]])
                        + list(after[[_pos(d) > last + 1 + h for d in after]]))
            assert list(c.train) == expected          # 양쪽 경계가 모두 빠진다
            if h == 20 and len(after):            # 주 1회 신호일 + 라벨 20거래일이면 뒤쪽 4개가 겹친다
                assert len(after) - len([d for d in after if _pos(d) > last + 1 + h]) == 4


def test_purge_after_is_the_mirror_of_purge():
    e = SIG[100]
    kept = purge_after(SIG, [e], DAYS, h=20)
    assert all(_pos(d) > _pos(e) + 21 for d in kept)
    assert all(_pos(d) <= _pos(e) + 21 for d in SIG.difference(kept))
    assert len(purge_after(pd.DatetimeIndex([]), [e], DAYS, h=20)) == 0


def test_cross_fit_raises_when_nothing_is_left_after_purge():
    f = make_folds(SIG, DAYS, h=20)[0]        # 폴드 2016: 학습 2011~2013
    with pytest.raises(ValueError, match="남는 학습 신호일이 없다"):
        cross_fits(f, DAYS, h=10_000)


def _d(index):
    return [t.date() for t in pd.DatetimeIndex(index)]


@pytest.mark.parametrize("delay", [0, 7])
def test_purge_matches_c_guard_with_and_without_delayed_label_ends(delay):
    """A의 purge와 C의 guards.purge_signal_dates는 같은 신호일 집합을 내야 한다 (DEC-09 (3) 규약)."""
    for f in make_folds(SIG, DAYS, h=20):
        train = SIG[(SIG.year >= 2011) & (SIG.year <= f.test_year - 3)]
        ends = {}
        if delay:   # 학습 구간 끝 10일의 라벨이 거래정지로 늦게 끝난 경우
            ends = {t: DAYS[_pos(t) + 1 + 20 + delay] for t in train[-10:]}
        mine = purge(train, f.val[0], DAYS, 20, ends)
        theirs = purge_signal_dates(
            _d(train), trading_days=_d(DAYS), h=20, next_segment_first_signal=f.val[0].date(),
            actual_end_dates={k.date(): v.date() for k, v in ends.items()} or None)
        assert _d(mine) == list(theirs)
    assert delay == 0 or len(mine) < len(purge(train, f.val[0], DAYS, 20))   # 밀리면 더 빠진다


def test_cross_fit_purges_more_when_label_ends_are_delayed():
    f = make_folds(SIG, DAYS, h=20)[-1]                    # 폴드 2025, 학습 2011~2022
    base = {c.year: c for c in cross_fits(f, DAYS, h=20)}
    tr = pd.DatetimeIndex(f.train)
    year = 2016
    ends = {t: DAYS[_pos(t) + 1 + 20 + 15] for t in tr[tr.year.isin([year - 1, year])]}
    slow = {c.year: c for c in cross_fits(f, DAYS, h=20, actual_end_dates=ends)}
    assert len(slow[year].train) < len(base[year].train)   # 앞뒤 양쪽에서 더 빠진다
    assert len(slow[year].predict) == len(base[year].predict)
    before = lambda c: sum(d.year < year for d in c.train)
    after = lambda c: sum(d.year > year for d in c.train)
    assert before(slow[year]) < before(base[year]) and after(slow[year]) < after(base[year])


def test_label_end_dates_takes_the_latest_end_per_signal_date():
    """C의 end_dates(date, ticker, h, nominal_end, actual_end)를 신호일별 최종 종료일로 줄인다."""
    frame = pd.DataFrame({
        "date": ["2016-01-08", "2016-01-08", "2016-01-08", "2016-01-15"],
        "ticker": ["000001", "000002", "000003", "000001"],
        "h": [20, 20, 5, 20],
        "nominal_end": ["2016-02-05", "2016-02-05", "2016-01-15", "2016-02-12"],
        "actual_end": ["2016-02-05", "2016-03-02", "2016-01-15", "2016-02-12"],   # 000002만 재개가 늦었다
    })
    ends = label_end_dates(frame, h=20)
    assert ends == {pd.Timestamp("2016-01-08"): pd.Timestamp("2016-03-02"),       # h = 5 행은 안 본다
                    pd.Timestamp("2016-01-15"): pd.Timestamp("2016-02-12")}
    with pytest.raises(ValueError, match="h = 1 행이 없다"):
        label_end_dates(frame, h=1)
    with pytest.raises(ValueError, match="없는 열"):
        label_end_dates(frame.drop(columns=["actual_end"]), h=20)
