"""PM-09 뉴스 피처 테스트 (6.2절, DEC-06, DEC-09 (1)). 합성 자료만 쓴다."""

import numpy as np
import pandas as pd
import pytest

from src.common import io
from src.features.news_features import (NEWS_FEATURES, OUTPUT_COLUMNS, analysis_articles, build_news_features,
                                        compute_news_features, current_name_since, name_change_dates,
                                        rank_news_features)

HOLIDAY = pd.Timestamp("2026-03-02")                                   # 월요일 휴장
DAYS = pd.bdate_range("2025-10-01", "2026-04-30").drop(HOLIDAY)
T = pd.Timestamp("2026-03-09")                                        # 월요일 신호일
COVERAGE = ("2025-09-01", "2026-04-30")
CFG = {"features": {"news_window_offsets": [-5, -1], "news_baseline_offsets": [-65, -6]}}


def day(offset, t=T):
    """신호일 t에서 거래일 offset만큼 떨어진 날."""
    return DAYS[DAYS.get_loc(t) + offset]


def universe(tickers=("000001", "000002"), dates=(T,)):
    return pd.DataFrame([(d, t, True) for d in dates for t in tickers], columns=["date", "ticker", "in_universe"])


def articles(rows):
    """rows: (날짜, 종목, 감성 점수 또는 None)"""
    return pd.DataFrame([(pd.Timestamp(d), t, np.nan if s is None else s) for d, t, s in rows],
                        columns=["date", "ticker", "sentiment"])


def raw(rows, tickers=("000001", "000002"), collected=None, coverage=COVERAGE, covered_from=None, dates=(T,),
        covered_until=None):
    out = compute_news_features(articles(rows), universe(tickers, dates), DAYS,
                                collected if collected is not None else tickers, coverage, covered_from,
                                covered_until=covered_until)
    return out.set_index(["date", "ticker"])


def value(frame, name, ticker="000001", t=T):
    return frame.loc[(t, ticker), name]


class TestWindows:
    def test_signal_day_articles_are_not_used(self):
        base = raw([(day(-1), "000001", 0.5)])
        with_t = raw([(day(-1), "000001", 0.5), (T, "000001", -1.0), (T, "000001", -1.0)])
        pd.testing.assert_frame_equal(base, with_t)

    def test_weekend_before_signal_counts_in_recent_window(self):
        # 월요일 신호일: 토·일 기사는 직전 거래일(금, t−1)에 더한다
        out = raw([("2026-03-07", "000001", 0.2), ("2026-03-08", "000001", 0.4)])
        assert value(out, "news_cnt_spike") == pytest.approx((2 + 1) / (0 + 1))
        assert value(out, "news_sent_5") == pytest.approx(0.3)

    def test_holiday_article_goes_to_previous_trading_day(self):
        # 3/2(월) 휴장일 기사는 2/27(금) 몫이다
        assert day(-5) == pd.Timestamp("2026-02-27")                        # t=3/9: 2/27이 t−5라 최근 창
        out = raw([(HOLIDAY, "000001", 0.9)])
        assert value(out, "news_cnt_spike") == pytest.approx(2.0)
        assert value(out, "news_sent_5") == pytest.approx(0.9)
        later = pd.Timestamp("2026-03-10")                                   # t=3/10: 2/27이 t−6이라 기준 창
        assert day(-5, later) == pd.Timestamp("2026-03-03")
        out = raw([(HOLIDAY, "000001", 0.9)], dates=(later,))
        assert value(out, "news_cnt_spike", t=later) == pytest.approx(1 / (1 * 5 / 60 + 1))
        assert np.isnan(value(out, "news_sent_5", t=later))

    def test_recent_window_start_boundary(self):
        inside = raw([(day(-5), "000001", 0.1)])
        outside = raw([(day(-5) - pd.Timedelta(days=1), "000001", 0.1)])     # 전날 = t−6 거래일 몫
        assert value(inside, "news_cnt_spike") == pytest.approx(2.0)
        assert value(inside, "news_sent_5") == pytest.approx(0.1)
        assert value(outside, "news_cnt_spike") == pytest.approx(1 / (1 * 5 / 60 + 1))
        assert np.isnan(value(outside, "news_sent_5"))

    def test_baseline_start_boundary(self):
        inside = raw([(day(-65), "000001", None)])
        before = raw([(day(-65) - pd.Timedelta(days=1), "000001", None)])
        assert value(inside, "news_cnt_spike") == pytest.approx(1 / (5 / 60 + 1))
        assert value(before, "news_cnt_spike") == pytest.approx(1.0)

    def test_formula(self):
        rows = [(day(-1), "000001", 0.0)] * 3 + [(day(-30), "000001", 0.0)] * 12
        assert value(raw(rows), "news_cnt_spike") == pytest.approx((3 + 1) / (12 * 5 / 60 + 1))

    def test_sentiment_is_mean_of_recent_articles_only(self):
        rows = [(day(-1), "000001", 0.6), (day(-3), "000001", -0.2), (day(-10), "000001", -1.0)]
        assert value(raw(rows), "news_sent_5") == pytest.approx(0.2)

    def test_unscored_article_counts_but_does_not_change_mean(self):
        out = raw([(day(-1), "000001", 0.4), (day(-2), "000001", None)])
        assert value(out, "news_cnt_spike") == pytest.approx(3.0)
        assert value(out, "news_sent_5") == pytest.approx(0.4)

    def test_no_articles_is_zero_count_and_missing_sentiment(self):
        out = raw([])
        assert value(out, "news_cnt_spike") == pytest.approx(1.0)
        assert np.isnan(value(out, "news_sent_5"))


class TestMissing:
    def test_not_collected_ticker_is_missing_not_zero(self):
        out = raw([(day(-1), "000002", 0.5)], collected=["000001"])
        assert np.isnan(value(out, "news_cnt_spike", "000002"))
        assert np.isnan(value(out, "news_sent_5", "000002"))
        assert value(out, "news_cnt_spike", "000001") == pytest.approx(1.0)

    def test_coverage_end_before_last_window_day(self):
        out = raw([(day(-2), "000001", 0.5)], coverage=("2025-09-01", day(-2)))
        assert np.isnan(value(out, "news_cnt_spike"))
        assert np.isnan(value(out, "news_sent_5"))

    def test_coverage_start_inside_baseline_keeps_sentiment(self):
        out = raw([(day(-1), "000001", 0.5)], coverage=(day(-30), "2026-04-30"))
        assert np.isnan(value(out, "news_cnt_spike"))
        assert value(out, "news_sent_5") == pytest.approx(0.5)

    def test_rename_inside_window_is_missing(self):
        # 현재 상호를 쓰기 전 구간은 옛 상호 기사를 세지 못한다
        out = raw([(day(-1), "000001", 0.5)], covered_from={"000001": day(-20)})
        assert np.isnan(value(out, "news_cnt_spike"))
        assert value(out, "news_sent_5") == pytest.approx(0.5)
        later = raw([(day(-1), "000001", 0.5)], covered_from={"000001": day(-3)})
        assert np.isnan(later.loc[(T, "000001"), "news_sent_5"])

    def test_covered_until_inside_window_is_missing(self):
        # 새 상호가 일반어·지명과 겹쳐 그 뒤 건수를 믿을 수 없는 종목 (예: 보령제약 → 보령)
        out = raw([(day(-1), "000001", 0.5)], covered_until={"000001": day(-2)})
        assert np.isnan(value(out, "news_cnt_spike"))
        assert np.isnan(value(out, "news_sent_5"))
        assert value(out, "news_cnt_spike", "000002") == pytest.approx(1.0)

    def test_covered_until_after_windows_does_not_matter(self):
        out = raw([(day(-1), "000001", 0.5)], covered_until={"000001": T})
        assert value(out, "news_cnt_spike") == pytest.approx(2.0)

    def test_rename_before_windows_does_not_matter(self):
        out = raw([(day(-1), "000001", 0.5)], covered_from={"000001": day(-70)})
        assert value(out, "news_cnt_spike") == pytest.approx(2.0)

    def test_rename_after_signal_day_does_not_hide_past_window(self):
        # DEC-35 회귀: 변경일 목록이면 t 뒤 변경은 t의 창을 결측으로 만들지 않는다 (v6는 날짜 하나라 결측이 됐다)
        rows = [(day(-1), "000001", 0.5), (day(9), "000001", 0.3)]
        later = day(10)
        out = raw(rows, covered_from={"000001": [day(3)]}, dates=(T, later))
        assert value(out, "news_cnt_spike") == pytest.approx(2.0)
        assert value(out, "news_sent_5") == pytest.approx(0.5)
        assert np.isnan(value(out, "news_cnt_spike", t=later))           # 기준 창에 변경일이 든다
        assert value(out, "news_sent_5", t=later) == pytest.approx(0.3)  # 최근 창은 변경 뒤다
        fixed = raw(rows, covered_from={"000001": day(3)}, dates=(T, later))
        assert np.isnan(value(fixed, "news_cnt_spike"))                  # 날짜 하나는 모든 신호일에 적용된다

    def test_rename_list_uses_last_change_up_to_signal_day(self):
        out = raw([(day(-1), "000001", 0.5)], covered_from={"000001": [day(3), day(-20)]})
        assert np.isnan(value(out, "news_cnt_spike"))
        assert value(out, "news_sent_5") == pytest.approx(0.5)
        on_signal_day = raw([(day(-1), "000001", 0.5)], covered_from={"000001": [T]})
        assert np.isnan(value(on_signal_day, "news_sent_5"))            # t 당일 변경은 t까지 일어난 변경이다
        empty = raw([(day(-1), "000001", 0.5)], covered_from={"000001": []})
        assert value(empty, "news_cnt_spike") == pytest.approx(2.0)


class TestRowsAndRanks:
    def test_rows_are_exactly_universe(self):
        u = universe(("000001", "000002"), (T, day(5)))
        u.loc[len(u)] = (T, "000003", False)
        out = compute_news_features(articles([(day(-1), "999999", 0.5)]), u, DAYS, ["000001", "000002"], COVERAGE)
        want = u[u["in_universe"]][["date", "ticker"]].sort_values(["date", "ticker"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(out[["date", "ticker"]], want, check_dtype=False)

    def test_rank_transform_and_flags(self):
        tickers = ("000001", "000002", "000003")
        rows = [(day(-1), "000001", 0.9)] * 3 + [(day(-1), "000002", -0.5)]
        out = build_news_features(articles(rows), universe(tickers), DAYS, CFG, ["000001", "000002"], COVERAGE)
        assert list(out.columns) == OUTPUT_COLUMNS
        got = out.set_index("ticker")
        assert got.loc["000001", "news_cnt_spike"] == 1.0 and got.loc["000002", "news_cnt_spike"] == 0.5
        assert got.loc["000003", "news_cnt_spike"] == 0.5 and got.loc["000003", "news_cnt_spike_na"] == 1
        assert got.loc["000001", "news_sent_5"] == 1.0 and got.loc["000002", "news_sent_5"] == 0.5
        assert got.loc["000003", "news_sent_5_na"] == 1 and got.loc["000001", "news_sent_5_na"] == 0
        io.validate("features", out)

    def test_signal_not_trading_day_or_short_history(self):
        with pytest.raises(ValueError, match="거래일 목록에 없다"):
            compute_news_features(articles([]), universe(dates=(pd.Timestamp("2026-03-07"),)), DAYS, ["000001"], COVERAGE)
        with pytest.raises(ValueError, match="거래일이 65개보다 적다"):
            compute_news_features(articles([]), universe(dates=(DAYS[10],)), DAYS, ["000001"], COVERAGE)

    def test_duplicate_universe_rows(self):
        u = pd.concat([universe(), universe()], ignore_index=True)
        with pytest.raises(ValueError, match="두 번"):
            compute_news_features(articles([]), u, DAYS, ["000001"], COVERAGE)


def test_future_articles_do_not_change_past_features():
    # 6.2절 완료 기준 (2)의 축소판: t+1일 이후 기사를 지워도 t의 피처가 같다
    rng = np.random.default_rng(9)
    pool = pd.date_range("2025-10-01", "2026-04-30", freq="D")
    rows = [(pool[i], f"00000{k}", float(rng.uniform(-1, 1))) for i, k in
            zip(rng.integers(0, len(pool), 3000), rng.integers(1, 4, 3000))]
    tickers = ("000001", "000002", "000003")
    dates = tuple(DAYS[(DAYS >= "2026-01-15") & (DAYS <= "2026-04-20")][::7])
    full = build_news_features(articles(rows), universe(tickers, dates), DAYS, CFG, tickers, COVERAGE)
    for t in dates:
        cut = [r for r in rows if r[0] <= t]
        part = build_news_features(articles(cut), universe(tickers, (t,)), DAYS[DAYS <= t], CFG, tickers, COVERAGE)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), full[full["date"] == t].reset_index(drop=True))


def test_future_renames_do_not_change_past_features():
    # DEC-35 (4)의 축소판: 신호일마다 t까지 알 수 있던 변경일만 넣어 다시 계산해도 t의 피처가 같다
    rng = np.random.default_rng(35)
    pool = pd.date_range("2025-10-01", "2026-04-30", freq="D")
    rows = [(pool[i], f"00000{k}", float(rng.uniform(-1, 1))) for i, k in
            zip(rng.integers(0, len(pool), 3000), rng.integers(1, 4, 3000))]
    tickers = ("000001", "000002", "000003")
    renames = {"000001": [pd.Timestamp("2026-02-10"), pd.Timestamp("2026-04-01")],
               "000002": [pd.Timestamp("2026-03-20")], "000003": [pd.Timestamp("2025-11-03")]}
    dates = tuple(DAYS[(DAYS >= "2026-01-15") & (DAYS <= "2026-04-20")][::4])
    full = build_news_features(articles(rows), universe(tickers, dates), DAYS, CFG, tickers, COVERAGE, renames)
    for t in dates:
        known = {k: [d for d in v if d <= t] for k, v in renames.items()}
        part = build_news_features(articles(rows), universe(tickers, (t,)), DAYS, CFG, tickers, COVERAGE, known)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), full[full["date"] == t].reset_index(drop=True))
    first = full[full["date"] == dates[0]].set_index("ticker")
    assert first.loc["000002", "news_cnt_spike_na"] == 0                 # 3/20 변경은 1/15 신호일에 영향이 없다


def test_rank_news_features_empty():
    assert list(rank_news_features(pd.DataFrame(columns=["date", "ticker", *NEWS_FEATURES])).columns) == OUTPUT_COLUMNS


def test_analysis_articles_keeps_only_analysis_targets_and_joins_scores():
    links = pd.DataFrame({"news_id": ["a", "b", "c", "a"], "ticker": ["000001", "000001", "000002", "000002"],
                          "date": pd.to_datetime(["2026-01-05"] * 4), "bk_excluded": ["", "예외", None, ""]})
    scores = pd.DataFrame({"news_id": ["a", "c"], "sentiment": [0.3, -0.4]})
    out = analysis_articles(links, scores)
    assert out.to_dict("list") == {"date": [pd.Timestamp("2026-01-05")] * 3, "ticker": ["000001", "000002", "000002"],
                                   "sentiment": [0.3, -0.4, 0.3]}


def test_current_name_since_takes_latest_change():
    renames = pd.DataFrame({"ticker": ["000001", "000001", "000002"],
                            "change_date": ["2015-03-01", "2019-07-01", "2012-01-02"]})
    assert current_name_since(renames) == {"000001": pd.Timestamp("2019-07-01"), "000002": pd.Timestamp("2012-01-02")}


def test_name_change_dates_keeps_every_change_in_order():
    renames = pd.DataFrame({"ticker": ["000001", "000001", "000002", "000001", "000003"],
                            "change_date": ["2019-07-01", "2015-03-01", "2012-01-02", "2015-03-01", None]})
    assert name_change_dates(renames) == {"000001": [pd.Timestamp("2015-03-01"), pd.Timestamp("2019-07-01")],
                                          "000002": [pd.Timestamp("2012-01-02")]}
