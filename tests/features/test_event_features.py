"""이벤트 피처 ev_* 테스트 (명세서 6.2절). 합성 자료만 쓴다."""

import numpy as np
import pandas as pd
import pytest

from src.common import io
from src.features.event_features import build_event_features, event_names

KST = "Asia/Seoul"
DAYS = pd.bdate_range("2026-01-05", periods=80)
SIGNAL = DAYS[40]
COVERAGE = ("2025-12-01", "2026-12-31")
KEYWORDS = {"ev_rights": ["유상증자결정"], "ev_cbbw": ["전환사채권발행결정", "신주인수권부사채권발행결정"]}
FILTERS = {"strip_prefixes": ["[첨부추가]", "[정정명령부과]"], "exclude_prefixes": ["[기재정정]", "[첨부정정]"],
           "exclude_terms": ["철회", "취소"]}
CFG = {"features": {"event_keywords": KEYWORDS, "event_window_days": 20},
       "events": {"title_filters": FILTERS}, "calendar": {"signal_cutoff_time": "15:30"}}
TITLE = "주요사항보고서(유상증자결정)"
TICKERS = pd.DataFrame({"ticker": ["000001", "000002", "000003"], "listing_date": pd.to_datetime(["2020-01-01"] * 3),
                        "delisting_date": pd.to_datetime([None, None, None])})


def disclosures(rows):
    """rows: (rcept_no, 접수일, available_at 또는 None, stock_code, title). None이면 날짜만 아는 공시(23:59:59)."""
    frame = pd.DataFrame(rows, columns=["rcept_no", "rcept_date", "available_at", "stock_code", "report_nm_norm"])
    frame["rcept_date"] = pd.to_datetime(frame["rcept_date"])
    frame["available_at"] = pd.to_datetime([f"{day:%Y-%m-%d} 23:59:59" if at is None else at for day, at in
                                            zip(frame["rcept_date"], frame["available_at"])]).tz_localize(KST)
    frame["corp_code"] = "C" + frame["stock_code"].astype(str)
    return frame


def universe(dates=(SIGNAL,), names=("000001", "000002")):
    rows = [{"date": t, "ticker": name, "in_universe": True} for t in dates for name in names]
    rows += [{"date": t, "ticker": "000003", "in_universe": False} for t in dates]
    return pd.DataFrame(rows)


def compute(rows, members=None, days=DAYS, coverage=COVERAGE):
    return build_event_features(disclosures(rows), TICKERS, universe() if members is None else members, days, CFG,
                                coverage)


def value(rows, feature="ev_rights", ticker="000001"):
    frame = compute(rows)
    return int(frame.loc[frame["ticker"] == ticker, feature].iloc[0])


@pytest.mark.parametrize("title, expected", [
    (TITLE, ["ev_rights"]),
    ("주요사항보고서(유상증자결정)(자회사의주요경영사항)", ["ev_rights"]),   # 자회사 제외는 T4·T5만 (Q-13)
    ("[첨부추가]" + TITLE, ["ev_rights"]),
    ("[정정명령부과][첨부추가]" + TITLE, ["ev_rights"]),
    ("주요사항보고서(신주인수권부사채권발행결정)", ["ev_cbbw"]),
    ("[기재정정]" + TITLE, []),
    ("[첨부정정]" + TITLE, []),
    ("[정정명령부과][기재정정]" + TITLE, []),
    ("유상증자결정철회", []),
    ("유상증자결정취소", []),
    ("무상증자결정", []),
])
def test_event_names(title, expected):
    assert event_names(title, KEYWORDS, FILTERS) == expected


def test_window_is_twenty_trading_days():
    first = DAYS[40 - 19]
    before = DAYS[40 - 20]
    assert value([("1", first, None, "000001", TITLE)]) == 1
    assert value([("1", before, None, "000001", TITLE)]) == 0


def test_weekend_filing_counts_by_date_range():
    saturday = DAYS[34] + pd.Timedelta(days=1)                           # 창 안의 토요일
    assert saturday.weekday() == 5 and DAYS[21] < saturday < SIGNAL
    assert value([("1", saturday, None, "000001", TITLE)]) == 1

    signal = DAYS[44]                                                    # t−20은 금요일, t−19는 월요일
    early = DAYS[24] + pd.Timedelta(days=1)                              # 그 사이 토요일은 창 밖이다
    assert early.weekday() == 5 and DAYS[24] < early < DAYS[25]
    frame = compute([("1", early, None, "000001", TITLE)], universe([signal]))
    assert frame["ev_rights"].sum() == 0


def test_cutoff_is_strictly_before_1530():
    day = f"{SIGNAL:%Y-%m-%d}"
    assert value([("1", SIGNAL, f"{day} 15:29:59", "000001", TITLE)]) == 1
    assert value([("1", SIGNAL, f"{day} 15:30:00", "000001", TITLE)]) == 0


def test_date_only_filing_waits_for_next_signal():
    assert value([("1", SIGNAL, None, "000001", TITLE)]) == 0            # 23:59:59 (DEC-09)
    assert value([("1", DAYS[39], None, "000001", TITLE)]) == 1


def test_follow_up_and_unlinked_do_not_count():
    assert value([("1", DAYS[39], None, "000001", "[기재정정]" + TITLE)]) == 0
    assert value([("1", DAYS[39], None, "", TITLE)]) == 0                # 종목코드 없음
    assert value([("1", DAYS[39], None, "999999", TITLE)]) == 0          # tickers에 없음


def test_each_type_and_ticker_is_separate():
    frame = compute([("1", DAYS[39], None, "000001", TITLE),
                     ("2", DAYS[39], None, "000002", "주요사항보고서(전환사채권발행결정)")])
    assert frame.set_index("ticker")["ev_rights"].to_dict() == {"000001": 1, "000002": 0}
    assert frame.set_index("ticker")["ev_cbbw"].to_dict() == {"000001": 0, "000002": 1}


def test_rows_are_exactly_universe_and_pass_schema():
    dates = [DAYS[30], DAYS[35], SIGNAL]
    frame = compute([("1", DAYS[29], None, "000003", TITLE)], universe(dates))
    assert sorted(frame.columns) == sorted(["date", "ticker", "ev_rights", "ev_cbbw"])
    assert set(frame["ticker"]) == {"000001", "000002"}                  # U_t 밖 종목 행 없음 (완료 기준 3)
    assert len(frame) == 6 and not frame.duplicated(["date", "ticker"]).any()
    assert not frame.isna().any().any() and frame["ev_rights"].sum() == 0
    io.validate("features", frame)


def test_no_events_gives_all_zero_rows():
    frame = compute([])
    assert len(frame) == 2 and frame[["ev_rights", "ev_cbbw"]].to_numpy().sum() == 0


def test_signal_after_coverage_is_refused():
    with pytest.raises(ValueError, match="보관 구간"):
        compute([], coverage=("2025-12-01", f"{DAYS[39]:%Y-%m-%d}"))


def test_window_before_coverage_is_refused():
    with pytest.raises(ValueError, match="보관 구간"):
        compute([], coverage=(f"{DAYS[30]:%Y-%m-%d}", "2026-12-31"))


def test_signal_must_be_trading_day_with_full_window():
    with pytest.raises(ValueError, match="거래일 목록"):
        compute([], days=DAYS.delete(40))
    with pytest.raises(ValueError, match="거래일이"):
        compute([], universe([DAYS[5]]))


def test_leakage_future_removed_inputs_give_same_features():
    """완료 기준 (2): 무작위 신호일 20개에서 t+1일 이후를 모두 지운 입력과 전체 입력의 결과가 같다."""
    rng = np.random.default_rng(20261109)
    calendar = pd.bdate_range("2025-12-01", periods=260)
    names = [f"{i:06d}" for i in range(1, 41)]
    titles = [TITLE, "주요사항보고서(전환사채권발행결정)", "[기재정정]" + TITLE, "유상증자결정철회", "무상증자결정"]
    rows = []
    for i in range(3000):
        day = calendar[rng.integers(len(calendar))]
        at = None if rng.random() < 0.5 else f"{day:%Y-%m-%d} {rng.integers(7, 19):02d}:{rng.integers(60):02d}:00"
        rows.append((str(i), day, at, names[rng.integers(len(names))], titles[rng.integers(len(titles))]))
    table = disclosures(rows)
    tickers = pd.DataFrame({"ticker": names, "listing_date": pd.Timestamp("2020-01-01"),
                            "delisting_date": pd.NaT})
    signals = list(calendar[25::5])
    members = pd.DataFrame([{"date": t, "ticker": n, "in_universe": True} for t in signals for n in names])
    coverage = ("2025-12-01", "2027-12-31")
    full = build_event_features(table, tickers, members, calendar, CFG, coverage)
    assert full[["ev_rights", "ev_cbbw"]].to_numpy().sum() > 0

    picked = rng.choice(len(signals), size=20, replace=False)
    for t in [signals[i] for i in picked]:
        cut = build_event_features(table[table["rcept_date"] <= t], tickers, members[members["date"] == t],
                                   calendar[calendar <= t], CFG, coverage)
        expected = full[full["date"] == t].reset_index(drop=True)
        pd.testing.assert_frame_equal(cut.reset_index(drop=True), expected)
