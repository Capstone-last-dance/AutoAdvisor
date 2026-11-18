"""forward 신호일 하루치 `ev_*` (DEC-36). M6 테스트의 합성 공시 원본을 그대로 쓴다.

합성 달력의 신호일 T(2026-02-20, 금)의 컷오프는 T 15:30 KST, 기록 마감은 2/23(월) 09:00 KST다.
`ev_*`는 컷오프 전에 받은 관측만 쓰고(C #70 리뷰), M6 근거는 기록 마감 전 최신 관측을 쓴다.
"""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.events import alerts as m9
from src.features.event_features import signal_day_event_features
from src.rag import evidence
from tests.conftest import item
from tests.rag.test_evidence import DAY, DAYS, T, archive_filings, inputs  # noqa: F401  (inputs는 fixture)

CFG = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "prereg.yaml").read_text(encoding="utf-8"))
NAMES = list(CFG["features"]["event_keywords"])
TICKERS = pd.DataFrame({"ticker": ["000001", "000002", "000003"],
                        "listing_date": pd.to_datetime(["2020-01-01"] * 3),
                        "delisting_date": pd.to_datetime([None] * 3)})
UNIVERSE = pd.DataFrame({"date": [DAY, DAY, DAY - pd.Timedelta(days=7)],
                         "ticker": ["000001", "000002", "000003"], "in_universe": [True, True, True]})
POSITION = DAYS.get_loc(DAY)
EARLIEST, TOO_OLD = DAYS[POSITION - 19], DAYS[POSITION - 20]
LOOKBACK = (DAY - TOO_OLD).days
BEFORE_CUTOFF = "2026-02-20T08:00:00+09:00"   # 신호일 아침: t−1까지 창이 끝났다
AFTER_CUTOFF = "2026-02-21T08:00:00+09:00"    # 컷오프 뒤, 기록 마감 전
CUTOFF = pd.Timestamp(f"{T} 15:30", tz=m9.KST)
RECORD_DEADLINE = pd.Timestamp("2026-02-23 09:00", tz=m9.KST)


def filing(n, day, code, title):
    return item(n, day=day.strftime("%Y%m%d"), stock_code=code, corp_code=f"0000{code[-4:]}", report_nm=title)


def archive(client_factory, filings=(), *, observed_at=BEFORE_CUTOFF, day=DAY, lookback=LOOKBACK):
    return archive_filings(client_factory, filings, day=day, observed_at=observed_at, lookback=lookback)


def run():
    return signal_day_event_features(T, UNIVERSE, TICKERS, CFG).set_index("ticker")


def events(out, ticker):
    row = out.loc[ticker].drop("date")
    return sorted(row[row == 1].index)


def m6_versions():
    """M6 규칙(기록 마감 전 최신 관측)으로 고른 접수번호별 (종목, 제목)."""
    frame = evidence._disclosures(DAY, DAYS[POSITION - 19:POSITION + 1], CUTOFF, RECORD_DEADLINE, TICKERS)
    return dict(zip(frame["rcept_no"], zip(frame["ticker"], frame["report_nm"])))


def test_window_cutoff_filters_and_universe_rows(inputs, client_factory):  # noqa: F811
    filings = [
        filing(1, DAY - pd.Timedelta(days=1), "000001", "주요사항보고서(자기주식취득결정)"),  # 창 안 → 1
        filing(2, DAY, "000001", "주요사항보고서(유상증자결정)"),                          # 신호일 접수: 컷오프 전엔 창이 안 끝남 → 0
        filing(3, TOO_OLD, "000002", "주식소각결정"),                                      # 창보다 이름 → 0
        filing(4, DAY - pd.Timedelta(days=1), "000002", "[기재정정]자기주식취득결정"),       # 후속 정정 → 0
        filing(5, EARLIEST, "000002", "영업(잠정)실적(공정공시)"),                          # 창 첫 거래일 → 1
        filing(6, DAY - pd.Timedelta(days=1), "000003", "주식소각결정"),                    # U_t 밖 종목 → 행 없음
    ]
    archive(client_factory, filings)

    out = run()
    assert list(out.columns) == ["date", *NAMES] and sorted(out.index) == ["000001", "000002"]
    assert (out["date"] == DAY).all() and out[NAMES].dtypes.eq("int8").all()
    assert out.loc["000001"].drop("date").to_dict() == {**dict.fromkeys(NAMES, 0), "ev_buyback": 1}
    assert out.loc["000002"].drop("date").to_dict() == {**dict.fromkeys(NAMES, 0), "ev_earnings": 1}


def test_no_filings_is_all_zero_when_collection_is_complete(inputs, client_factory):  # noqa: F811
    archive(client_factory)
    out = run()
    assert (out[NAMES] == 0).all().all() and len(out) == 2


def test_incomplete_collection_stops_instead_of_zero(inputs, client_factory):  # noqa: F811
    archive(client_factory, lookback=(DAY - EARLIEST).days - 1)   # 창 첫 거래일이 빠졌다
    with pytest.raises(m9.AlertsNotCollected):
        run()


@pytest.mark.parametrize("observed_at", [AFTER_CUTOFF, "2026-02-23T09:00:01+09:00"])
def test_collection_finished_after_cutoff_cannot_feed_signal_day(inputs, client_factory, observed_at):  # noqa: F811
    """컷오프 뒤에만 수집됐으면 신호 시점 버전을 확인할 수 없다. 기록 마감 전이어도 0으로 채우지 않는다."""
    archive(client_factory, [filing(1, DAY - pd.Timedelta(days=1), "000001", "자기주식취득결정")],
            day=DAY + pd.Timedelta(days=1), observed_at=observed_at)
    with pytest.raises(m9.AlertsNotCollected):
        run()


def test_title_revised_after_cutoff_does_not_reach_ev(inputs, client_factory):  # noqa: F811
    """C #70 리뷰 사례: 신호 전 관측한 공시의 제목이 컷오프 뒤·기록 마감 전에 바뀌어도 t의 ev_*는 원래 제목을 쓴다."""
    received = DAY - pd.Timedelta(days=4)
    archive(client_factory, [filing(1, received, "000001", "자기주식취득결정")])
    archive(client_factory, [filing(1, received, "000001", "주식소각결정")],
            day=DAY + pd.Timedelta(days=1), observed_at=AFTER_CUTOFF)

    # 두 제목은 서로 다른 ev_*를 만들고, 수정 제목은 기록 마감 전에 보인다(M6 근거에는 이쪽이 실린다).
    (rcept_no, (code, title)), = m6_versions().items()
    assert (code, title) == ("000001", "주식소각결정")

    out = run()
    assert events(out, "000001") == ["ev_buyback"] and events(out, "000002") == []


def test_stock_code_relinked_after_cutoff_keeps_signal_time_ticker(inputs, client_factory):  # noqa: F811
    received = DAY - pd.Timedelta(days=4)
    archive(client_factory, [filing(1, received, "000001", "자기주식취득결정")])
    archive(client_factory, [filing(1, received, "000002", "자기주식취득결정")],
            day=DAY + pd.Timedelta(days=1), observed_at=AFTER_CUTOFF)

    assert [code for code, _ in m6_versions().values()] == ["000002"]
    out = run()
    assert events(out, "000001") == ["ev_buyback"] and events(out, "000002") == []


def test_latest_observation_before_cutoff_wins(inputs, client_factory):  # noqa: F811
    received = DAY - pd.Timedelta(days=4)
    archive(client_factory, [filing(1, received, "000001", "주식소각결정")],
            day=DAY - pd.Timedelta(days=1), observed_at="2026-02-19T20:00:00+09:00")
    archive(client_factory, [filing(1, received, "000001", "자기주식취득결정")])   # 2/20 08:00, 컷오프 전 최신

    out = run()
    assert events(out, "000001") == ["ev_buyback"]


def test_signal_day_must_have_universe(inputs, client_factory):  # noqa: F811
    with pytest.raises(ValueError, match="U_t가 비어"):
        signal_day_event_features(T, UNIVERSE.assign(in_universe=False), TICKERS, CFG)
