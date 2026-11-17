"""forward 신호일 하루치 `ev_*` (DEC-36). M6 테스트의 합성 공시 원본을 그대로 쓴다."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.events import alerts as m9
from src.features.event_features import signal_day_event_features
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


def filing(n, day, code, title):
    return item(n, day=day.strftime("%Y%m%d"), stock_code=code, corp_code=f"0000{code[-4:]}", report_nm=title)


def run():
    return signal_day_event_features(T, UNIVERSE, TICKERS, CFG).set_index("ticker")


def test_window_cutoff_filters_and_universe_rows(inputs, client_factory):  # noqa: F811
    filings = [
        filing(1, DAY - pd.Timedelta(days=1), "000001", "주요사항보고서(자기주식취득결정)"),  # 창 안 → 1
        filing(2, DAY, "000001", "주요사항보고서(유상증자결정)"),                          # 신호일 접수는 23:59:59 이용 → 0
        filing(3, TOO_OLD, "000002", "주식소각결정"),                                      # 창보다 이름 → 0
        filing(4, DAY - pd.Timedelta(days=1), "000002", "[기재정정]자기주식취득결정"),       # 후속 정정 → 0
        filing(5, EARLIEST, "000002", "영업(잠정)실적(공정공시)"),                          # 창 첫 거래일 → 1
        filing(6, DAY - pd.Timedelta(days=1), "000003", "주식소각결정"),                    # U_t 밖 종목 → 행 없음
    ]
    archive_filings(client_factory, filings, lookback=(DAY - TOO_OLD).days)

    out = run()
    assert list(out.columns) == ["date", *NAMES] and sorted(out.index) == ["000001", "000002"]
    assert (out["date"] == DAY).all() and out[NAMES].dtypes.eq("int8").all()
    assert out.loc["000001"].drop("date").to_dict() == {**dict.fromkeys(NAMES, 0), "ev_buyback": 1}
    assert out.loc["000002"].drop("date").to_dict() == {**dict.fromkeys(NAMES, 0), "ev_earnings": 1}


def test_no_filings_is_all_zero_when_collection_is_complete(inputs, client_factory):  # noqa: F811
    archive_filings(client_factory, [], lookback=(DAY - TOO_OLD).days)
    out = run()
    assert (out[NAMES] == 0).all().all() and len(out) == 2


def test_incomplete_collection_stops_instead_of_zero(inputs, client_factory):  # noqa: F811
    archive_filings(client_factory, [], lookback=(DAY - EARLIEST).days - 1)   # 창 첫 거래일이 빠졌다
    with pytest.raises(m9.AlertsNotCollected):
        run()


def test_archive_after_record_deadline_cannot_feed_signal_day(inputs, client_factory):  # noqa: F811
    archive_filings(client_factory, [filing(1, DAY - pd.Timedelta(days=1), "000001", "자기주식취득결정")],
                    observed_at="2026-02-23T09:00:01+09:00", lookback=(DAY - TOO_OLD).days)
    with pytest.raises(m9.AlertsNotCollected):
        run()


def test_signal_day_must_have_universe(inputs, client_factory):  # noqa: F811
    with pytest.raises(ValueError, match="U_t가 비어"):
        signal_day_event_features(T, UNIVERSE.assign(in_universe=False), TICKERS, CFG)
