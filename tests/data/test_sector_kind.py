from datetime import date

import pandas as pd
import pytest

from src.data.schemas import validate
from src.data.sector_kind import UNKNOWN_SECTOR_LABEL, build_sector, parse_corp_list, unclassified_coverage


def _html(rows: list[tuple[str, str, str]]) -> bytes:
    """회사명, 종목코드, 업종만 담은 최소 KIND 표 (실제 응답은 EUC-KR, 테스트는 ASCII로 충분)."""
    trs = "".join(f"<tr><td>{n}</td><td>{c}</td><td>{s}</td></tr>" for n, c, s in rows)
    html = f"<table><tr><th>회사명</th><th>종목코드</th><th>업종</th></tr>{trs}</table>"
    return html.encode("euc-kr")


def _tickers(rows: list[tuple[str, object, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ticker", "listing_date", "delisting_date"]).assign(
        listing_date=lambda d: pd.to_datetime(d["listing_date"]),
        delisting_date=lambda d: pd.to_datetime(d["delisting_date"]),
    )


def test_parse_corp_list_keeps_name_ticker_sector():
    df = parse_corp_list(_html([("가전자", "000001", "소프트웨어 개발 및 공급업"), ("나상사", "5", "금융 지원 서비스업")]))
    assert list(df.columns) == ["ticker", "name", "sector"]
    assert df["ticker"].tolist() == ["000001", "000005"]           # 6자리로 0 채움


def test_parse_corp_list_rejects_duplicate_ticker():
    with pytest.raises(RuntimeError, match="중복"):
        parse_corp_list(_html([("가", "000001", "A"), ("나", "000001", "B")]))


def test_parse_corp_list_rejects_blank_sector():
    with pytest.raises(RuntimeError, match="업종이 빈"):
        parse_corp_list(_html([("가", "000001", "")]))


def test_build_sector_uses_kind_for_current_and_unknown_for_delisted():
    tickers = _tickers([
        ("000001", "2010-01-04", None),           # 현재 상장, KIND에 있음
        ("000002", "2010-01-04", "2020-01-02"),   # 폐지 (KIND 최신 목록엔 당연히 없음)
        ("000003", "2015-06-01", None),           # 현재 상장인데 KIND에 없음 (예: 우선주)
    ])
    corp_list = pd.DataFrame({"ticker": ["000001"], "name": ["가"], "sector": ["소프트웨어 개발 및 공급업"]})
    sector = build_sector(tickers, corp_list)
    validate("sector", sector)
    s = sector.set_index("ticker")["sector"]
    assert s["000001"] == "소프트웨어 개발 및 공급업"
    assert s["000002"] == UNKNOWN_SECTOR_LABEL and s["000003"] == UNKNOWN_SECTOR_LABEL
    v = sector.set_index("ticker")
    assert v.loc["000001", "valid_from"] == pd.Timestamp("2010-01-04") and pd.isna(v.loc["000001", "valid_to"])
    assert v.loc["000002", "valid_to"] == pd.Timestamp("2020-01-02")


def test_unclassified_coverage_ratio_per_signal_date():
    universe = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-08"] * 2),
        "ticker": ["000001", "000002", "000003", "000001", "000002"],
        "mcap_rank": [1, 2, 3, 1, 2],
        "in_universe": [True, True, False, True, True],   # 000003은 2020-01-01에 U_t 밖
    })
    sector = pd.DataFrame({
        "ticker": ["000001", "000002"],
        "sector": ["A", UNKNOWN_SECTOR_LABEL],
        "valid_from": pd.to_datetime(["2010-01-01", "2010-01-01"]),
        "valid_to": [pd.NaT, pd.NaT],
    })
    cov = unclassified_coverage(universe, sector).set_index("date")
    assert cov.loc[pd.Timestamp("2020-01-01"), "n_total"] == 2            # in_universe인 것만: 000001, 000002
    assert cov.loc[pd.Timestamp("2020-01-01"), "unclassified_ratio"] == pytest.approx(0.5)
    assert cov.loc[pd.Timestamp("2020-01-08"), "unclassified_ratio"] == pytest.approx(0.5)
