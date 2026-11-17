"""작업 B-05: 9.3절 표 → C의 M5 엔진 입력 (engine_adapter)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.backtest.accounting import CostRates
from src.common import io
from src.portfolio.engine_adapter import build_engine, cost_rates, engine_inputs, load_engine_inputs

DAYS = pd.bdate_range("2020-01-01", periods=15)
D = [d.strftime("%Y-%m-%d") for d in DAYS]


def _tables(closes: pd.DataFrame, halted: pd.DataFrame | None = None, raw_ratio: float = 1.0) -> dict[str, pd.DataFrame]:
    """거래일 × 종목 종가 표로 prices, status, index 표를 만든다. 결측 칸은 행이 없다."""
    long = closes.stack().dropna().rename("close").reset_index()
    long.columns = ["date", "ticker", "close"]
    h = (halted if halted is not None else closes.notna() & False).stack()
    long["is_halted"] = h.reindex(pd.MultiIndex.from_frame(long[["date", "ticker"]])).to_numpy(dtype=bool)
    prices = long.assign(close_raw=long["close"] / raw_ratio)
    status = long[["date", "ticker", "is_halted"]]
    index = pd.concat(
        [
            pd.DataFrame({"date": closes.index, "index_code": "KOSPI200", "close": 100.0}),
            pd.DataFrame({"date": closes.index, "index_code": "KOSDAQ150", "close": -1.0}),  # 쓰지 않는 행
        ]
    )
    return {"prices": prices, "status": status, "index": index}


def _delisted(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ticker", "delist_date", "last_price"]).astype({"last_price": "float64"})


def _held_until_delisting(tables, delisted, cfg, ticker="000002", **kw):
    inputs = engine_inputs(tables["prices"], tables["status"], tables["index"], delisted, **kw)
    engine = build_engine(inputs, cfg, record_ledger=True)
    res = engine.run_period(engine.initial_state(), {ticker: 0.5}, D[0], D[8])
    return inputs, engine, res


def test_tables_become_engine_matrices(cfg):
    closes = pd.DataFrame({"000001": 100.0, "000002": 50.0}, index=DAYS)
    closes.loc[DAYS[:3], "000002"] = np.nan  # 상장 전
    halted = pd.DataFrame(False, index=DAYS, columns=closes.columns)
    halted.loc[DAYS[5], "000001"] = True
    t = _tables(closes, halted)
    inputs = engine_inputs(t["prices"], t["status"], t["index"], _delisted([]))
    pd.testing.assert_frame_equal(inputs.closes, closes, check_freq=False)
    assert str(inputs.halted.dtypes.iloc[0]) == "boolean"
    assert inputs.halted.loc[DAYS[:3], "000002"].isna().all()  # 상장 전은 알 수 없음
    assert bool(inputs.halted.at[DAYS[5], "000001"]) and not inputs.halted["000002"].fillna(False).any()
    assert (inputs.bm_close == 100.0).all()  # KOSPI200 행만 쓴다
    assert inputs.delistings == {}


def test_price_without_status_row_or_benchmark_day_is_an_error(cfg):
    closes = pd.DataFrame({"000001": 100.0}, index=DAYS)
    t = _tables(closes)
    with pytest.raises(ValueError, match="status"):
        engine_inputs(t["prices"], t["status"].iloc[1:], t["index"], _delisted([]))
    with pytest.raises(ValueError, match="KOSPI200"):
        engine_inputs(t["prices"], t["status"], t["index"].iloc[1:], _delisted([]))


def test_delisting_moves_to_last_price_day_in_adjusted_units(cfg):
    """delist_date는 마지막 가격일의 다음 거래일이다. 엔진에는 마지막 가격일을 넘기고,
    원종가 last_price는 그날 close ÷ close_raw로 수정종가 단위로 바꾼다."""
    closes = pd.DataFrame({"000001": 100.0, "000002": 40.0}, index=DAYS)
    closes.loc[DAYS[5]:, "000002"] = np.nan
    t = _tables(closes, raw_ratio=0.5)  # 수정종가 = 원종가 × 0.5
    inputs, engine, res = _held_until_delisting(t, _delisted([("000002", DAYS[5], 60.0)]), cfg)
    assert inputs.delistings == {"000002": (D[4], 30.0)}
    row = next(r for r in engine.ledger if r["date"] == D[4])
    (fill,) = [f for f in row["fills"] if f.ticker == "000002"]
    assert fill.price == pytest.approx(30.0)
    assert "000002" not in res.state.holdings


def test_delisting_without_price_is_valued_at_zero(cfg):
    closes = pd.DataFrame({"000001": 100.0, "000002": 40.0}, index=DAYS)
    closes.loc[DAYS[5]:, "000002"] = np.nan
    _, engine, res = _held_until_delisting(_tables(closes), _delisted([("000002", DAYS[5], None)]), cfg)
    row = next(r for r in engine.ledger if r["date"] == D[4])
    assert row["nav_after"] < 0.51 and "000002" not in res.state.holdings  # 보유분 0.5가 0이 된다


@pytest.mark.parametrize(
    ("delist_day", "message"),
    [(DAYS[4], "에도 종가"), (DAYS[0], "전 거래일"), (pd.Timestamp("2021-01-04"), "전 거래일")],
)
def test_delist_date_must_follow_the_last_price(cfg, delist_day, message):
    closes = pd.DataFrame({"000001": 100.0, "000002": 40.0}, index=DAYS)
    closes.loc[DAYS[5]:, "000002"] = np.nan
    t = _tables(closes)
    with pytest.raises(ValueError, match=message):
        engine_inputs(t["prices"], t["status"], t["index"], _delisted([("000002", delist_day, 1.0)]))


def test_reused_ticker_old_company_liquidated_and_new_company_buyable(cfg):
    """DEC-28: delisted_reused의 옛 회사 폐지를 넘기면, 옛 회사 보유분은 폐지일에 청산되고
    가격 공백 뒤 새 회사는 새 종목으로 살 수 있다."""
    closes = pd.DataFrame({"000001": 100.0, "101970": 40.0}, index=DAYS)
    closes.loc[DAYS[5]:DAYS[8], "101970"] = np.nan
    closes.loc[DAYS[9]:, "101970"] = 90.0
    t = _tables(closes)
    reused = _delisted([("101970", DAYS[5], 41.0)])
    inputs, engine, res = _held_until_delisting(t, _delisted([]), cfg, ticker="101970", delisted_reused=reused)
    assert inputs.delistings == {"101970": (D[4], 41.0)}
    assert "101970" not in res.state.holdings
    new = engine.run_period(res.state, {"101970": 0.3}, D[8], D[12])
    assert new.state.holdings["101970"] > 0
    with pytest.raises(ValueError, match="두 개"):
        engine_inputs(t["prices"], t["status"], t["index"], reused, delisted_reused=reused)


def test_price_gap_without_a_delisting_at_its_start_stops(cfg):
    """DEC-28: 가격이 끊겼다가 다시 나오는 종목은 끊긴 첫날에 폐지 기록이 있어야 한다. delisted_reused가
    빠지거나 비어 있으면 그 종목을 들고 있지 않아도 멈춘다 (A 리뷰, PR #21)."""
    closes = pd.DataFrame({"000001": 100.0, "101970": 40.0}, index=DAYS)
    closes.loc[DAYS[5]:DAYS[8], "101970"] = np.nan
    t = _tables(closes)
    for reused in (None, _delisted([])):
        with pytest.raises(ValueError, match="2020-01-08부터 가격이 끊겼다가"):
            engine_inputs(t["prices"], t["status"], t["index"], _delisted([]), delisted_reused=reused)
    twice = closes.copy()
    twice.loc[DAYS[11], "101970"] = np.nan
    t2 = _tables(twice)
    with pytest.raises(ValueError, match="2번 끊긴다"):
        engine_inputs(t2["prices"], t2["status"], t2["index"], _delisted([]), _delisted([("101970", DAYS[5], 41.0)]))


def test_load_engine_inputs_reads_delisted_reused_through_io(cfg, monkeypatch):
    """D PR #18 뒤로 delisted_reused도 io.read_table로 읽는다. 파일이 없으면 FileNotFoundError가 그대로 나간다."""
    closes = pd.DataFrame({"000001": 100.0, "101970": 40.0}, index=DAYS)
    closes.loc[DAYS[5]:DAYS[8], "101970"] = np.nan
    tables = {**_tables(closes), "delisted": _delisted([]), "delisted_reused": _delisted([("101970", DAYS[5], 41.0)])}
    read: list[str] = []

    def fake_read_table(name, *args, **kwargs):
        read.append(name)
        if name not in tables:
            raise FileNotFoundError(name)
        return tables[name]

    monkeypatch.setattr(io, "read_table", fake_read_table)
    assert load_engine_inputs().delistings == {"101970": (D[4], 41.0)}
    assert read == ["prices", "status", "index", "delisted", "delisted_reused"]
    del tables["delisted_reused"]
    with pytest.raises(FileNotFoundError, match="delisted_reused"):
        load_engine_inputs()


def test_relisting_the_engine_does_not_recognize_stops(cfg):
    """엔진이 새 상장으로 보는 코드는 DEC-28의 두 건뿐이다. 다른 코드에 같은 공백이 있으면 멈춘다."""
    closes = pd.DataFrame({"000001": 100.0, "000009": 40.0}, index=DAYS)
    closes.loc[DAYS[5]:DAYS[8], "000009"] = np.nan
    t = _tables(closes)
    inputs = engine_inputs(t["prices"], t["status"], t["index"], _delisted([("000009", DAYS[5], 40.0)]))
    with pytest.raises(ValueError, match="DEC-28"):
        build_engine(inputs, cfg)


def test_halted_holding_is_valued_at_last_traded_close(cfg):
    """DEC-25·26: 정지일(volume = 0) 원천 종가가 바뀌어도 엔진은 마지막 거래일 수정종가로 평가한다."""
    closes = pd.DataFrame({"000001": 100.0}, index=DAYS)
    closes.loc[DAYS[3], "000001"] = 130.0
    halted = pd.DataFrame(False, index=DAYS, columns=closes.columns)
    halted.loc[DAYS[3], "000001"] = True
    t = _tables(closes, halted)
    engine = build_engine(engine_inputs(t["prices"], t["status"], t["index"], _delisted([])), cfg, record_ledger=True)
    engine.run_period(engine.initial_state(), {"000001": 0.5}, D[0], D[5])
    navs = {r["date"]: r["nav_after"] for r in engine.ledger}
    assert navs[D[3]] == pytest.approx(navs[D[2]])


def test_cost_rates_base_and_stress(cfg):
    c = cfg["costs"]
    assert cost_rates(cfg) == CostRates(c["sell_tax"], c["commission_per_side"], c["slippage_per_side"], 1.0)
    assert cost_rates(cfg, stress=True).stress_multiplier == c["stress_multiplier"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [("costs", "cash_return", 0.01), ("execution", "delisted_without_price_value", 0.5),
     ("calendar", "execution", "same_day_close")],
)
def test_settings_the_engine_does_not_implement_are_rejected(cfg, section, key, value):
    bad = copy.deepcopy(cfg)
    bad[section][key] = value
    closes = pd.DataFrame({"000001": 100.0}, index=DAYS)
    t = _tables(closes)
    inputs = engine_inputs(t["prices"], t["status"], t["index"], _delisted([]))
    with pytest.raises(ValueError, match=key):
        build_engine(inputs, bad)
