"""4.6절 라벨의 수익률 기준일, 결측·거래정지·상장폐지 처리를 검증한다."""

from datetime import date

import pandas as pd
import pytest

from src.backtest.labels import LabelRules, build_labels, publish_labels
from src.common.guards import DataAccessPolicy, GuardViolation

POLICY = DataAccessPolicy(
    latest_training_signal_date=date(2025, 12, 31),
    holdout_start=date(2026, 1, 1),
    holdout_end=date(2026, 10, 31),
    holdout_open_date=date(2026, 12, 11),
)
RULES = LabelRules((1, 5, 20), 20, -0.15)


def sample():
    days = [stamp.date() for stamp in pd.bdate_range("2025-01-02", periods=26)]
    tickers = ("005930", "000660")
    prices = pd.DataFrame(
        [
            {
                "date": day,
                "ticker": ticker,
                "close": (
                    80.0
                    if ticker == "005930" and position >= 21
                    else 120.0 if ticker == "000660" and position >= 21 else 100.0
                ),
            }
            for position, day in enumerate(days)
            for ticker in tickers
        ]
    )
    index = pd.DataFrame(
        {"date": days, "index_code": "KOSPI200", "close": [100.0] * len(days)}
    )
    universe = pd.DataFrame(
        {"date": [days[0]] * 2, "ticker": tickers, "in_universe": [True, True]}
    )
    status = pd.DataFrame(
        [
            {"date": day, "ticker": ticker, "is_halted": False}
            for day in days
            for ticker in tickers
        ]
    )
    delisted = pd.DataFrame(columns=["ticker", "delist_date", "last_price"])
    return days, prices, index, universe, status, delisted


def build(sample_data, **overrides):
    days, prices, index, universe, status, delisted = sample_data
    inputs = dict(
        prices=prices,
        index=index,
        universe=universe,
        status=status,
        delisted=delisted,
        trading_days=days,
        signal_dates=[days[0]],
        rules=RULES,
        policy=POLICY,
        allow_holdout=False,
        run_date=days[-1],
    )
    inputs.update(overrides)
    return build_labels(**inputs)


def test_horizon_starts_at_next_close_and_ranks_within_signal_universe():
    result = build(sample())
    assert list(result.labels) == [
        "date",
        "ticker",
        "h",
        "ret",
        "er",
        "y_rank",
        "y_crash",
    ]
    twenty = result.labels[result.labels.h == 20].set_index("ticker")
    assert twenty.loc["005930", "ret"] == pytest.approx(-0.2)
    assert twenty.loc["005930", "er"] == pytest.approx(-0.2)
    assert twenty.loc["005930", "y_rank"] == pytest.approx(0.5)
    assert twenty.loc["005930", "y_crash"] == 1
    assert twenty.loc["000660", "y_rank"] == pytest.approx(1.0)
    assert twenty.loc["000660", "y_crash"] == 0
    assert result.labels.loc[result.labels.h != 20, "y_crash"].isna().all()


def test_publish_uses_common_io_and_rechecks_holdout(monkeypatch, tmp_path):
    from src.common import io

    monkeypatch.setattr(io, "ROOT", tmp_path)
    result = build(sample())
    path = publish_labels(result, policy=POLICY, allow_holdout=False, run_date=date(2026, 11, 3))
    assert path == tmp_path / "data" / "processed" / "labels.parquet"
    pd.testing.assert_frame_equal(io.read_table("labels"), result.labels, check_dtype=False)
    result.labels.loc[:, "date"] = pd.Timestamp("2026-01-05")
    with pytest.raises(GuardViolation):
        publish_labels(result, policy=POLICY, allow_holdout=False, run_date=date(2026, 11, 3))
    assert io.read_table("labels")["date"].max() < pd.Timestamp("2026-01-01")


def test_entry_halt_omits_all_horizons_without_zero_fill():
    data = sample()
    days, _, _, _, status, _ = data
    status.loc[(status.date == days[1]) & (status.ticker == "000660"), "is_halted"] = (
        True
    )
    result = build(data)
    assert set(result.labels.ticker) == {"005930"}
    assert result.missing_counts == {"entry_halted": 3}


def test_halted_end_uses_first_resumption_price_and_records_actual_end():
    data = sample()
    days, prices, _, _, status, _ = data
    status.loc[(status.date == days[21]) & (status.ticker == "005930"), "is_halted"] = (
        True
    )
    prices.loc[(prices.date == days[22]) & (prices.ticker == "005930"), "close"] = 75.0
    result = build(data)
    row = result.labels[
        (result.labels.ticker == "005930") & (result.labels.h == 20)
    ].iloc[0]
    end = result.end_dates[
        (result.end_dates.ticker == "005930") & (result.end_dates.h == 20)
    ].iloc[0]
    assert row.ret == pytest.approx(-0.25)
    assert end.nominal_end == days[21]
    assert end.actual_end == days[22]


def test_delisting_without_last_price_counts_as_total_loss():
    data = sample()
    days = data[0]
    delisted = pd.DataFrame(
        [{"ticker": "005930", "delist_date": days[10], "last_price": None}]
    )
    result = build(data, delisted=delisted)
    row = result.labels[
        (result.labels.ticker == "005930") & (result.labels.h == 20)
    ].iloc[0]
    assert row.ret == -1.0
    assert row.y_crash == 1


def test_not_mature_and_holdout_signal_are_not_calculated():
    data = sample()
    days = data[0]
    result = build(data, run_date=days[10])
    assert not result.labels.h.eq(20).any()
    assert result.missing_counts["label_not_mature"] == 2
    with pytest.raises(GuardViolation):
        build(data, signal_dates=["2026-01-02"], run_date=date(2026, 11, 2))


def test_missing_nonhalted_end_price_is_not_silently_dropped():
    data = sample()
    days, prices = data[0], data[1]
    prices = prices.drop(
        prices[(prices.date == days[21]) & (prices.ticker == "005930")].index
    )
    with pytest.raises(ValueError, match="종료 수정종가"):
        build(data, prices=prices)


@pytest.mark.parametrize("ticker", ["101970", "036220"])
def test_reused_ticker_old_delisting_does_not_remove_new_company_labels(ticker):
    days = [stamp.date() for stamp in pd.bdate_range("2025-01-02", periods=18)]
    active = list(range(5)) + list(range(10, 18))
    prices = pd.DataFrame([
        {"date": days[i], "ticker": ticker, "close": 50.0 if i == 4 else 80.0 if i >= 10 else 100.0}
        for i in active
    ])
    index = pd.DataFrame({"date": days, "index_code": "KOSPI200", "close": 100.0})
    universe = pd.DataFrame({"date": [days[0], days[10]], "ticker": ticker, "in_universe": True})
    status = prices[["date", "ticker"]].copy()
    status["is_halted"] = False
    delisted = pd.DataFrame([{"ticker": ticker, "delist_date": days[5], "last_price": 50.0}])
    result = build_labels(
        prices=prices, index=index, universe=universe, status=status, delisted=delisted,
        trading_days=days, signal_dates=[days[0], days[10]],
        rules=LabelRules((1, 5), 5, -0.15), policy=POLICY,
        allow_holdout=False, run_date=days[-1],
    )
    old = result.labels[(result.labels.date == pd.Timestamp(days[0])) & (result.labels.h == 5)]
    new = result.labels[result.labels.date == pd.Timestamp(days[10])]
    assert old.iloc[0].ret == pytest.approx(-0.5)
    assert set(new.h) == {1, 5}
    assert new.ret.tolist() == pytest.approx([0.0, 0.0])


def test_dec28_separate_old_delisting_and_current_company_exit():
    days = [stamp.date() for stamp in pd.bdate_range("2025-01-02", periods=18)]
    ticker = "101970"
    active = list(range(5)) + list(range(10, 16))
    prices = pd.DataFrame([
        {"date": days[i], "ticker": ticker,
         "close": 50.0 if i == 4 else 80.0 if i >= 10 else 100.0,
         "close_raw": 25.0 if i == 4 else 80.0 if i >= 10 else 50.0}
        for i in active
    ])
    index = pd.DataFrame({"date": days, "index_code": "KOSPI200", "close": 100.0})
    universe = pd.DataFrame({"date": [days[0], days[10]], "ticker": ticker, "in_universe": True})
    status = prices[["date", "ticker"]].copy()
    status["is_halted"] = False
    old = pd.DataFrame([{"ticker": ticker, "delist_date": days[5], "last_price": 25.0}])
    current = pd.DataFrame([{"ticker": ticker, "delist_date": days[16], "last_price": 40.0}])
    result = build_labels(
        prices=prices, index=index, universe=universe, status=status,
        delisted=current, delisted_reused=old,
        trading_days=days, signal_dates=[days[0], days[10]],
        rules=LabelRules((1, 5), 5, -0.15), policy=POLICY,
        allow_holdout=False, run_date=days[-1],
    )
    old_label = result.labels[(result.labels.date == pd.Timestamp(days[0])) & (result.labels.h == 5)]
    new_label = result.labels[(result.labels.date == pd.Timestamp(days[10])) & (result.labels.h == 5)]
    assert old_label.iloc[0].ret == pytest.approx(-0.5)
    assert new_label.iloc[0].ret == pytest.approx(-0.5)
