import math

import numpy as np
import pandas as pd
import pytest

from src.data.schemas import validate
from src.models.baseline import baseline_signals, run_step


def _close(n_days=400, n_tk=30, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"{i:06d}" for i in range(n_tk)]
    vol = np.linspace(0.005, 0.05, n_tk)                      # 종목 번호가 클수록 변동성이 크다
    drift = np.linspace(0.002, -0.002, n_tk)                  # 종목 번호가 작을수록 모멘텀이 좋다
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(drift, vol, (n_days, n_tk)), axis=0)), index=idx, columns=cols)


def test_schema_ranks_and_flag_count():
    c = _close(); t = c.index[-1]
    s = baseline_signals(c, list(c.columns), t)
    validate("signals", s)
    assert len(s) == 30 and s["rank"].tolist() == list(range(1, 31))
    assert s["crash_flag"].sum() == math.ceil(0.10 * 30) == 3
    assert set(s.loc[s["crash_flag"] == 1, "ticker"]) == {"000029", "000028", "000027"}   # vol_60 최상위 3개
    assert s["model_version"].eq("baseline").all() and s["date"].eq(t).all()


def test_rank_follows_mom_12_1_descending():
    c = _close(); t = c.index[-1]
    s = baseline_signals(c, list(c.columns), t).set_index("ticker")
    mom = (c.iloc[-1 - 21] / c.iloc[-1 - 252] - 1)
    assert s["rank"].idxmin() == mom.idxmax() and s["rank"].idxmax() == mom.idxmin()


def test_no_lookahead():
    c = _close(n_days=500); t = c.index[300]
    a = baseline_signals(c, list(c.columns), t)
    b = baseline_signals(c.loc[:t], list(c.columns), t)       # t 이후 데이터를 지운 입력
    pd.testing.assert_frame_equal(a, b)


def test_missing_history_gets_neutral_and_no_flag():
    c = _close(); c["000005"] = np.nan                         # 시세가 전혀 없는 종목
    s = baseline_signals(c, list(c.columns) + ["999999"], c.index[-1]).set_index("ticker")   # c에 없는 종목도 U_t면 포함
    assert s.loc["000005", "rank_score"] == 0.5 and s.loc["999999", "crash_prob"] == 0.5
    assert s.loc["000005", "crash_flag"] == 0 and s.loc["999999", "crash_flag"] == 0
    assert s["rank"].is_unique


def test_deterministic_ties():
    c = _close(); c["000001"] = c["000000"]                    # 완전히 같은 가격 경로
    a = baseline_signals(c, list(c.columns), c.index[-1]); b = baseline_signals(c, list(c.columns), c.index[-1])
    pd.testing.assert_frame_equal(a, b)


class _FakeCtx:
    """D의 StepContext와 같은 모양(read_table/write_table)만 흉내 낸다 (run_weekly.py는 쓰지 않는다)."""

    def __init__(self, date, tables):
        self.date = date
        self._tables = tables
        self.written = {}

    def read_table(self, name):
        return self._tables[name]

    def write_table(self, name, df, date=None):
        self.written[name] = (date, df)


def test_run_step_reads_universe_and_prices_and_writes_signals():
    c = _close(); t = c.index[-1]
    prices = c.stack().rename("close").reset_index()
    prices.columns = ["date", "ticker", "close"]
    universe = pd.DataFrame({"date": t, "ticker": list(c.columns), "mcap_rank": range(1, len(c.columns) + 1),
                             "in_universe": True})
    ctx = _FakeCtx(f"{t:%Y-%m-%d}", {"prices": prices, "universe": universe})
    run_step(ctx)
    name, (date, df) = next(iter(ctx.written.items()))
    assert name == "signals" and date == ctx.date
    validate("signals", df)
    pd.testing.assert_frame_equal(df, baseline_signals(c, list(c.columns), t))


def test_run_step_rejects_missing_signal_date():
    ctx = _FakeCtx("2099-01-01", {"prices": pd.DataFrame(columns=["date", "ticker", "close"]),
                                  "universe": pd.DataFrame(columns=["date", "ticker", "in_universe"])})
    with pytest.raises(ValueError, match="신호일"):
        run_step(ctx)
