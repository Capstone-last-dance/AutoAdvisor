import numpy as np
import pandas as pd
import pytest

from src.features.price_features import compute_market_features, compute_stock_features, rank_and_fill, ss_allowed


def _synth(n_days=400, n_tk=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"{i:06d}" for i in range(n_tk)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_tk)), axis=0)), index=idx, columns=cols)
    value = pd.DataFrame(rng.uniform(1e9, 5e9, (n_days, n_tk)), index=idx, columns=cols)
    mcap = pd.DataFrame(rng.uniform(1e11, 5e11, (n_days, n_tk)), index=idx, columns=cols)
    flows = [pd.DataFrame(rng.normal(0, 1e8, (n_days, n_tk)), index=idx, columns=cols) for _ in range(3)]
    return close, close * 10, value, mcap, *flows


def test_mom_and_rev_definitions_by_hand():
    close, raw, value, mcap, r, f, i = _synth()
    ft = compute_stock_features(close, raw, value, mcap, r, f, i)
    t = close.index[300]; k = 300
    P = lambda s: close.iloc[k - s]
    pd.testing.assert_series_equal(ft["mom_12_1"].loc[t], (P(21) / P(252) - 1), check_names=False)
    pd.testing.assert_series_equal(ft["mom_12_7"].loc[t], (P(147) / P(252) - 1), check_names=False)
    pd.testing.assert_series_equal(ft["rev_1m"].loc[t], (P(0) / P(21) - 1), check_names=False)
    pd.testing.assert_series_equal(ft["rev_1w"].loc[t], (P(0) / P(5) - 1), check_names=False)
    assert ft["high52_prox"].loc[t].iloc[0] == pytest.approx(close.iloc[k, 0] / close.iloc[k - 251: k + 1, 0].max())


def test_vol_and_max_ret_windows():
    close, raw, value, mcap, r, f, i = _synth()
    ft = compute_stock_features(close, raw, value, mcap, r, f, i)
    k = 200
    lr = np.log(close.iloc[k - 59: k + 1, 0] / close.iloc[k - 60: k, 0].values)   # t-59..t 일간 로그수익률 60개
    assert ft["vol_60"].iloc[k, 0] == pytest.approx(lr.std(ddof=1) * np.sqrt(252))
    sr = close.iloc[k - 19: k + 1, 0].values / close.iloc[k - 20: k, 0].values - 1
    assert ft["max_ret_20"].iloc[k, 0] == pytest.approx(sr.max())
    assert ft["vol_60"].iloc[59].isna().all() and ft["vol_60"].iloc[60].notna().all()       # 수익률 60개 = 가격 61개가 모이는 첫날


def test_investor_flow_window_uses_t_minus_1_and_earlier():
    close, raw, value, mcap, r, f, i = _synth()
    k = 100
    ft = compute_stock_features(close, raw, value, mcap, r, f, i)
    expect = r.iloc[k - 20: k, 0].sum() / mcap.iloc[k - 1, 0]                               # t-20 ~ t-1
    assert ft["retail_net_20"].iloc[k, 0] == pytest.approx(expect)
    r2 = r.copy(); r2.iloc[k, :] = 1e15                                                    # t일 자료를 바꿔도 t의 피처는 같다
    assert compute_stock_features(close, raw, value, mcap, r2, f, i)["retail_net_20"].iloc[k, 0] == pytest.approx(expect)


def test_turnover_spike_windows():
    close, raw, value, mcap, r, f, i = _synth()
    k = 150
    turn = value / mcap
    expect = turn.iloc[k - 4: k + 1, 0].mean() / turn.iloc[k - 64: k - 4, 0].mean()        # 분자 t-4~t, 분모 t-64~t-5
    assert compute_stock_features(close, raw, value, mcap, r, f, i)["turnover_spike"].iloc[k, 0] == pytest.approx(expect)


def test_penny_flag_uses_raw_close():
    close, raw, value, mcap, r, f, i = _synth()
    raw.iloc[10, 0] = 999; raw.iloc[10, 1] = 1000
    pf = compute_stock_features(close, raw, value, mcap, r, f, i)["penny_flag"]
    assert pf.iloc[10, 0] == 1.0 and pf.iloc[10, 1] == 0.0


def test_no_lookahead_features_identical_when_future_removed():
    """6.2절 누수 테스트: 신호일 이후 데이터를 지운 입력과 전체 입력의 피처가 완전히 같다."""
    full = _synth(n_days=500)
    ft_full = compute_stock_features(*full)
    rng = np.random.default_rng(1)
    for k in rng.choice(np.arange(260, 480), size=20, replace=False):
        cut = [x.iloc[: k + 1] for x in full]
        ft_cut = compute_stock_features(*cut)
        for name in ft_full:
            a, b = ft_full[name].iloc[k], ft_cut[name].iloc[k]
            pd.testing.assert_series_equal(a, b, check_names=False)


def test_market_features_and_ss_allowed():
    idx = pd.bdate_range("2020-01-01", periods=60)
    bm = pd.Series(np.linspace(100, 130, 60), index=idx)
    m = compute_market_features(bm, pd.Series(0, index=idx), pd.Series(2, index=idx))
    assert m["mkt_ret_20"].iloc[30] == pytest.approx(bm.iloc[30] / bm.iloc[10] - 1)
    assert np.isnan(m["mkt_vol_20"].iloc[19]) and not np.isnan(m["mkt_vol_20"].iloc[20])
    assert ss_allowed(0, False, False) == 1 and ss_allowed(2, True, True) == 0
    assert ss_allowed(1, True, False) == 1 and ss_allowed(1, False, False) == 0
    assert np.isnan(ss_allowed(1, pd.NA, pd.NA))


def test_rank_and_fill():
    df = pd.DataFrame({"vol_60": [0.1, 0.3, np.nan, 0.2]})
    out = rank_and_fill(df, rank_cols=["vol_60"])
    assert out["vol_60"].tolist() == [pytest.approx(1 / 3), 1.0, 0.5, pytest.approx(2 / 3)]   # 순위는 비결측 3개 기준 pct
    assert out["vol_60_na"].tolist() == [0, 0, 1, 0]
