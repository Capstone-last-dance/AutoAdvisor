import numpy as np
import pandas as pd
import pytest

from src.features.price_features import RANK_FEATURES
from src.features.store import compute_feature_matrix, merge_features, missing_rate_table, stub_regime_mc
from tests.features.test_price_features import _synth


def _inputs(n_days=520, n_tk=12):
    close, raw, value, mcap, r, f, i = _synth(n_days, n_tk)
    bm = close.mean(axis=1)
    idx = close.index
    ss = pd.Series(0, index=idx)
    sig = [idx[k] for k in (300, 340, 400, 460)]
    uni = {t: list(close.columns[:8]) for t in sig}
    return (close, raw, value, mcap, r, f, i), bm, ss, uni


def _run(args, bm, ss, uni, **kw):
    return compute_feature_matrix(*args, bm=bm, regime_mc=stub_regime_mc(bm.index), ss_regime=ss, universe=uni, **kw)


def test_shape_columns_and_only_universe_rows():
    args, bm, ss, uni = _inputs()
    f = _run(args, bm, ss, uni)
    assert len(f) == 4 * 8
    for name in RANK_FEATURES:
        assert name in f and f"{name}_na" in f
        assert f[name].between(0, 1).all()                               # 순위 피처는 0~1
    for t, g in f.groupby("date"):
        assert set(g["ticker"]) == set(uni[t])                           # U_t 밖 종목 행 없음 (6.2절 완료 기준 (3))
    assert f["ticker"].str.fullmatch(r"\d{6}").all()
    assert (f["ss_allowed"] == 1).all() and (f["ss_regime"] == 0).all()


def test_ranks_are_within_universe_per_date():
    args, bm, ss, uni = _inputs()
    f = _run(args, bm, ss, uni)
    g = f[f["date"] == f["date"].iloc[0]]
    assert sorted(g["mom_12_1"].round(6).tolist()) == pytest.approx([i / 8 for i in range(1, 9)])


def test_no_lookahead_feature_matrix():
    """6.2절 완료 기준 (2): 무작위 신호일 20개에 대해 t 이후 데이터를 지운 입력과 전체 입력의 피처가 같다 (A 담당 피처)."""
    args, bm, ss, _ = _inputs(n_days=560)
    idx = args[0].index
    rng = np.random.default_rng(7)
    ks = sorted(rng.choice(np.arange(270, 540), size=20, replace=False))
    uni = {idx[k]: list(args[0].columns[:8]) for k in ks}
    full = _run(args, bm, ss, uni)
    for k in ks[::4]:                                                    # 5개 시점 재계산 (전체 20개 비교는 입력 자르기 비용이 커서 격일로)
        t = idx[k]
        cut = [x.loc[:t] for x in args]
        part = compute_feature_matrix(*cut, bm=bm.loc[:t], regime_mc=stub_regime_mc(bm.loc[:t].index), ss_regime=ss.loc[:t],
                                      universe={t: uni[t]})
        a = full[full["date"] == t].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, part.reset_index(drop=True))


def test_raw_feature_missing_raises():
    args, bm, ss, uni = _inputs()
    ss = pd.Series(1, index=bm.index)                                    # regime 1인데 구성종목 정보가 없으면 ss_allowed 결측
    with pytest.raises(ValueError, match="결정 필요"):
        _run(args, bm, ss, uni)


def test_ss_allowed_with_constituents():
    args, bm, ss, uni = _inputs()
    ss = pd.Series(1, index=bm.index)
    cols = list(args[0].columns)
    k2 = pd.DataFrame(pd.NA, index=list(uni), columns=cols, dtype="boolean")
    k2.loc[:, cols[:3]] = True
    k2 = k2.fillna(False)
    q1 = pd.DataFrame(False, index=list(uni), columns=cols, dtype="boolean")
    f = _run(args, bm, ss, uni, in_kospi200=k2, in_kosdaq150=q1)
    assert f.groupby("ticker")["ss_allowed"].first().to_dict() == {c: (1.0 if c in cols[:3] else 0.0) for c in cols[:8]}


def test_merge_features_rules():
    args, bm, ss, uni = _inputs()
    a = _run(args, bm, ss, uni)
    pm = a[["date", "ticker"]].assign(ev_earnings=0, news_cnt_spike=0.5, news_cnt_spike_na=0)
    m = merge_features(a, pm)
    assert len(m) == len(a) and "ev_earnings" in m
    with pytest.raises(ValueError, match="U_t 밖"):
        merge_features(a, pd.concat([pm, pm.iloc[:1].assign(ticker="999999")]))
    with pytest.raises(ValueError, match="없는 U_t"):
        merge_features(a, pm.iloc[1:])
    with pytest.raises(ValueError, match="겹친다"):
        merge_features(a, a[["date", "ticker", "vol_20"]])
    assert merge_features(a, None) is a


def test_missing_rate_table_counts_history_gaps():
    args, bm, ss, _ = _inputs(n_days=520)
    idx = args[0].index
    uni = {idx[100]: list(args[0].columns[:8]), idx[400]: list(args[0].columns[:8])}
    f = _run(args, bm, ss, uni)
    tab = missing_rate_table(f)
    assert tab.loc["mom_12_1", "missing_rate"] == 0.5                    # 100번째 날은 252일 이력이 없어 전부 결측
    assert tab.loc["vol_60", "missing_rate"] == 0.0
