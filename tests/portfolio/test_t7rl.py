"""T7-80-RL (src/portfolio/t7rl.py) 단위 테스트. 합성 자료만 쓴다 (고정 파일·실제 정책·2026 자료 없음)."""
from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.common.guards import GuardViolation
from src.portfolio import t7rl
from src.portfolio.params import BufferRuleParams
from tests.portfolio.fakes import ScriptedEngine, make_snapshot


def test_smooth_signals_rolling_mean_and_rerank():
    d = pd.to_datetime(["2026-01-02"] * 3 + ["2026-01-09"] * 3 + ["2026-01-16"] * 3)
    sig = pd.DataFrame({"date": d, "ticker": ["000001", "000002", "000003"] * 3,
                        "rank_score": [0.9, 0.5, 0.1, 0.1, 0.5, 0.9, 0.2, 0.2, 0.8], "rank": 0})
    out = t7rl.smooth_signals(sig, weeks=2)
    # 첫 신호일은 그 값, 다음부터 최근 2개 평균 (앞 신호가 적으면 있는 만큼)
    assert out.loc[out["date"] == d[0], "rank_score"].tolist() == [0.9, 0.5, 0.1]
    assert np.allclose(out.loc[out["date"] == d[3], "rank_score"], [0.5, 0.5, 0.5])
    assert np.allclose(out.loc[out["date"] == d[6], "rank_score"], [0.15, 0.35, 0.85])
    # 같은 점수는 종목코드 순으로 순위
    assert out.loc[out["date"] == d[3], "rank"].tolist() == [1, 2, 3]
    assert out.loc[out["date"] == d[6], "rank"].tolist() == [3, 2, 1]


def test_basket_features_definitions():
    days = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    px = pd.Series(np.exp(np.cumsum(rng.normal(0.0, 0.01, len(days)))), index=days)
    f = t7rl.basket_features(px)
    assert list(f.columns) == list(t7rl.OWN)
    k = 280
    lp = np.log(px.to_numpy())
    assert math.isclose(f["o_r5"].iloc[k], lp[k] - lp[k - 5])
    assert math.isclose(f["o_r126"].iloc[k], lp[k] - lp[k - 126])
    r1 = np.diff(lp)
    assert math.isclose(f["o_vol20"].iloc[k], np.std(r1[k - 20:k], ddof=1) * math.sqrt(252))
    assert math.isclose(f["o_dd252"].iloc[k], lp[k] - lp[k - 251:k + 1].max())
    assert math.isclose(f["o_ma200"].iloc[k], px.iloc[k] / px.iloc[k - 199:k + 1].mean() - 1)
    assert f["o_ma200"].iloc[198] != f["o_ma200"].iloc[198]  # 200일 전에는 빈 값


def test_us_features_union_calendar_and_strictly_before():
    vix = pd.Series([20.0, 22.0, 24.0], index=pd.to_datetime(["2020-03-02", "2020-03-03", "2020-03-05"]))
    other = pd.Series([1.0, 1.0], index=pd.to_datetime(["2020-03-04", "2020-03-05"]))  # VIX 없는 날 03-04
    raw = {"VIXCLS": vix, "BAA10Y": other, "DEXKOUS": other, "NASDAQCOM": other}
    us = t7rl.us_table(raw)
    assert list(us.index) == list(pd.to_datetime(["2020-03-02", "2020-03-03", "2020-03-04", "2020-03-05"]))
    assert math.isclose(us.loc["2020-03-04", "g_vix"], math.log(22.0))  # 앞 값으로 채운다
    g = t7rl.us_features(us, ["2020-03-05", "2020-03-06"])
    # 결정일 당일 미국 값은 쓰지 않는다 (t보다 앞선 마지막 미국 날짜)
    assert list(g["us_date"]) == list(pd.to_datetime(["2020-03-04", "2020-03-05"]))
    assert math.isclose(g["g_vix"].iloc[1], math.log(24.0))
    with pytest.raises(ValueError):
        t7rl.us_features(us, ["2020-03-02"])
    with pytest.raises(ValueError):
        t7rl.us_table({"VIXCLS": vix})


def test_cash_daily_and_extend_monthly():
    monthly = pd.Series([0.0365, 0.0730], index=pd.PeriodIndex(["2025-12", "2026-01"], freq="M"))
    days = pd.to_datetime(["2025-12-30", "2026-01-02", "2026-01-05"])
    r = t7rl.cash_daily(days, monthly)
    assert r[0] == 0.0
    assert math.isclose(r[1], 1.0365 ** (3 / 365) - 1)  # 전날(12월)의 금리, 달력일 3일
    assert math.isclose(r[2], 1.0730 ** (3 / 365) - 1)
    with pytest.raises(KeyError):
        t7rl.cash_daily(pd.to_datetime(["2026-02-27", "2026-03-02"]), monthly)
    ext, filled = t7rl.extend_monthly(monthly, pd.Period("2026-03", freq="M"))
    assert filled == ["2026-02", "2026-03"] and float(ext[pd.Period("2026-03", freq="M")]) == 0.0730


def test_book_fills_next_day_with_costs():
    px = np.array([1.0, 1.0, 1.1, 1.1])
    rc = np.zeros(4)
    b = t7rl.Book(px, 0, 0.5, rc, rate=0.001, tax=0.002)
    assert math.isclose(b.weight(), 0.5)
    b.run(1, 0.2)  # 0.5 → 0.2: 매도 0.3에 수수료 + 매도세
    assert math.isclose(b.nav, 1.0 - 0.3 * (0.001 + 0.002))
    assert math.isclose(b.weight(), 0.2)
    b.run(3, 0.2)  # 가격 10% 상승 뒤 비중이 늘고, 맞추며 판다
    assert b.i == 3 and b.weight() < 0.21


class _Fixed:
    """predict(obs)가 정한 행동을 돌려주는 가짜 정책. obs를 기록한다."""

    def __init__(self, action_of):
        self.action_of, self.seen = action_of, []

    def predict(self, obs, deterministic=True):
        assert deterministic and obs.dtype == np.float32
        self.seen.append(obs.copy())
        return np.array(self.action_of(obs)), None  # SB3처럼 관측 하나에 0차원 행동


def test_exposure_path_mean_of_levels_and_book_weight():
    days = pd.bdate_range("2025-01-01", periods=40)
    px = pd.Series(1.0, index=days)
    dec_days = days[[0, 10, 20, 30]]
    feats = pd.DataFrame({"date": dec_days, **{c: 0.0 for c in t7rl.FEATS}})
    models = (_Fixed(lambda o: 10), _Fixed(lambda o: 0), _Fixed(lambda o: 5), _Fixed(lambda o: 5),
              _Fixed(lambda o: 10 if o[-1] < 0.5 else 0))  # 마지막은 장부 비중을 본다
    pol = t7rl.FoldPolicy(2026, models, np.zeros(10), np.ones(10))
    out = t7rl.exposure_path(px, feats, lambda d: pol, np.zeros(len(days)), rate=0.0, tax=0.0, w0=0.466107)
    assert out["book_weight"].iloc[0] == pytest.approx(0.466107)
    assert out["exposure"].iloc[0] == pytest.approx((1.0 + 0.0 + 0.5 + 0.5 + 1.0) / 5)
    # 다음 결정일 장부 비중 = 직전 E (가격 변화·비용 없음)
    assert out["book_weight"].iloc[1] == pytest.approx(out["exposure"].iloc[0])
    assert out["exposure"].iloc[1] == pytest.approx((1.0 + 0.0 + 0.5 + 0.5 + 0.0) / 5)
    assert list(out.columns) == ["date", "fold", "book_weight", "a0", "a1", "a2", "a3", "a4", "exposure"]
    # 표준화 ±5 자르기와 관측 길이 11
    o = pol.observation(np.full(10, 100.0), 0.3)
    assert o.shape == (11,) and float(o[:10].max()) == t7rl.CLIP and o[-1] == np.float32(0.3)
    bad = feats.copy()
    bad.loc[1, "g_vix"] = np.nan
    with pytest.raises(ValueError):
        t7rl.exposure_path(px, bad, lambda d: pol, np.zeros(len(days)), 0.0, 0.0)


def test_expand_and_chain_basket():
    assert t7rl.expand([0.1, 0.2, 0.3], 5) == [0.1, 0.1, 0.2, 0.2, 0.3]
    with pytest.raises(ValueError):
        t7rl.expand([0.1, 0.2], 5)
    frozen = pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-12-29", "2025-12-30"]))
    cal = pd.to_datetime(["2025-12-29", "2025-12-30", "2026-01-02", "2026-01-05", "2026-01-06"])
    e1 = pd.Series([0.99, 1.1], index=pd.to_datetime(["2026-01-05", "2026-01-06"]))
    px = t7rl.chain_basket(frozen, e1, cal)
    assert px.tolist() == [1.0, 2.0, 2.0, 1.98, 2.2]  # 첫 체결 전은 그대로, 뒤는 마지막 고정 가격 × NAV


def test_build_positions_hold_week_keeps_weights(cfg, risk_params):
    buffer = t7rl.t7_buffer(BufferRuleParams.from_prereg(cfg))
    snaps = [make_snapshot("2026-01-02", n=100), make_snapshot("2026-01-09", n=100, crash=(1,)),
             make_snapshot("2026-01-16", n=100)]
    eng = ScriptedEngine()
    pos = t7rl.build_positions(snaps, eng, buffer, risk_params, [0.5, 0.5, 0.3])
    w = {d: g[g["ticker"] != "CASH"].set_index("ticker")["target_weight"] for d, g in pos.groupby("date")}
    d0, d1, d2 = sorted(w)
    assert len(w[d0]) == 80 and np.allclose(w[d0], 0.5 / 80)  # 리밸런싱 주: 같은 비중 × E
    # 사이 주: E를 다시 곱하지 않고 들고 있는 비중 그대로, crash_flag 종목(rank 1)만 판다
    assert "000001" not in w[d1].index and len(w[d1]) == 79 and np.allclose(w[d1], 0.5 / 80)
    assert np.allclose(w[d2], 0.3 / len(w[d2]))  # 다음 리밸런싱 주에 새 E
    assert len(eng.calls) == 2
    with pytest.raises(ValueError):
        t7rl.build_positions(snaps, ScriptedEngine(), buffer, risk_params, [0.5, 0.5])


def test_verify_files_detects_change(tmp_path):
    (tmp_path / "models").mkdir()
    f = tmp_path / "models" / "a.bin"
    f.write_bytes(b"policy")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"models/a.bin": t7rl.sha256(f)}}), encoding="utf-8")
    t7rl.verify_files(["models/a.bin"], tmp_path)
    f.write_bytes(b"changed")
    with pytest.raises(ValueError):
        t7rl.verify_files(["models/a.bin"], tmp_path)
    with pytest.raises(ValueError):
        t7rl.verify_files(["models/none.bin"], tmp_path)


@pytest.mark.parametrize("allow, run_date", [(False, date(2026, 12, 14)), (True, date(2026, 11, 30))])
def test_run_holdout_blocked_before_open(monkeypatch, allow, run_date):
    def no_read(*a, **k):
        raise AssertionError("가드보다 먼저 자료를 읽었다")

    monkeypatch.setattr(t7rl.io, "read_table", no_read)
    monkeypatch.setattr(t7rl, "frozen_inputs", no_read)
    monkeypatch.setattr(t7rl, "load_policy", no_read)
    with pytest.raises(GuardViolation):
        t7rl.run_holdout("20261211_A_holdout", None, {}, pd.Series(dtype=float), allow_holdout=allow, run_date=run_date)


def test_cd91_from_fred_csv(tmp_path):
    f = tmp_path / "IR3TIB01KRM156N.csv"
    f.write_text("observation_date,IR3TIB01KRM156N\n2025-11-01,2.54\n2025-12-01,.\n2026-01-01,2.80\n", encoding="utf-8")
    s = t7rl.cd91_from_fred_csv(f)
    assert list(s.index.astype(str)) == ["2025-11", "2026-01"]  # 빈 값('.')은 뺀다
    assert math.isclose(float(s[pd.Period("2026-01", freq="M")]), 0.028)
