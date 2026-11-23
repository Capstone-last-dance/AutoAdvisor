"""run_weekly 5b단계 B 병행 T7-80-C53 (src/portfolio/t7c53.py): forward 기록이 백테스트 규칙과 같은지."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.pipeline.run_weekly import StepContext
from src.portfolio import t7c53, t7rl, weekly
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.params import BufferRuleParams
from src.portfolio.rl_data import build_snapshots
from tests.portfolio.fakes import market_tables
from tests.portfolio.test_weekly import PeriodAsForward

N_FORWARD = 7
SMALL_N, SMALL_BAND = 20, 15  # 합성 시장은 80종목이라 보유 수를 줄여 버퍼 규칙이 실제로 고르게 한다


@dataclass
class RunCtx(StepContext):
    """io 대신 메모리 표를 읽고 쓴다. 병행 기록은 run_id로 쓴다."""

    tables: dict = field(default_factory=dict)
    written: dict = field(default_factory=dict)  # (name, run_id) → df

    def read_table(self, name, **kwargs):
        df = self.tables[name]
        if name == "signals":
            return df.loc[pd.to_datetime(df["date"]) == pd.Timestamp(kwargs["date"])].reset_index(drop=True)
        return df

    def write_table(self, name, df, **kwargs):
        self.written[(name, kwargs.get("run_id"))] = df
        return f"{name}/wf_{kwargs.get('run_id')}.parquet"


@pytest.fixture(scope="module")
def setup(cfg, risk_params, market):
    tables, kept = market_tables(market, halt_share=0.1)
    dates = kept.signal_dates[-N_FORWARD - 1:-1]
    fcfg = copy.deepcopy(cfg)
    fcfg["splits"]["forward_test"]["ml_signals_from"] = dates[0]
    sig = kept.signals.copy()
    sig["date"] = pd.to_datetime(sig["date"])
    sig["model_version"] = np.where(sig["date"] < pd.Timestamp(dates[0]), "baseline", "final_test")
    cooldown = lambda d: kept.cooldown.get(d, frozenset())  # noqa: E731
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(t7rl, "N", SMALL_N)
        mp.setattr(t7rl, "BAND", SMALL_BAND)
        buffer = t7rl.t7_buffer(BufferRuleParams.from_prereg(fcfg))
    fwd = sig[sig["date"].isin(pd.to_datetime(dates))].reset_index(drop=True)
    snaps = build_snapshots(dates, t7rl.smooth_signals(fwd), kept.features, kept.universe, kept.sector, kept.bm_close,
                            cooldown, risk_params)
    engine = build_engine(engine_inputs(tables["prices"], tables["status"], tables["index"], tables["delisted"]), fcfg)
    ref = t7rl.build_positions(snaps, engine, buffer, risk_params, [t7rl.E_C53] * len(dates))
    by_date = {d: g.reset_index(drop=True) for d, g in ref.groupby(ref["date"].dt.strftime("%Y-%m-%d"))}
    all_tables = {
        **tables,
        "delisted_reused": pd.DataFrame(columns=["ticker", "delist_date", "last_price"]).astype({"last_price": "float64"}),
        "signals": sig, "universe": kept.universe, "sector": kept.sector,
    }
    by_sig = {d.strftime("%Y-%m-%d"): g.reset_index(drop=True) for d, g in sig.groupby("date")}
    return SimpleNamespace(cfg=fcfg, risk=risk_params, dates=dates, cooldown=cooldown, tables=all_tables, ref=by_date,
                           signals=by_sig, earlier=kept.signal_dates[-N_FORWARD - 2])


@pytest.fixture
def wired(setup, monkeypatch):
    """PM·D 함수를 연결한 상태 (weekly 테스트와 같다)."""
    monkeypatch.setattr(t7rl, "N", SMALL_N)
    monkeypatch.setattr(t7rl, "BAND", SMALL_BAND)
    monkeypatch.setattr(weekly, "_load_params", lambda: (setup.cfg, BufferRuleParams.from_prereg(setup.cfg), setup.risk))
    monkeypatch.setattr(weekly, "M9_COOLDOWN_ENTRY", setup.cooldown)
    monkeypatch.setattr(weekly, "M9_ENGINE_ENTRY", lambda first, last: ({}, {}))
    real_build = weekly.build_engine

    def build(inputs, cfg, **kw):
        engine = real_build(inputs, cfg, **kw)
        return engine if hasattr(engine, "run_to_signal_close") else PeriodAsForward(engine)

    monkeypatch.setattr(weekly, "build_engine", build)
    monkeypatch.setattr(t7c53, "PAST_SIGNALS_ENTRY", lambda t, table: {d: s for d, s in setup.signals.items() if d < t})
    return setup


def _run(setup, date, recorded, monkeypatch):
    monkeypatch.setattr(t7c53, "PAST_PARALLEL_ENTRY", lambda t: recorded)
    ctx = RunCtx(date=date, tables=setup.tables, log=logging.getLogger("test_t7c53"))
    t7c53.step(ctx)
    return ctx


def test_forward_matches_the_backtest_rule(wired, monkeypatch):
    """매주 5b단계가 내는 병행 positions = 같은 신호일의 백테스트 T7 규칙 positions (E = 0.466107, 격주).
    지난 보유는 그 전 주까지의 병행 기록을 엔진으로 재생해 만든다 (거래정지로 목표와 실제 보유가 다른 주 포함)."""
    s = wired
    for k, t in enumerate(s.dates):
        ctx = _run(s, t, {d: s.ref[d] for d in s.dates[:k]}, monkeypatch)
        got = ctx.written[("positions", t7c53.run_id(t))].sort_values("ticker").reset_index(drop=True)
        want = s.ref[t].sort_values("ticker").reset_index(drop=True)
        assert got["ticker"].tolist() == want["ticker"].tolist(), t
        np.testing.assert_allclose(got["target_weight"], want["target_weight"], rtol=0, atol=1e-12)
        assert (got["constructor"] == "buffer").all() and np.allclose(got["exposure"], t7rl.E_C53)
        assert got["regime_mc"].iloc[0] == want["regime_mc"].iloc[0]
        assert got["target_weight"].sum() == pytest.approx(1.0, abs=1e-9)
        assert ctx.memory[t7c53.WRITTEN_KEY] == f"positions/wf_{t7c53.run_id(t)}.parquet"
    # 사이 주에는 E를 다시 곱하지 않는다: 사이 주 주식 합은 직전 리밸런싱 주 합에서 가격 변화만큼만 다르다
    held = [s.ref[d].loc[s.ref[d]["ticker"] != "CASH", "target_weight"].sum() for d in s.dates]
    assert all(h <= t7rl.E_C53 * 1.2 for h in held) and held[0] == pytest.approx(t7rl.E_C53, abs=1e-9)


def test_nothing_before_the_start_date(wired, monkeypatch):
    ctx = _run(wired, wired.earlier, {}, monkeypatch)
    assert ctx.written == {} and t7c53.WRITTEN_KEY not in ctx.memory


def test_baseline_signals_are_not_used(wired, monkeypatch):
    tables = dict(wired.tables)
    sig = tables["signals"].copy()
    sig.loc[sig["date"] == pd.Timestamp(wired.dates[0]), "model_version"] = "baseline"
    tables["signals"] = sig
    ctx = RunCtx(date=wired.dates[0], tables=tables, log=logging.getLogger("test_t7c53"))
    monkeypatch.setattr(t7c53, "PAST_PARALLEL_ENTRY", lambda t: {})
    with pytest.raises(ValueError, match="ML 신호"):
        t7c53.step(ctx)


def test_stub_inputs_skip_without_writing(wired, monkeypatch):
    ctx = RunCtx(date=wired.dates[0], tables=wired.tables, log=logging.getLogger("test_t7c53"))
    ctx.stub_tables.add("signals")
    ctx.memory[t7c53.WRITTEN_KEY] = "지난 실행 값"  # 이번 실행에서 쓰지 않았으면 표시가 남지 않는다
    monkeypatch.setattr(t7c53, "PAST_PARALLEL_ENTRY", lambda t: {})
    t7c53.step(ctx)
    assert ctx.written == {} and t7c53.WRITTEN_KEY not in ctx.memory


def test_missing_week_still_counts_for_the_biweekly_parity(wired, monkeypatch, caplog):
    """기록이 빠진 주도 센다: 3번째 신호일(2번)은 앞 주 기록이 없어도 리밸런싱 주다. 빠진 주는 직전 보유를 유지해 재생한다."""
    s = wired
    recorded = {s.dates[0]: s.ref[s.dates[0]]}  # 1번(사이 주) 기록이 없다
    with caplog.at_level(logging.INFO, logger="test_t7c53"):
        ctx = _run(s, s.dates[2], recorded, monkeypatch)
    df = ctx.written[("positions", t7c53.run_id(s.dates[2]))]
    stocks = df.loc[df["ticker"] != "CASH", "target_weight"]
    assert len(stocks) <= SMALL_N and stocks.sum() <= t7rl.E_C53 + 1e-12
    assert "리밸런싱" in caplog.text and "직전 보유를 유지" in caplog.text


def test_past_records_need_the_d_function(wired, monkeypatch):
    monkeypatch.setattr(t7c53, "PAST_PARALLEL_ENTRY", None)
    ctx = RunCtx(date=wired.dates[0], tables=wired.tables, log=logging.getLogger("test_t7c53"))
    t7c53.step(ctx)  # 첫 기록은 지난 기록이 필요 없다
    assert ("positions", t7c53.run_id(wired.dates[0])) in ctx.written
    ctx = RunCtx(date=wired.dates[1], tables=wired.tables, log=logging.getLogger("test_t7c53"))
    with pytest.raises(RuntimeError, match="D 함수"):
        t7c53.step(ctx)


def test_replay_rejects_records_on_non_signal_dates():
    with pytest.raises(ValueError, match="신호일이 아니다"):
        t7c53.replay_weights(object(), {"2026-11-28": pd.DataFrame()}, ["2026-11-27"], "2026-12-04")
