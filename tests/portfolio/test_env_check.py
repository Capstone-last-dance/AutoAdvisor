"""작업 B-05 완료 기준: RL 환경이 M5 엔진 위에서 끝까지 돌고, 같은 비중이면 버퍼 규칙과 체결·비용이 같다.

실제 베이스라인 신호(A-04)로는 A PR #1 병합 뒤 리포트에서 확인한다. 여기서는 9.3절 형식의 합성 표
(거래정지 칸, 보유 중 폐지 가능 종목 포함)를 어댑터로 C의 엔진에 넣는다.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.portfolio.buffer_rule import run_buffer_rule
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.env_check import compare_buffer_and_env, run_full_episode
from src.portfolio.params import BufferRuleParams
from src.portfolio.rl_data import build_snapshots
from tests.portfolio.fakes import market_tables

LAST_PRICE_DAY = "2012-06-15"  # 신호일(금요일)이다. 앞 구간(06-08 신호, 06-11 체결) 안에서 폐지된다


def _inputs(t):
    return engine_inputs(t["prices"], t["status"], t["index"], t["delisted"])


def _snapshots(m, risk):
    return build_snapshots(
        m.signal_dates, m.signals, m.features, m.universe, m.sector, m.bm_close, lambda d: m.cooldown[d], risk
    )


@pytest.fixture(scope="module")
def buffer(cfg) -> BufferRuleParams:
    return BufferRuleParams.from_prereg(cfg)


@pytest.fixture(scope="module")
def held_ticker(market, cfg, risk_params, buffer):
    """폐지 없이 LAST_PRICE_DAY까지 돌려 그날 보유한 종목. 그 종목을 폐지시키면 보유 중 폐지가 일어난다
    (폐지는 그날 뒤의 가격과 신호만 바꾸므로 그날까지의 실행은 같다)."""
    t, m = market_tables(market, delisted=None)
    snaps = _snapshots(m, risk_params)
    engine = build_engine(_inputs(t), cfg, record_ledger=True)
    run_buffer_rule(snaps[: m.signal_dates.index(LAST_PRICE_DAY) + 1], engine, buffer, risk_params)
    row = next(r for r in reversed(engine.ledger) if r["date"] == LAST_PRICE_DAY)
    return min(row["holdings"])


@pytest.fixture(scope="module")
def tables_and_market(market, held_ticker):
    return market_tables(market, delisted=held_ticker, last_price_day=LAST_PRICE_DAY)


@pytest.fixture(scope="module")
def inputs(tables_and_market):
    return _inputs(tables_and_market[0])


@pytest.fixture(scope="module")
def snaps(tables_and_market, risk_params):
    return _snapshots(tables_and_market[1], risk_params)


@pytest.mark.parametrize("policy", ["zero", "random"])
def test_full_episode_runs_to_the_end_on_c_engine(snaps, inputs, cfg, rl_params, risk_params, policy):
    """완료 기준 (1) (합성 신호). 첫 신호일부터 마지막 신호일까지 매 구간 엔진으로 체결한다."""
    rng = np.random.default_rng(0)
    act = None if policy == "zero" else (lambda obs: rng.uniform(-3, 3, rl_params.slots + 1).astype(np.float32))
    engine = build_engine(inputs, cfg)
    out = run_full_episode(snaps, engine, rl_params, risk_params, lam=1.0, policy=act)
    assert out["n_steps"] == len(snaps) and out["n_executed"] == len(snaps) - 1
    assert (out["first_date"], out["last_date"]) == (snaps[0].date, snaps[-1].date)
    assert out["rebalance_count"] == len(snaps) - 1  # 새 엔진이라 이 실행의 횟수다
    assert np.isfinite(out["nav_end"]) and out["nav_end"] > 0
    assert out["ms_per_step"] < 50  # 6절 예산의 여유 확인. 실제 크기 측정은 리포트에서 한다


def test_same_weights_give_same_fills_and_costs(snaps, inputs, cfg, rl_params, risk_params, buffer):
    """완료 기준 (2). 버퍼 규칙 실행과, 행동 해석 자리에 버퍼 비중을 넣은 RL 환경 실행이 같은 체결을 낸다."""
    out = compare_buffer_and_env(
        snaps, lambda: build_engine(inputs, cfg, record_ledger=True), rl_params, risk_params, buffer
    )
    assert out["mismatched_days"] == []
    assert out["n_periods"] == len(snaps) - 1 and out["ledger_days"] > 5 * out["n_periods"]  # 매 거래일 행
    for key in ("max_weight_diff", "max_fill_diff", "max_nav_diff", "max_excess_return_diff", "max_turnover_diff"):
        assert out[key] <= 1e-12, key
    assert out["ledger_counts"]["rejected_halted"] > 0  # 정지 칸에서 체결 거부가 실제로 일어났다
    assert out["ledger_counts"]["delisting_fills"] == 1  # 보유 중 폐지 청산이 한 번 일어났다


def test_comparison_detects_different_costs(snaps, inputs, cfg, rl_params, risk_params, buffer):
    """비교가 비어 있지 않은지: 두 번째 엔진만 비용 배율을 바꾸면 차이가 잡힌다."""
    engines = iter([build_engine(inputs, cfg, record_ledger=True), build_engine(inputs, cfg, stress=True, record_ledger=True)])
    out = compare_buffer_and_env(snaps, lambda: next(engines), rl_params, risk_params, buffer)
    assert out["max_fill_diff"] > 1e-6 or out["mismatched_days"]
