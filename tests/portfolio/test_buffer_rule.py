"""작업 B-03: 6.5절 버퍼 규칙 단계 1~5와 M5 엔진 위 실행."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.portfolio.buffer_rule import buffer_rule_weights, buffer_targets, run_buffer_rule
from src.portfolio.interfaces import PeriodResult
from src.portfolio.params import BufferRuleParams
from src.portfolio.risk_layer import CASH
from src.portfolio.rl_data import build_snapshots
from tests.portfolio.fakes import FakeEngine, ScriptedEngine, make_snapshot


@pytest.fixture(scope="module")
def buffer(cfg) -> BufferRuleParams:
    return BufferRuleParams.from_prereg(cfg)


def _tk(rank: int) -> str:
    return f"{rank:06d}"


def _signals(n: int = 60, crash: tuple[int, ...] = ()) -> pd.DataFrame:
    """종목코드 000001~이 rank 1~n인 signals. crash는 rank 번호로 준다."""
    rank = np.arange(1, n + 1)
    return pd.DataFrame({"ticker": [_tk(r) for r in rank], "rank": rank, "crash_flag": np.isin(rank, crash).astype(np.int8)})


def _weights(holdings=(), crash=(), cooldown=(), universe=None, buffer=None) -> pd.Series:
    sig = _signals(crash=tuple(crash))
    members = sig["ticker"] if universe is None else universe
    return buffer_rule_weights([_tk(r) for r in holdings], sig, members, [_tk(r) for r in cooldown], buffer)


def test_empty_holdings_buys_top_20_equally(buffer):
    w = _weights(buffer=buffer)  # 단계 3·4
    assert list(w.index) == [_tk(r) for r in range(1, 21)]
    assert np.allclose(w, 1 / 20) and w.sum() == pytest.approx(1.0)


def test_holdings_are_kept_up_to_rank_40(buffer):
    """단계 2의 버퍼: rank 21~40 보유 종목은 유지하고, 41부터 판다. 같은 순위대의 미보유 종목은 사지 않는다."""
    w = _weights(holdings=(25, 40, 41), buffer=buffer)
    assert _tk(25) in w and _tk(40) in w and _tk(41) not in w
    assert _tk(30) not in w  # 보유하지 않은 rank 30은 추가 대상이 아니다 (rank ≤ 20만)
    assert list(w.index) == [_tk(r) for r in range(1, 19)] + [_tk(25), _tk(40)]  # 유지 2 + 추가 18 = 20
    assert np.allclose(w, 1 / 20)


def test_step2_sells_flagged_cooldown_outside_universe_and_unranked(buffer):
    universe = [_tk(r) for r in range(1, 61) if r != 7]  # rank 7은 U_t 밖
    w = buffer_rule_weights(
        [_tk(3), _tk(5), _tk(7), _tk(9), "999999"],  # 999999는 signals에 없다
        _signals(crash=(3,)), universe, [_tk(5)], buffer,
    )
    assert _tk(9) in w
    assert not {_tk(3), _tk(5), _tk(7), "999999"} & set(w.index)
    # 단계 3: rank ≤ 20 중 보유(7, 9)·crash(3)·쿨다운(5)을 뺀 16개를 추가한다
    assert len(w) == 17 and np.allclose(w, 1 / 17)


def test_step3_adds_only_what_is_available(buffer):
    w = _weights(crash=tuple(range(1, 16)), buffer=buffer)  # rank 1~15가 crash_flag = 1
    assert list(w.index) == [_tk(r) for r in range(16, 21)] and np.allclose(w, 1 / 5)


def test_no_candidate_is_all_cash(buffer):
    w = _weights(crash=tuple(range(1, 61)), buffer=buffer)
    assert w.empty and w.dtype == np.float64


def test_kept_holdings_count_toward_max_holdings(buffer):
    w = _weights(holdings=tuple(range(21, 41)), buffer=buffer)  # rank 21~40 20종목 보유
    assert list(w.index) == [_tk(r) for r in range(21, 41)]  # 모두 유지하고 추가하지 않는다


def test_duplicate_signals_are_rejected(buffer):
    sig = pd.concat([_signals(), _signals().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        buffer_rule_weights([], sig, sig["ticker"], [], buffer)


def test_step5_goes_through_m8(buffer, risk_params):
    """단계 5: 업종 한도 0.30, 노출 E_t가 그대로 적용되고, crash_flag·쿨다운 종목은 사지 않는다."""
    snap = make_snapshot(crash=(1,), cooldown=(2,), sectors=["X"] * 60, exposure=0.8)
    targets, cash, exposure = buffer_targets(snap, [], buffer, risk_params)
    assert sorted(targets) == [_tk(r) for r in range(3, 21)]  # 18종목
    assert sum(targets.values()) == pytest.approx(risk_params.sector_cap * 0.8)  # 한 업종 1.0 → 0.30 → × E_t
    assert cash == pytest.approx(1.0 - 0.30 * 0.8) and exposure == 0.8


def test_step5_modes(buffer, risk_params):
    snap = make_snapshot(cooldown=(2,), sectors=["X"] * 60, exposure=0.8)
    off, cash_off, e_off = buffer_targets(snap, [], buffer, risk_params, mode="off")  # G7: M8·M9 미적용
    assert _tk(2) in off and sum(off.values()) == pytest.approx(1.0) and cash_off == pytest.approx(0.0) and e_off == 1.0
    no_e, _, e = buffer_targets(snap, [], buffer, risk_params, mode="no_exposure")  # G4 진단: E_t = 1
    assert sum(no_e.values()) == pytest.approx(risk_params.sector_cap) and e == 1.0


def test_holdings_come_from_the_engine(buffer, risk_params):
    """단계 1: H는 엔진이 돌려준 실제 보유다. 목표에 없던 종목도 엔진이 보유 중이면 단계 2로 판단한다."""

    def script(k: int, targets) -> PeriodResult:
        held = {_tk(30): 0.5} if k == 0 else dict(targets)  # 첫 구간 뒤 엔진은 rank 30 하나만 보유
        return PeriodResult(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, held, 1.0 - sum(held.values()), None)

    engine = ScriptedEngine(script)
    snaps = [make_snapshot(date=d) for d in ("2020-01-03", "2020-01-10", "2020-01-17")]
    run = run_buffer_rule(snaps, engine, buffer, risk_params)
    week2 = set(engine.calls[1][0])
    assert _tk(30) in week2 and week2 == {_tk(r) for r in range(1, 20)} | {_tk(30)}
    assert len(engine.calls) == 2  # 마지막 신호일은 체결하지 않는다
    assert sorted(run.positions["date"].astype(str).unique()) == ["2020-01-03", "2020-01-10", "2020-01-17"]


@pytest.fixture(scope="module")
def snaps(market, risk_params):
    return build_snapshots(
        market.signal_dates, market.signals, market.features, market.universe, market.sector, market.bm_close,
        lambda d: market.cooldown[d], risk_params,
    )


def test_runs_on_engine_over_all_signal_dates(snaps, market, buffer, risk_params, cfg):
    """B-03 완료 기준 '엔진 위에서 실행된다'. C의 M5가 병합되기 전이라 같은 형태의 FakeEngine을 쓴다 (B-05에서 교체)."""
    run = run_buffer_rule(snaps, FakeEngine(market.closes, market.bm_close, cfg["costs"]), buffer, risk_params)
    pos = run.positions
    assert list(pos.columns) == ["date", "ticker", "target_weight", "constructor", "exposure", "regime_mc"]
    assert pos["date"].nunique() == len(snaps) and (pos["constructor"] == "buffer").all()
    assert len(run.excess_returns) == len(run.turnover) == len(snaps) - 1 and np.isfinite(run.excess_returns).all()
    assert np.allclose(pos.groupby("date")["target_weight"].sum(), 1.0, atol=1e-9)
    stocks = pos[pos["ticker"] != CASH]
    assert (stocks["target_weight"] <= risk_params.stock_cap + 1e-9).all()
    assert stocks.groupby("date").size().max() <= buffer.max_holdings
    for snap in snaps:
        day = stocks.loc[stocks["date"] == pd.Timestamp(snap.date), "ticker"]
        assert not set(snap.tickers[snap.crash_flag | snap.cooldown]) & set(day)  # crash_flag·쿨다운 종목은 담지 않는다


def test_runs_on_c_execution_engine(snaps, market, buffer, risk_params, cfg):
    """같은 실행을 C의 실제 M5 엔진(PR #8)에서 한다. 엔진이 아직 병합되지 않았으면 건너뛴다."""
    engine = pytest.importorskip("src.backtest.engine")
    accounting = pytest.importorskip("src.backtest.accounting")
    c = cfg["costs"]
    rates = accounting.CostRates(c["sell_tax"], c["commission_per_side"], c["slippage_per_side"])  # 기본 실행은 배율 1
    real = run_buffer_rule(snaps, engine.ExecutionEngine(market.closes, market.bm_close, rates), buffer, risk_params)
    fake = run_buffer_rule(snaps, FakeEngine(market.closes, market.bm_close, c), buffer, risk_params)
    pd.testing.assert_frame_equal(real.positions, fake.positions)  # 같은 신호, 같은 M8, 같은 보유 → 같은 목표 비중
    # 목표 합이 1인 체결에서만 다르다. FakeEngine은 현금을 음수로 두고, M5는 매수를 같은 비율로 줄인다
    assert np.allclose(real.excess_returns, fake.excess_returns, atol=1e-3)
    assert np.allclose(real.turnover, fake.turnover, atol=1e-2)


def test_buffer_lowers_turnover(snaps, market, buffer, risk_params, cfg):
    """버퍼(rank 40까지 유지)가 없으면, 즉 rank 20을 넘는 즉시 팔면 회전율이 높아진다."""
    engine = FakeEngine(market.closes, market.bm_close, cfg["costs"])
    with_buffer = run_buffer_rule(snaps, engine, buffer, risk_params)
    no_buffer = run_buffer_rule(snaps, engine, dataclasses.replace(buffer, exit_rank_greater_than=20), risk_params)
    assert with_buffer.turnover[1:].mean() < no_buffer.turnover[1:].mean()
