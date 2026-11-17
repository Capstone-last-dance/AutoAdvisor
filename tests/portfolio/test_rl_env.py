"""RL 환경 (6.5절 상태, 행동, 보상)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from src.portfolio.interfaces import PeriodResult
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_env import PortfolioEnv
from tests.portfolio.fakes import FakeEngine, ScriptedEngine, make_snapshot

N_SLOT_OBS = 60 * 6


def _dates(n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-03", periods=n, freq="W-FRI")]


def _snaps(count: int, **kwargs):
    return [make_snapshot(date=d, **kwargs) for d in _dates(count)]


def _held(weights: dict[str, float], cash: float = 0.5, turnover: float = 0.0):
    """보유 상태를 돌려주는 엔진 스크립트."""
    return lambda i, targets: PeriodResult(1.0, 1.0, 1.0, 0.0, 0.0, turnover, weights, cash, None)


@pytest.fixture(scope="module")
def train_snaps(market, risk_params):
    dates = [d for d in market.signal_dates if d < "2013-01-01"]
    return build_snapshots(
        dates, market.signals, market.features, market.universe, market.sector, market.bm_close,
        lambda d: market.cooldown[d], risk_params,
    )


# 행동 범위 [−10, 10]([결정 필요] D3)와 무한 관측 범위(mkt_vol_20, mkt_ret_20)에 대한 권고 경고는 의도한 것이다
@pytest.mark.filterwarnings("ignore:We recommend you to use a symmetric and normalized Box action space")
@pytest.mark.filterwarnings("ignore:.*For Box action spaces, we recommend using a symmetric and normalized space")
@pytest.mark.filterwarnings("ignore:.*A Box observation space (minimum|maximum) value is")
def test_env_passes_sb3_and_gymnasium_checks(market, train_snaps, rl_params, risk_params, cfg):
    engine = FakeEngine(market.closes, market.bm_close, cfg["costs"])
    sb3_check_env(PortfolioEnv(train_snaps, engine, rl_params, risk_params, lam=1.0))
    gym_check_env(PortfolioEnv(train_snaps, engine, rl_params, risk_params, lam=1.0), skip_render_check=True)


def test_observation_layout_at_start(rl_params, risk_params):
    snap = make_snapshot()
    env = PortfolioEnv([snap, make_snapshot(date="2020-01-10")], ScriptedEngine(), rl_params, risk_params, 0.0, "full")
    obs, _ = env.reset()
    assert obs.shape == (rl_params.slots * 6 + 6,) and obs.dtype == np.float32
    slot = obs[:N_SLOT_OBS].reshape(60, 6)
    np.testing.assert_allclose(slot[:40, 0], snap.rank_score_pct[:40], rtol=1e-6)
    np.testing.assert_allclose(slot[:40, 1], snap.crash_prob[:40], rtol=1e-6)
    np.testing.assert_allclose(slot[:40, 3], snap.vol_20[:40], rtol=1e-6)
    assert (slot[:40, [2, 4]] == 0.0).all() and (slot[:40, 5] == 1.0).all()
    assert (slot[40:] == 0.0).all()
    np.testing.assert_allclose(obs[N_SLOT_OBS:], [1.0, 0.0, 0.15, 0.01, 0.0, 1.0], rtol=1e-6)


def test_slots_add_holdings_and_put_out_of_universe_last(rl_params, risk_params):
    weights = {"000002": 0.05, "000050": 0.04, "999999": 0.03}
    engine = ScriptedEngine(_held(weights, cash=0.88, turnover=0.12))
    env = PortfolioEnv(_snaps(3), engine, rl_params, risk_params, 0.0, "full")
    env.reset()
    obs, *_ = env.step(np.zeros(61, dtype=np.float32))
    slot = obs[:N_SLOT_OBS].reshape(60, 6)
    assert env._slot_tickers == [f"{i:06d}" for i in range(1, 41)] + ["000050", "999999"]
    assert slot[1, 2] == pytest.approx(0.05) and slot[1, 4] == 1.0
    assert slot[40, 0] == pytest.approx((60 - 50 + 1) / 60) and slot[40, 2] == pytest.approx(0.04)
    np.testing.assert_allclose(slot[41], [0.0, 0.0, 0.03, 0.0, 1.0, 1.0], rtol=1e-6)
    assert (slot[42:] == 0.0).all()
    np.testing.assert_allclose(obs[N_SLOT_OBS : N_SLOT_OBS + 2], [0.88, 0.12], rtol=1e-6)
    assert not env._tradable[41]


def test_masked_slots_get_no_weight(rl_params, risk_params):
    snaps = _snaps(3, crash=(1, 2), cooldown=(3,))
    engine = ScriptedEngine(_held({"999999": 0.05}))
    env = PortfolioEnv(snaps, engine, rl_params, risk_params, 0.0, "full")
    env.reset()
    env.step(np.zeros(61, dtype=np.float32))
    action = np.append(np.where(env._tradable, 0.0, 10.0), 0.0).astype(np.float32)  # 마스킹 대상에 큰 로짓
    _, _, _, _, info = env.step(action)
    # 남은 37개 종목과 현금이 같은 비중 1/38. 앞 슬롯부터 20개 (rank 4~23)
    assert set(info["targets"]) == {f"{i:06d}" for i in range(4, 24)}
    assert all(w == pytest.approx(1 / 38) for w in info["targets"].values())


def test_top_k_truncation_and_min_weight(rl_params, risk_params):
    env = PortfolioEnv(_snaps(2), ScriptedEngine(), rl_params, risk_params, 0.0, "full")
    env.reset()
    _, _, _, _, info = env.step(np.zeros(61, dtype=np.float32))
    # 슬롯 40개와 현금이 모두 1/41. 동률은 앞 슬롯 우선이라 rank 1~20만 남고 나머지는 현금
    assert set(info["targets"]) == {f"{i:06d}" for i in range(1, 21)}
    assert info["cash"] == pytest.approx(21 / 41)

    env.reset()
    action = np.full(61, -10.0, dtype=np.float32)
    action[0], action[60] = 5.0, 0.0
    _, _, _, _, info = env.step(action)
    # 슬롯 0만 min_weight 이상이고 종목 한도 0.10에서 잘린다
    assert info["targets"] == {"000001": pytest.approx(0.10)}


def test_all_masked_is_all_cash(rl_params, risk_params):
    env = PortfolioEnv(_snaps(2, crash=range(1, 61)), ScriptedEngine(), rl_params, risk_params, 0.0, "full")
    env.reset()
    _, _, _, _, info = env.step(np.full(61, 10.0, dtype=np.float32))
    assert info["targets"] == {} and info["cash"] == 1.0


def test_reward_formula(rl_params, risk_params):
    results = [
        PeriodResult(1.00, 1.05, 1.06, 0.05, 0.01, 0.2, {}, 1.0, None),  # DD 0 → 1 − 1.05/1.06
        PeriodResult(1.05, 1.00, 1.05, 1.00 / 1.05 - 1.0, -0.02, 0.1, {}, 1.0, None),  # DD 증가
        PeriodResult(1.00, 1.07, 1.07, 0.07, 0.03, 0.1, {}, 1.0, None),  # 새 고점, DD 감소는 보상 없음
    ]
    lam = 2.0
    env = PortfolioEnv(_snaps(4), ScriptedEngine(lambda i, _: results[i]), rl_params, risk_params, lam, "full")
    env.reset()
    rewards = [env.step(np.zeros(61, dtype=np.float32))[1] for _ in range(3)]
    dd0, dd1 = 1.0 - 1.05 / 1.06, 1.0 - 1.00 / 1.06
    assert rewards[0] == pytest.approx(0.05 - 0.01 - lam * dd0)
    assert rewards[1] == pytest.approx((1.00 / 1.05 - 1.0) + 0.02 - lam * (dd1 - dd0))
    assert rewards[2] == pytest.approx(0.07 - 0.03)


def test_random_episode_is_52_steps_inside_segment(rl_params, risk_params):
    dates = _dates(60)
    engine = ScriptedEngine()
    env = PortfolioEnv(_snaps(60), engine, rl_params, risk_params, 0.0, "random")
    starts = set()
    for seed in range(30):
        env.reset(seed=seed)
        starts.add(dates.index(env.current_date))
    assert starts <= set(range(0, 60 - 52)) and len(starts) > 1

    env.reset(seed=3)
    first = dates.index(env.current_date)
    engine.calls.clear()
    flags = []
    while True:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        flags.append((terminated, truncated, info["executed"]))
        if terminated or truncated:
            break
    assert len(flags) == 52 and flags[-1] == (False, True, True)
    assert all(f == (False, False, True) for f in flags[:-1])
    assert [c[1:] for c in engine.calls] == [(dates[i], dates[i + 1]) for i in range(first, first + 52)]

    env.reset(seed=3)
    assert dates.index(env.current_date) == first


def test_full_episode_acts_on_every_date_without_last_execution(rl_params, risk_params):
    engine = ScriptedEngine()
    env = PortfolioEnv(_snaps(5), engine, rl_params, risk_params, 0.0, "full")
    env.reset()
    flags = [env.step(np.zeros(61, dtype=np.float32))[3:5] for _ in range(5)]
    assert [truncated for truncated, _ in flags] == [False, False, False, False, True]
    assert [info["executed"] for _, info in flags] == [True, True, True, True, False]
    assert len(engine.calls) == 4


def test_purged_signal_date_is_never_stepped_over(rl_params, risk_params):
    """D3 (g): 한 스텝은 달력에서 바로 이어지는 두 신호일 사이다. purge로 빠진 신호일을 건너뛰지 않는다."""
    calendar = _dates(120)
    snaps = [make_snapshot(date=d) for d in calendar[:60] + calendar[61:]]  # 61번째 신호일이 purge됨
    engine = ScriptedEngine()
    env = PortfolioEnv(snaps, engine, rl_params, risk_params, 0.0, "random", calendar=calendar)
    starts = set()
    for seed in range(40):
        env.reset(seed=seed)
        starts.add(calendar.index(env.current_date))
        while not env.step(env.action_space.sample())[3]:
            pass
    assert starts <= set(range(0, 8)) | set(range(61, 68)) and min(starts) < 60 < max(starts)
    assert all(calendar.index(b) == calendar.index(a) + 1 for _, a, b in engine.calls)

    # full: 끊긴 자리에서는 체결하지 않고, 다음 신호일에 현금 100%로 다시 시작한다
    engine = ScriptedEngine(_held({"000001": 0.5}))
    env = PortfolioEnv(snaps[57:63], engine, rl_params, risk_params, 0.0, "full", calendar=calendar)
    env.reset()
    steps = []
    for _ in range(6):
        obs, _, _, _, info = env.step(np.zeros(61, dtype=np.float32))
        steps.append((info["executed"], float(obs[N_SLOT_OBS])))  # 공통 관측의 첫 값 = 현금 비중
    assert steps == [(True, 0.5), (True, 0.5), (False, 1.0), (True, 0.5), (True, 0.5), (False, 0.5)]
    assert [(calendar.index(a), calendar.index(b)) for _, a, b in engine.calls] == [(57, 58), (58, 59), (61, 62), (62, 63)]


def test_calendar_must_hold_the_snapshots_in_order(rl_params, risk_params):
    calendar = _dates(70)
    with pytest.raises(ValueError, match="달력에 없는"):
        PortfolioEnv(_snaps(3), ScriptedEngine(), rl_params, risk_params, 0.0, "full", calendar=calendar[1:])
    with pytest.raises(ValueError, match="오름차순"):
        backwards = [make_snapshot(date=d) for d in (calendar[2], calendar[1])]
        PortfolioEnv(backwards, ScriptedEngine(), rl_params, risk_params, 0.0, "full", calendar=calendar)
    with pytest.raises(ValueError, match="구간이 없다"):  # 69개지만 끊기지 않은 구간은 35개, 34개
        split = [make_snapshot(date=d) for d in calendar[:35] + calendar[36:]]
        PortfolioEnv(split, ScriptedEngine(), rl_params, risk_params, 0.0, "random", calendar=calendar)


def test_slot_overflow_keeps_every_holding(rl_params, risk_params):
    held = {f"{i:06d}": 0.01 for i in range(41, 66)}  # 체결 제약으로 25종목 보유
    env = PortfolioEnv(_snaps(3, n=70), ScriptedEngine(_held(held)), rl_params, risk_params, 0.0, "full")
    env.reset()
    env.step(np.zeros(61, dtype=np.float32))
    assert env._slot_tickers == [f"{i:06d}" for i in list(range(1, 36)) + list(range(41, 66))]


def test_build_snapshots_from_tables(market, train_snaps, risk_params):
    snap = train_snaps[0]
    u = market.universe
    members = u[(pd.to_datetime(u["date"]) == pd.Timestamp(snap.date)) & u["in_universe"]]
    assert set(snap.tickers) == set(members["ticker"]) and len(snap.tickers) == 60
    assert (np.diff(snap.rank) > 0).all() and snap.rank_score_pct.max() == 1.0
    assert snap.cooldown.sum() == 2
    unknown = np.array([int(t) % 10 == 1 for t in snap.tickers])  # 업종 정보 없는 종목 → '미분류' 하나
    assert len(set(snap.sector_code[unknown])) == 1
    assert not set(snap.sector_code[~unknown]) & set(snap.sector_code[unknown])

    features = market.features.copy()
    first = pd.to_datetime(features["date"]) == pd.Timestamp(snap.date)
    features.loc[first, "regime_mc"] = 1 - snap.regime_mc
    with pytest.raises(ValueError, match="regime_mc"):
        build_snapshots(
            [snap.date], market.signals, features, market.universe, market.sector, market.bm_close,
            lambda d: market.cooldown[d], risk_params,
        )
