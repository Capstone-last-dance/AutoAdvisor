"""M4 RL 정책의 거래 환경 (6.5절, B 소유).

신호일 1개가 1스텝이다. 관측에는 신호일 t에 알 수 있는 정보만 쓰고 (R-03), 행동 해석으로 얻은
w_raw는 M8을 거쳐 체결한다 (R-06). 체결, 비용, NAV는 M5 엔진(ExecutionEngine, C)이 계산한다.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.portfolio.interfaces import ExecutionEngine
from src.portfolio.params import RiskLayerParams, RLParams
from src.portfolio.risk_layer import check_invariants, risk_layer_weights
from src.portfolio.rl_data import SignalSnapshot

# 관측 순서 (6.5절 표와 같은 순서)
SLOT_FEATURES = ("rank_score_pct", "crash_prob", "weight", "vol_20", "held", "valid")
COMMON_FEATURES = ("cash", "turnover", "mkt_vol_20", "mkt_ret_20", "regime_mc", "exposure")

# 부록 1에 없는 구현 세부값. docs/design_B.md [결정 필요] D3에서 확정하기 전까지의 잠정값이다.
ACTION_BOUND = 10.0  # 행동(로짓) 범위 [−10, 10]. SB3 check_env가 유한한 범위를 요구한다
OUT_OF_UNIVERSE_FILL = 0.0  # U_t 밖 보유 종목 슬롯의 rank_score 백분위, crash_prob, vol_20

Episode = Literal["random", "full"]


class PortfolioEnv(gym.Env):
    """episode="random": 학습용. 시작 신호일을 균등 추첨해 episode_steps 스텝을 진행한다.
    episode="full": 검증·테스트용. 첫 신호일에 현금 100%로 시작해 모든 신호일에서 행동한다.
    마지막 신호일에는 다음 신호일이 없으므로 비중만 기록하고 체결하지 않는다.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        snapshots: Sequence[SignalSnapshot],
        engine: ExecutionEngine,
        rl: RLParams,
        risk: RiskLayerParams,
        lam: float,
        episode: Episode = "random",
    ) -> None:
        super().__init__()
        if episode not in ("random", "full"):
            raise ValueError(f"episode = {episode!r}")
        n_min = rl.episode_steps + 1 if episode == "random" else 1
        if len(snapshots) < n_min:
            raise ValueError(f"신호일이 {len(snapshots)}개다. {n_min}개 이상 필요하다")
        self.snapshots = list(snapshots)
        self.engine = engine
        self.rl = rl
        self.risk = risk
        self.lam = float(lam)
        self.episode = episode
        n_obs = rl.slots * len(SLOT_FEATURES) + len(COMMON_FEATURES)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-ACTION_BOUND, ACTION_BOUND, shape=(rl.slots + 1,), dtype=np.float32)

    @property
    def current_date(self) -> str:
        return self.snapshots[self._i].date

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self.episode == "random":
            # 마지막 스텝의 보상 구간(다음 신호일)까지 snapshots 안에 들어가는 시작점만 뽑는다
            self._i = int(self.np_random.integers(0, len(self.snapshots) - self.rl.episode_steps))
            self._end = self._i + self.rl.episode_steps
        else:
            self._i = 0
            self._end = len(self.snapshots)
        self._state = self.engine.initial_state()
        self._weights: dict[str, float] = {}
        self._cash = 1.0
        self._turnover = 0.0
        self._peak = 1.0
        self._dd = 0.0
        self._obs = self._observe()
        return self._obs, {"date": self.current_date}

    def step(self, action: np.ndarray):
        snap = self.snapshots[self._i]
        w_raw = self._interpret(np.asarray(action, dtype=np.float64))
        w = risk_layer_weights(w_raw, ~self._tradable, self._slot_sector, snap.exposure, self.risk)  # M8
        cash = check_invariants(w, self._slot_sector, self.risk)  # G0
        targets = {self._slot_tickers[j]: float(w[j]) for j in np.flatnonzero(w > 0.0)}
        info: dict[str, Any] = {
            "date": snap.date,
            "targets": targets,
            "cash": cash,
            "exposure": snap.exposure,
            "regime_mc": snap.regime_mc,
            "executed": False,
        }
        reward = 0.0
        nxt = self._i + 1
        if nxt < len(self.snapshots):
            res = self.engine.run_period(self._state, targets, snap.date, self.snapshots[nxt].date)
            # r = R_net − R_BM − λ × max(0, DD(t′) − DD(t)). DD 고점은 일간 NAV 기준 ([결정 필요] D2)
            self._peak = max(self._peak, res.nav_max, res.nav_end)
            dd = 1.0 - res.nav_end / self._peak
            reward = res.r_net - res.r_bm - self.lam * max(0.0, dd - self._dd)
            self._state, self._dd = res.state, dd
            self._weights, self._cash, self._turnover = dict(res.weights_obs), res.cash_obs, res.turnover
            info.update(
                executed=True,
                r_net=res.r_net,
                r_bm=res.r_bm,
                excess_return=res.r_net - res.r_bm,
                dd=dd,
                nav=res.nav_end,
                turnover=res.turnover,
            )
        self._i = nxt
        # 52스텝 종료는 MDP의 종착 상태가 아니라 표본 추출의 절단이다 ([결정 필요] D3)
        truncated = self._i >= self._end
        if self._i < len(self.snapshots):
            self._obs = self._observe()
        return self._obs, float(reward), False, truncated, info

    def _observe(self) -> np.ndarray:
        """신호일 t의 관측 (6.5절). 후보 슬롯 C_t = (rank ≤ candidate_top_k) ∪ (보유 종목)."""
        snap = self.snapshots[self._i]
        n_slots = self.rl.slots
        held = self._weights
        held_in = np.array(sorted(snap.index[t] for t in held if t in snap.index), dtype=np.int64)
        held_out = sorted(t for t in held if t not in snap.index)  # U_t 밖 보유 종목은 맨 뒤
        cand = np.union1d(np.flatnonzero(snap.rank <= self.rl.candidate_top_k), held_in)  # rank 오름차순
        n_over = len(cand) + len(held_out) - n_slots
        if n_over > 0:
            # 체결 제약으로 보유가 max_holdings를 넘은 경우: 보유하지 않은 후보를 rank가 나쁜 순서로 뺀다 (D3)
            nonheld = cand[~np.isin(cand, held_in)]
            if n_over > len(nonheld):
                raise RuntimeError(f"{snap.date}: 보유 종목 {len(held)}개가 슬롯 {n_slots}개를 넘는다")
            cand = cand[~np.isin(cand, nonheld[len(nonheld) - n_over :])]
        n_in, n_valid = len(cand), len(cand) + len(held_out)
        tickers = [str(t) for t in snap.tickers[cand]] + held_out

        slot = np.zeros((n_slots, len(SLOT_FEATURES)))
        slot[:n_in, 0] = snap.rank_score_pct[cand]
        slot[:n_in, 1] = snap.crash_prob[cand]
        slot[:n_valid, 2] = [held.get(t, 0.0) for t in tickers]
        slot[:n_in, 3] = snap.vol_20[cand]
        slot[:n_valid, 4] = [t in held for t in tickers]
        slot[:n_valid, 5] = 1.0
        slot[n_in:n_valid, [0, 1, 3]] = OUT_OF_UNIVERSE_FILL

        self._slot_tickers = tickers
        self._tradable = np.zeros(n_slots, dtype=bool)
        self._tradable[:n_in] = ~(snap.crash_flag[cand] | snap.cooldown[cand])
        self._slot_sector = np.full(n_slots, snap.n_sector_codes, dtype=np.int64)  # U_t 밖·빈 슬롯: 비중 0
        self._slot_sector[:n_in] = snap.sector_code[cand]

        common = [self._cash, self._turnover, snap.mkt_vol_20, snap.mkt_ret_20, snap.regime_mc, snap.exposure]
        return np.concatenate([slot.ravel(), common]).astype(np.float32)

    def _interpret(self, action: np.ndarray) -> np.ndarray:
        """6.5절 행동 해석 (1)~(3). 반환값은 슬롯별 w_raw이고, 현금은 1 − 합이다."""
        n_slots = self.rl.slots
        # (1) 빈 슬롯, crash_flag = 1, M9 쿨다운, U_t 밖 종목의 로짓을 −∞로
        z = np.append(np.where(self._tradable, action[:n_slots], -np.inf), action[n_slots])
        # (2) softmax (현금 로짓은 마스킹하지 않으므로 최댓값이 유한하다)
        e = np.exp(z - z.max())
        w = e[:n_slots] / e.sum()
        # (3) 상위 max_holdings개만 남기고 min_weight 미만도 0. 동률이면 앞 슬롯(rank가 좋은 종목) 우선
        order = np.lexsort((np.arange(n_slots), -w))
        keep = np.zeros(n_slots, dtype=bool)
        keep[order[: self.rl.max_holdings]] = True
        keep &= w >= self.rl.min_weight
        return np.where(keep, w, 0.0)
