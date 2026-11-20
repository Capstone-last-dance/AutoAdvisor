"""M8 변형 공용 (성능 트랙, 11/19). 정의는 confirm_combo.py·confirm_g7.py와 같고 E1(노출 조절 끔)을 더했다.

- RL 쪽(MEnv): 정책 입력(관측의 exposure 등)은 그대로 두고, 체결 직전 M8의 E_t와 Step 1 차단만 바꾼다.
  NAV 낙폭은 환경의 체결 후 NAV와 고점(일간 NAV 최댓값, 시작 NAV 1 포함)으로 잰다.
- 버퍼 쪽(buffer_m8): common.buffer_run의 노출·차단 고리. E1은 buffer_run(mode="no_exposure")
  (Step 4~5만 끄고 Step 1~3·M9는 켠다, 공식 `buffer_no_exposure`와 같은 모드).
"""
from dataclasses import replace

import pandas as pd

from common import buffer_run
from rl_screen import XEnv

M8 = {
    "M8 사전등록": {},
    "E1 노출 조절 끔": {"e1": True},
    "M-dd-0.15-f0.3": {"dd": (0.15, 0.3)},
    "M-dd+tr+tag": {"trend": True, "dd": (0.15, 0.5), "block_v20": 0.8},
}


class MEnv(XEnv):
    """XEnv + M8 변형. m8이 비면 XEnv와 같다."""

    def __init__(self, *args, m8=None, bm=None, ma200=None, **kw):
        self.m8, self._bm, self._ma = dict(m8 or {}), bm, ma200
        super().__init__(*args, **kw)
        self._orig = list(self.snapshots)

    def reset(self, *, seed=None, options=None):
        self.snapshots = list(self._orig)
        return super().reset(seed=seed, options=options)

    def _exposure(self, snap) -> float:
        if self.m8.get("e1"):
            return 1.0
        e = float(snap.exposure)
        if self.m8.get("trend"):
            t = pd.Timestamp(snap.date)
            if self._bm[t] < self._ma[t]:
                e = min(e, 0.3)
        if self.m8.get("dd"):
            th, factor = self.m8["dd"]
            if self._dd > th:
                e *= factor
        return e

    def _interpret(self, action):
        w = super()._interpret(action)
        q = self.m8.get("block_v20")
        if q is not None:  # step이 ~self._tradable을 M8 Step 1 차단으로 넘긴다
            snap = self._orig[self._i]
            for j, t in enumerate(self._slot_tickers):
                k = snap.index.get(t)
                if k is not None and snap.vol_20[k] > q:
                    self._tradable[j] = False
        return w

    def step(self, action):
        if self.m8.get("e1") or self.m8.get("trend") or self.m8.get("dd"):
            snap = self._orig[self._i]
            self.snapshots[self._i] = replace(snap, exposure=self._exposure(snap))
        return super().step(action)


def market(ctx):
    bm = ctx.inputs.bm_close
    return bm, bm.rolling(200, min_periods=200).mean()


def buffer_m8(ctx, snaps, spec: dict, **kw):
    """버퍼 규칙(사전등록, kw로 buffer·choose_fn을 바꿀 수 있다) + M8 변형 + M9의 positions."""
    if spec.get("e1"):
        return buffer_run(ctx, snaps, mode="no_exposure", with_m9=True, **kw)
    if not spec:
        return buffer_run(ctx, snaps, mode="full", **kw)
    bm, ma200 = market(ctx)

    def exposure_fn(i, snap, e, nav, chosen):
        t = pd.Timestamp(snap.date)
        if spec.get("trend") and bm[t] < ma200[t]:
            e = min(e, 0.3)
        if spec.get("dd") and len(nav):
            th, factor = spec["dd"]
            if 1.0 - float(nav.iloc[-1]) / float(nav.max()) > th:
                e *= factor
        return e

    q = spec.get("block_v20")
    block_fn = (lambda i, snap: snap.vol_20 > q) if q is not None else None
    return buffer_run(ctx, snaps, mode="full", exposure_fn=exposure_fn, block_fn=block_fn, **kw)
