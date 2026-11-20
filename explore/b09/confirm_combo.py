"""G5·G7 확인 대상을 한 시스템으로 합친 확인 (11/19): M8 변형(G7 확인 2개) × RL 운용 변형(G5 확인 2개).

  python -I explore/b09/run.py confirm_combo.py

- 왜: 최종 모델은 M8 하나로 G5와 G7을 같이 넘어야 한다. G5 비교 대상은 '같은 신호·M8·M9를 쓴 사전등록 버퍼'라
  M8을 바꾸면 비교 버퍼도 같은 M8을 쓴다. G7은 그 버퍼와 M8·M9 미적용 버퍼를 비교한다 (confirm_g7과 같다).
- 두 쪽 모두 확인을 거친 변형을 바꾸지 않고 합친다. 합치기로 한 것은 2021~2025를 본 뒤라 결과에 '사후 조합'으로 적는다.
- RL 쪽 M8 변형: 정책 입력(관측의 exposure 등)은 그대로 두고, 체결 직전 M8에 쓰는 E_t와 Step 1 차단만 바꾼다.
  NAV 낙폭은 환경의 체결 후 NAV와 고점(일간 NAV 최댓값, 시작 NAV 1 포함)으로 잰다.
- 대조: 버퍼 + M8 변형의 MDD·CAGR이 confirm_g7.json과 같아야 한다.
- 결과: confirm_combo.json, 원장.
"""
import json
from dataclasses import replace

from common import ALL_YEARS, CONFIRM_YEARS, HERE, METRIC_KEYS, REPO, Ctx, Ledger, brief, buffer_run, g7_check
from rl_infer import VARIANTS
from rl_screen import XEnv, g5_judgment
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.portfolio.params import RLParams
from src.portfolio.rl_train import DEVICE, rollout

M8 = {  # confirm_g7.py와 같은 정의
    "M-dd-0.15-f0.3": {"dd": (0.15, 0.3)},
    "M-dd+tr+tag": {"trend": True, "dd": (0.15, 0.5), "block_v20": 0.8},
}
RL_PICKS = ("I-band0.01-smooth0.5", "I-smoothk-0.7")
RL_RUNS = (("공식 정책", {}),) + tuple((v, VARIANTS[v]) for v in RL_PICKS)
G5_KEYS = ("ir_gain", "turnover_ratio", "cagr_not_lower", "pass_by_ir", "pass_by_alternative", "pass")


class MEnv(XEnv):
    """XEnv + M8 변형 (E_t 조정·Step 1 vol_20 차단). m8이 비면 XEnv와 같다."""

    def __init__(self, *args, m8=None, bm=None, ma200=None, **kw):
        self.m8, self._bm, self._ma = dict(m8 or {}), bm, ma200
        super().__init__(*args, **kw)
        self._orig = list(self.snapshots)

    def reset(self, *, seed=None, options=None):
        self.snapshots = list(self._orig)
        return super().reset(seed=seed, options=options)

    def _exposure(self, snap) -> float:
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
        if self.m8.get("trend") or self.m8.get("dd"):
            snap = self._orig[self._i]
            self.snapshots[self._i] = replace(snap, exposure=self._exposure(snap))
        return super().step(action)


def buffer_fns(ctx, spec):
    bm = ctx.inputs.bm_close
    ma200 = bm.rolling(200, min_periods=200).mean()

    def exposure_fn(i, snap, e, nav, chosen):
        if spec.get("trend") and bm[pd.Timestamp(snap.date)] < ma200[pd.Timestamp(snap.date)]:
            e = min(e, 0.3)
        if spec.get("dd") and len(nav):
            th, factor = spec["dd"]
            if 1.0 - float(nav.iloc[-1]) / float(nav.max()) > th:
                e *= factor
        return e

    q = spec.get("block_v20")
    block_fn = (lambda i, snap: snap.vol_20 > q) if q is not None else None
    return exposure_fn, block_fn, bm, ma200


ledger = Ledger(HERE / "ledger.jsonl")
for m in M8:
    for r in RL_PICKS:
        ledger.plan(f"{m} × {r}", "G5·G7-조합", "확인된 M8 변형과 RL 운용 변형을 바꾸지 않고 합침",
                    "최종 모델은 M8 하나로 G5·G7을 같이 넘어야 한다 (사후 조합, 2021~2025를 본 뒤 정함)")
g7_ref = json.loads((HERE / "confirm_g7.json").read_text(encoding="utf-8"))

out = {}
for years, label in ((CONFIRM_YEARS, "2021-2025"), (ALL_YEARS, "2016-2025")):
    ctx = Ctx(years)
    rl = RLParams.from_prereg(ctx.cfg)
    snaps = ctx.snaps()
    off = brief(ctx.evaluate(buffer_run(ctx, snaps, mode="off"), with_m9=False))
    pols = {(y, s): PPO.load(REPO / f"reports/20261117_B_rlwf_y{y}/seed_{s}/best.zip", device=DEVICE)
            for y in ctx.years for s in range(5)}
    res = {"buffer_off": off}
    for m, spec in M8.items():
        exposure_fn, block_fn, bm, ma200 = buffer_fns(ctx, spec)
        buf = brief(ctx.evaluate(buffer_run(ctx, snaps, mode="full", exposure_fn=exposure_fn, block_fn=block_fn), with_m9=True))
        g7 = g7_check(buf, off, ctx.kpi)
        ref = g7_ref[label][m]
        block = {"buffer": buf, "g7_mdd_red": g7["mdd_reduction_rel"], "g7_cagr_gap": buf["cagr"] - off["cagr"],
                 "g7_pass": g7["pass"], "diff_vs_confirm_g7": {k: buf[k] - ref[k] for k in ("mdd", "cagr")}}
        for name, x in RL_RUNS:
            per = {}
            for s in range(5):
                env = MEnv(snaps, ctx.engine(True, ledger=False), rl, ctx.risk, 0.0, "full", calendar=ctx.all_dates,
                           x=x, m8=spec, bm=bm, ma200=ma200)
                per[s] = ctx.evaluate(rollout(env, lambda d, s=s: pols[(int(d[:4]), s)]).positions, with_m9=True)
            g5 = g5_judgment(per, buf, ctx.kpi)
            mean = {k: float(np.mean([brief(v)[k] for v in per.values()])) for k in METRIC_KEYS}
            ref7 = g7_check(mean, off, ctx.kpi)  # 참고: RL 시스템의 MDD 감소율 (공식 G7 정의는 버퍼 기준)
            row = {"mean": mean, **{k: g5[k] for k in G5_KEYS},
                   "ref_rl_mdd_red_vs_off": ref7["mdd_reduction_rel"], "ref_rl_cagr_gap_vs_off": mean["cagr"] - off["cagr"],
                   "seeds": {s: brief(v) for s, v in per.items()}}
            block[name] = row
            if name in RL_PICKS:
                ledger.result(f"{m} × {name}", years=label, seeds=5, **{k: row[k] for k in G5_KEYS}, **mean,
                              g7_pass_buffer=g7["pass"], g7_mdd_red=g7["mdd_reduction_rel"])
            print(label, m, name, json.dumps({k: row[k] for k in ("mean",) + G5_KEYS}, ensure_ascii=False, default=float), flush=True)
        res[m] = block
        print(label, m, "buffer", json.dumps({k: block[k] for k in ("buffer", "g7_mdd_red", "g7_cagr_gap", "g7_pass", "diff_vs_confirm_g7")},
                                             ensure_ascii=False, default=float), flush=True)
    out[label] = res
    del ctx, pols
(HERE / "confirm_combo.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("done")
