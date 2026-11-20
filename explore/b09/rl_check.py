"""RL 하네스 대조.
1) 시험 운용: v7 공식 체크포인트(시드 0, 폴드 2016~2020)를 XEnv(x 없음)로 운용한 positions가
   공식 positions(wf_20261117_B_rl_wf_s0)의 2016~2020과 같은가. 지표와 G5 참고값도 적는다 (공식 RL 선별 기준값).
2) 학습: R0 폴드 2016 시드 0이 공식 체크포인트 지문(policy_sha256)과 같은가 (runs/R0 결과가 있을 때).
"""
import json
import sys

from common import REPO, SEARCH_YEARS, Ctx, brief, buffer_run
from rl_screen import RUNS, XEnv, g5_judgment
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.portfolio.params import RLParams
from src.portfolio.rl_train import DEVICE, rollout

ctx = Ctx(SEARCH_YEARS)
rl = RLParams.from_prereg(ctx.cfg)
buf = ctx.evaluate(buffer_run(ctx, ctx.snaps(), mode="full"), with_m9=True)
out = {"buffer_full": brief(buf)}
for seed in range(5):
    pols = {y: PPO.load(REPO / f"reports/20261117_B_rlwf_y{y}/seed_{seed}/best.zip", device=DEVICE) for y in ctx.years}
    env = XEnv(ctx.snaps(), ctx.engine(True, ledger=False), rl, ctx.risk, 0.0, "full", calendar=ctx.all_dates)
    pos = rollout(env, lambda d: pols[int(d[:4])]).positions
    off = pd.read_parquet(REPO / f"positions/wf_20261117_B_rl_wf_s{seed}.parquet")
    off = off[pd.to_datetime(off["date"]).isin(pd.to_datetime(ctx.dates))]
    k = ["date", "ticker"]
    a = pos.assign(date=pd.to_datetime(pos["date"])).sort_values(k).reset_index(drop=True)
    b = off.assign(date=pd.to_datetime(off["date"])).sort_values(k).reset_index(drop=True)
    same = len(a) == len(b) and bool((a[k].values == b[k].values).all())
    diff = float(np.abs(a["target_weight"].to_numpy() - b["target_weight"].to_numpy()).max()) if same else float("inf")
    s = ctx.evaluate(pos, with_m9=True)
    g5 = g5_judgment({seed: s}, buf, ctx.kpi)
    out[f"official_s{seed}"] = {"positions_same": same, "max_diff": diff, **brief(s), "ir_gain": g5["ir_gain"],
                                "turnover_ratio": g5["turnover_ratio"], "cagr_not_lower": g5["cagr_not_lower"]}
    print(seed, out[f"official_s{seed}"], flush=True)

r0 = RUNS / "R0" / "y2016_s0" / "result.json"
if r0.exists():
    m = json.loads((REPO / "reports/20261117_B_rlwf_y2016/metrics.json").read_text(encoding="utf-8"))
    off_fp = m["checkpoints"]["files"]["seed_0/best.zip"]["policy_sha256"]
    got = json.loads(r0.read_text(encoding="utf-8"))
    out["R0_training"] = {"policy_sha256_same": got["policy_sha256"] == off_fp, "got": got["policy_sha256"], "official": off_fp,
                          "best_val_ir": got["best_val_ir"], "best_timesteps": got["best_timesteps"]}
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
from common import HERE
(HERE / "rl_check.out.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
