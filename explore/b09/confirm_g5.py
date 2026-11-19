"""G5 트랙 확인 (11/19 선택 규칙으로 고른 2개): 2021~2025와 전체 10년, 5시드.

  python -I explore/b09/run.py confirm_g5.py

- 고른 두 변형은 운용 때만 해석을 바꾼 변형이라 v7 공식 체크포인트(`reports/20261117_B_rlwf_y<연도>/seed_<시드>/best.zip`)를
  그대로 쓴다 (재학습 없음). 비교 대상은 사전등록 버퍼 + 같은 신호·M8·M9 (선별과 같다).
- 대조: 해석을 바꾸지 않은 공식 정책. 10년 positions가 공식 `wf_20261117_B_rl_wf_s<시드>`와 같고, G5 값이 공식
  `20261118_B_buffer_wf` metrics.json과 같아야 한다 (차이를 결과에 적는다).
- 결과: confirm_g5.json, 원장. 변형 positions는 runs/confirm_g5/ (git 제외).
"""
import json

from common import ALL_YEARS, CONFIRM_YEARS, HERE, METRIC_KEYS, REPO, Ctx, Ledger, brief, buffer_run
from rl_infer import VARIANTS
from rl_screen import RUNS, XEnv, g5_judgment
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.portfolio.params import RLParams
from src.portfolio.rl_train import DEVICE, rollout

PICKS = ("I-band0.01-smooth0.5", "I-smoothk-0.7")
OFFICIAL = "공식 정책 (해석 그대로)"
OUT_POS = RUNS / "confirm_g5"
G5_KEYS = ("ir_gain", "turnover_ratio", "cagr_not_lower", "pass_by_ir", "pass_by_alternative", "pass")


def same_positions(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    k = ["date", "ticker"]
    a = a.assign(date=pd.to_datetime(a["date"])).sort_values(k).reset_index(drop=True)
    b = b.assign(date=pd.to_datetime(b["date"])).sort_values(k).reset_index(drop=True)
    if len(a) != len(b) or not bool((a[k].values == b[k].values).all()):
        return False
    return float(np.abs(a["target_weight"].to_numpy() - b["target_weight"].to_numpy()).max()) == 0.0


def yearly_return(nav: pd.Series) -> dict:
    """연도별 수익률 (체결 후 일간 NAV, 첫해 시작 1)."""
    last = nav.groupby(nav.index.year).last()
    prev = last.shift(1).fillna(1.0)
    return {int(y): float(v) for y, v in (last / prev - 1.0).items()}


ledger = Ledger(HERE / "ledger.jsonl")
for vid in PICKS:
    ledger.plan(vid, "G5-확인", f"선별 구간 선택 규칙 상위 2개, 공식 정책 + 운용 때 해석 {VARIANTS[vid]}",
                "2021~2025·10년 확인 (5시드, 공식 체크포인트)")
official_g5 = json.loads((REPO / "reports/20261118_B_buffer_wf/metrics.json").read_text(encoding="utf-8"))["g5"]

out = {}
for years, label in ((CONFIRM_YEARS, "2021-2025"), (ALL_YEARS, "2016-2025")):
    ctx = Ctx(years)
    rl = RLParams.from_prereg(ctx.cfg)
    buf = ctx.evaluate(buffer_run(ctx, ctx.snaps(), mode="full"), with_m9=True)
    pols = {(y, s): PPO.load(REPO / f"reports/20261117_B_rlwf_y{y}/seed_{s}/best.zip", device=DEVICE)
            for y in ctx.years for s in range(5)}
    res = {"buffer": {**brief(buf), "yearly": yearly_return(buf["_rep"].nav)}}
    for vid, x in ((OFFICIAL, {}),) + tuple((v, VARIANTS[v]) for v in PICKS):
        per, extra = {}, {}
        for s in range(5):
            env = XEnv(ctx.snaps(), ctx.engine(True, ledger=False), rl, ctx.risk, 0.0, "full", calendar=ctx.all_dates, x=x)
            pos = rollout(env, lambda d, s=s: pols[(int(d[:4]), s)]).positions
            per[s] = ctx.evaluate(pos, with_m9=True)
            if vid == OFFICIAL and label == "2016-2025":
                extra[f"s{s}_positions_same_as_official"] = same_positions(
                    pos, pd.read_parquet(REPO / f"positions/wf_20261117_B_rl_wf_s{s}.parquet"))
            if vid != OFFICIAL:
                OUT_POS.mkdir(parents=True, exist_ok=True)
                pos.to_parquet(OUT_POS / f"{vid}_s{s}_{label}.parquet", index=False)
        g5 = g5_judgment(per, buf, ctx.kpi)
        yr = [yearly_return(v["_rep"].nav) for v in per.values()]
        row = {"mean": {k: float(np.mean([brief(v)[k] for v in per.values()])) for k in METRIC_KEYS},
               **{k: g5[k] for k in G5_KEYS},
               "yearly_mean": {y: float(np.mean([r[y] for r in yr])) for y in yr[0]},
               "seeds": {s: brief(v) for s, v in per.items()}, **extra}
        if vid == OFFICIAL and label == "2016-2025":
            row["diff_vs_official_g5"] = {k: float(g5[k]) - float(official_g5[k]) for k in ("ir_gain", "turnover_ratio")}
        res[vid] = row
        if vid in PICKS:
            ledger.result(vid, years=label, seeds=5, **{k: row[k] for k in G5_KEYS}, **row["mean"])
        print(label, vid, json.dumps({k: row[k] for k in ("mean",) + G5_KEYS}, ensure_ascii=False, default=float), flush=True)
    out[label] = res
    del ctx, pols
(HERE / "confirm_g5.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("done")
