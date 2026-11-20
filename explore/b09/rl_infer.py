"""G5 트랙 '운용 때만' 변형: v7 공식 정책(재학습 없음)을 행동 해석만 바꿔 시험 운용한다 (5시드 평균, 선별 연도).
학습 때 본 해석과 다르므로 별도 변형(I-*)으로 원장에 적는다.

  python -I explore/b09/run.py rl_infer.py [--perseed] [변형 id ...]

- 인자 없음: 전체 → infer_search.csv. id를 주면 그 변형만 → infer_search_k.csv (같은 id 행만 새 값으로 바꾼다).
- --perseed: 같은 변형을 다시 돌려 시드별 값을 infer_perseed.csv에 남긴다 (새 변형이 아니다. 평균 표는 건드리지 않는다).
"""
import json
import sys

from common import HERE, REPO, SEARCH_YEARS, Ctx, Ledger, brief, buffer_run
from rl_screen import XEnv, g5_judgment
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.portfolio.params import RLParams
from src.portfolio.rl_train import DEVICE, rollout

VARIANTS = {
    "I-band-0.005": {"band": 0.005}, "I-band-0.01": {"band": 0.01}, "I-band-0.02": {"band": 0.02},
    "I-band-0.03": {"band": 0.03}, "I-smooth-0.3": {"smooth": 0.3}, "I-smooth-0.5": {"smooth": 0.5},
    "I-smooth-0.7": {"smooth": 0.7}, "I-band0.01-smooth0.5": {"band": 0.01, "smooth": 0.5},
    # 투자 비중 유지판 (11/18 추가: 위 변형은 섞을 때마다 현금이 쌓여 회전율이 줄었다)
    "I-smoothk-0.3": {"smooth": 0.3, "keep_invested": True}, "I-smoothk-0.5": {"smooth": 0.5, "keep_invested": True},
    "I-smoothk-0.7": {"smooth": 0.7, "keep_invested": True}, "I-bandk-0.01": {"band": 0.01, "keep_invested": True},
    "I-bandk-0.02": {"band": 0.02, "keep_invested": True},
    # 11/19 추가: 띠 0.03은 넘었지만 현금이 63%였다 → 투자 비중 유지판
    "I-bandk-0.03": {"band": 0.03, "keep_invested": True},
}


def upsert(path, new: pd.DataFrame, keys: list) -> None:
    """path의 표에서 new와 같은 keys 행만 바꾸고 나머지는 남긴다."""
    if path.exists():
        old = pd.read_csv(path)
        hit = old.set_index(keys).index.isin(new.set_index(keys).index)
        new = pd.concat([old[~hit], new], ignore_index=True)
    new.to_csv(path, index=False)



def main() -> None:
    args = sys.argv[1:]
    perseed = "--perseed" in args
    args = [a for a in args if a != "--perseed"]
    pick = args or list(VARIANTS)
    ledger = Ledger(HERE / "ledger.jsonl")
    for vid in pick:
        why = ("같은 변형 재실행: 시드별 값 기록 (새 변형 아님)" if perseed
               else "재학습 없이 회전율만 줄이면 대안 기준에 닿는가")
        ledger.plan(vid, "G5", f"공식 정책 + 운용 때 해석 {VARIANTS[vid]}", why)

    ctx = Ctx(SEARCH_YEARS)
    rl = RLParams.from_prereg(ctx.cfg)
    buf = ctx.evaluate(buffer_run(ctx, ctx.snaps(), mode="full"), with_m9=True)
    pols = {(y, s): PPO.load(REPO / f"reports/20261117_B_rlwf_y{y}/seed_{s}/best.zip", device=DEVICE)
            for y in ctx.years for s in range(5)}
    rows, seed_rows = [], []
    for vid in pick:
        per = {}
        for s in range(5):
            env = XEnv(ctx.snaps(), ctx.engine(True, ledger=False), rl, ctx.risk, 0.0, "full", calendar=ctx.all_dates, x=VARIANTS[vid])
            pos = rollout(env, lambda d, s=s: pols[(int(d[:4]), s)]).positions
            per[s] = ctx.evaluate(pos, with_m9=True)
            if perseed:
                g1 = g5_judgment({s: per[s]}, buf, ctx.kpi)
                seed_rows.append({"id": vid, "seed": s, **brief(per[s]), "ir_gain": g1["ir_gain"],
                                  "turnover_ratio": g1["turnover_ratio"], "cagr_not_lower": g1["cagr_not_lower"],
                                  "pass": g1["pass"]})
        g5 = g5_judgment(per, buf, ctx.kpi)
        row = {"id": vid, **{k: float(np.mean([brief(v)[k] for v in per.values()])) for k in ("ir", "cagr", "mdd", "annual_turnover", "mean_cash")},
               "ir_gain": g5["ir_gain"], "turnover_ratio": g5["turnover_ratio"], "cagr_not_lower": g5["cagr_not_lower"],
               "pass": g5["pass"], "buffer_cagr": float(buf["cagr"]), "buffer_turnover": float(buf["annual_turnover"])}
        ledger.result(vid, years="2016-2020", seeds=5, perseed_rerun=perseed, **{k: v for k, v in row.items() if k != "id"})
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=float), flush=True)
    if perseed:
        upsert(HERE / "infer_perseed.csv", pd.DataFrame(seed_rows), ["id", "seed"])
    else:
        upsert(HERE / ("infer_search.csv" if not args else "infer_search_k.csv"), pd.DataFrame(rows), ["id"])


if __name__ == "__main__":  # confirm_g5.py가 VARIANTS만 불러 쓴다
    main()
