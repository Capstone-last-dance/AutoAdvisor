"""G5 트랙 학습 변형 선별 평가: runs/<id>/y2016~2020_s<시드>가 모두 있는 변형을 2016~2020 시험 운용하고
G5 참고 판정(비교 대상 = 사전등록 버퍼 + 같은 신호·M8·M9)을 원장과 rl_search_s<시드>.csv에 남긴다.

  python -I explore/b09/run.py rl_eval_batch.py [시드=0] [변형 id ...]

R0(사전등록 그대로)은 공식 positions(wf_20261117_B_rl_wf_s<시드>)의 2016~2020과 같은지도 본다.
"""
import json
import sys

from common import HERE, REPO, SEARCH_YEARS, Ctx, Ledger, brief, buffer_run
from rl_screen import RUNS, VARIANTS, evaluate, spec
import numpy as np
import pandas as pd

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
want = sys.argv[2:] or list(VARIANTS)
ledger = Ledger(HERE / "ledger.jsonl")


def complete(vid: str) -> bool:
    src = spec(vid).get("src", vid)
    ck = "last.zip" if spec(vid)["ckpt"] == "last" else "best.zip"
    return all((RUNS / src / f"y{y}_s{seed}" / "result.json").exists() and (RUNS / src / f"y{y}_s{seed}" / ck).exists()
               for y in SEARCH_YEARS)


ready = [v for v in want if complete(v)]
missing = [v for v in want if v not in ready]
print("평가할 변형:", ready, "\n폴드가 덜 끝난 변형(건너뜀):", missing, flush=True)
ctx = Ctx(SEARCH_YEARS)
buf = ctx.evaluate(buffer_run(ctx, ctx.snaps(), mode="full"), with_m9=True)
rows = []
for vid in ready:
    ledger.plan(vid, "G5", f"학습 변형 {VARIANTS[vid].get('why', '')}", f"선별 구간 시드 {seed}")
    try:
        r = evaluate(vid, seed, ctx, buf)
        if vid == "R0":
            pos = pd.read_parquet(RUNS / "R0" / f"positions_s{seed}_{r['years']}.parquet")
            off = pd.read_parquet(REPO / f"positions/wf_20261117_B_rl_wf_s{seed}.parquet")
            off = off[pd.to_datetime(off["date"]).isin(pd.to_datetime(ctx.dates))]
            k = ["date", "ticker"]
            a = pos.assign(date=pd.to_datetime(pos["date"])).sort_values(k).reset_index(drop=True)
            b = off.assign(date=pd.to_datetime(off["date"])).sort_values(k).reset_index(drop=True)
            same = len(a) == len(b) and bool((a[k].values == b[k].values).all())
            r["same_as_official"] = same and float(np.abs(a["target_weight"] - b["target_weight"]).max()) == 0.0
        r.pop("buffer", None)
        ledger.result(vid, years=r["years"], seed=seed, **{k: v for k, v in r.items() if k not in ("vid", "seed", "years")})
    except Exception as e:  # 실패도 원장에 남긴다
        r = {"vid": vid, "seed": seed, "error": repr(e)}
        ledger.result(vid, seed=seed, error=repr(e))
    rows.append(r)
    print(json.dumps(r, ensure_ascii=False, default=float), flush=True)
out = pd.DataFrame(rows)
path = HERE / f"rl_search_s{seed}.csv"
if path.exists():  # 일부 변형만 다시 평가해도 나머지 행은 남긴다 (같은 id는 새 값으로)
    old = pd.read_csv(path)
    out = pd.concat([old[~old["vid"].isin(out["vid"])], out], ignore_index=True)
out.to_csv(path, index=False)
print("buffer", json.dumps(brief(buf), ensure_ascii=False, default=float))
print("done", len(rows))
