"""대조 실행: 바꾼 것 없이 돌려 공식 G5 비교 표(20261118_B_buffer_wf)를 그대로 재현하는지 본다.

1) 10년 (2016~2025): positions와 지표가 공식과 같아야 한다.
2) 선별 연도 (2016~2020): positions가 공식 positions의 앞 5년과 같아야 한다 (경로가 인과적).
"""
import json
import sys
import time

from common import ALL_YEARS, REPO, SEARCH_YEARS, Ctx, brief, buffer_run
import numpy as np
import pandas as pd

OFFICIAL = json.loads((REPO / "reports/20261118_B_buffer_wf/metrics.json").read_text(encoding="utf-8"))


def same_positions(a: pd.DataFrame, b: pd.DataFrame) -> float:
    k = ["date", "ticker"]
    a = a.assign(date=pd.to_datetime(a["date"])).sort_values(k).reset_index(drop=True)
    b = b.assign(date=pd.to_datetime(b["date"])).sort_values(k).reset_index(drop=True)
    if len(a) != len(b) or not (a[k].values == b[k].values).all():
        return float("inf")
    return float(np.max(np.abs(a["target_weight"].to_numpy() - b["target_weight"].to_numpy())))


out = {}
for years, label in ((ALL_YEARS, "all"), (SEARCH_YEARS, "search")):
    t0 = time.perf_counter()
    ctx = Ctx(years)
    snaps = ctx.snaps()
    for mode in ("full", "off"):
        t1 = time.perf_counter()
        pos = buffer_run(ctx, snaps, mode=mode)
        off = pd.read_parquet(REPO / f"positions/wf_20261118_B_buffer_wf_{mode}.parquet")
        off = off[pd.to_datetime(off["date"]).isin(pd.to_datetime(ctx.dates))]
        diff = same_positions(pos, off)
        row = {"positions_max_diff": diff, "run_seconds": round(time.perf_counter() - t1, 1)}
        if label == "all":
            s = ctx.evaluate(pos, with_m9=(mode != "off"))
            o = OFFICIAL["runs"][f"buffer_{mode}"]
            row["metric_max_diff"] = max(abs(float(s[k]) - float(o[k])) for k in ("ir", "cagr", "mdd", "annual_turnover", "mean_cash"))
            row["metrics"] = brief(s)
        out[f"{label}_{mode}"] = row
        print(label, mode, row, flush=True)
    print(label, "load", round(ctx.load_seconds, 1), "total", round(time.perf_counter() - t0, 1), flush=True)
    del ctx
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
