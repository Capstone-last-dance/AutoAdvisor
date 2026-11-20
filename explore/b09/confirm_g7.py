"""G7 트랙 확인 (11/18 고정한 2개): 2021~2025와 전체 10년. 변형 정의는 buffer_batch.py와 같다."""
import json
import math

from common import ALL_YEARS, CONFIRM_YEARS, HERE, Ctx, Ledger, brief, buffer_run, g7_check
import numpy as np
import pandas as pd

ledger = Ledger(HERE / "ledger.jsonl")
PICKS = ("M-dd+tr+tag", "M-dd-0.15-f0.3")
for vid in PICKS:
    ledger.plan(vid, "G7-확인", "선별 구간 선택 규칙 상위 2개", "2021~2025·10년 확인")

out = {}
for years, label in ((CONFIRM_YEARS, "2021-2025"), (ALL_YEARS, "2016-2025")):
    ctx = Ctx(years)
    risk, bm = ctx.risk, ctx.inputs.bm_close
    ma200 = bm.rolling(200, min_periods=200).mean()

    def exp_trend_b(i, snap, e, nav, chosen):
        t = pd.Timestamp(snap.date)
        return min(e, 0.3) if bm[t] < ma200[t] else e

    def exp_dd(th, factor):
        def f(i, snap, e, nav, chosen):
            if len(nav) == 0:
                return e
            dd = 1.0 - float(nav.iloc[-1]) / float(nav.max())
            return e * factor if dd > th else e
        return f

    def both(i, snap, e, nav, chosen):
        return exp_dd(0.15, 0.5)(i, snap, exp_trend_b(i, snap, e, nav, chosen), nav, chosen)

    block_v20 = lambda i, snap: snap.vol_20 > 0.8
    snaps = ctx.snaps()
    off = brief(ctx.evaluate(buffer_run(ctx, snaps, mode="off"), with_m9=False))
    pre = brief(ctx.evaluate(buffer_run(ctx, snaps, mode="full"), with_m9=True))
    runs = {
        "M8 사전등록": pre,
        "M-dd+tr+tag": brief(ctx.evaluate(buffer_run(ctx, snaps, mode="full", exposure_fn=both, block_fn=block_v20), with_m9=True)),
        "M-dd-0.15-f0.3": brief(ctx.evaluate(buffer_run(ctx, snaps, mode="full", exposure_fn=exp_dd(0.15, 0.3)), with_m9=True)),
    }
    res = {"off": off}
    for name, s in runs.items():
        g = g7_check(s, off, ctx.kpi)
        res[name] = {**s, "g7_mdd_red": g["mdd_reduction_rel"], "g7_cagr_gap": s["cagr"] - off["cagr"], "g7_pass": g["pass"]}
        if name in PICKS:
            ledger.result(name, years=label, **res[name])
    out[label] = res
    print(label, json.dumps(res, ensure_ascii=False, default=float), flush=True)
    del ctx
(HERE / "confirm_g7.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
