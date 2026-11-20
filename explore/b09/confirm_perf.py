"""성능 트랙 확인 (11/19): perf_grid 선택 규칙 상위 2개(둘 다 버퍼)를 2021~2025와 10년에서 잰다. 학습 없음.

  python -I explore/b09/run.py confirm_perf.py

- 확인 대상: 버퍼 규칙 + 신호 b1 + E1(노출 조절 끔), 버퍼 규칙 + 신호 b1ml + E1.
- 참고 행(확인 대상 아님): 같은 신호의 사전등록 M8·M8·M9 미적용, v7 ML 신호의 사전등록 M8·미적용.
- 대조: v7 ML 신호 + 사전등록 M8과 미적용의 10년 값이 공식 `20261118_B_buffer_wf`와 같아야 한다.
- 코스피 200(같은 거래일)의 CAGR·MDD·연도별 수익률을 같이 적는다.
- 결과: confirm_perf.json, 원장. 확인 대상 positions는 runs/confirm_perf/ (git 제외).
"""
import json

from common import ALL_YEARS, CONFIRM_YEARS, HERE, REPO, Ctx, Ledger, brief, buffer_run, g7_check
from m8x import M8, buffer_m8
from rl_screen import RUNS
import pandas as pd
from src.portfolio.buffer_wf_report import benchmark_summary

PICKS = (("b1", "E1 노출 조절 끔"), ("b1ml", "E1 노출 조절 끔"))
REFS = (("b1", "M8 사전등록"), ("b1", "M8·M9 미적용"), ("b1ml", "M8 사전등록"), ("b1ml", "M8·M9 미적용"),
        (None, "M8 사전등록"), (None, "M8·M9 미적용"))
OUT_POS = RUNS / "confirm_perf"


def yearly(nav: pd.Series, start: float = 1.0) -> dict:
    last = nav.groupby(nav.index.year).last()
    prev = last.shift(1).fillna(start)
    return {int(y): float(v) for y, v in (last / prev - 1.0).items()}


ledger = Ledger(HERE / "ledger.jsonl")
for sig, m in PICKS:
    ledger.plan(f"buf|{sig}|{m}", "성능-확인", f"버퍼 규칙, 신호 {sig}, {m}", "perf_grid 선택 규칙 상위 2개, 2021~2025·10년 확인 (2021~2025를 본 뒤 실험)")
official = json.loads((REPO / "reports/20261118_B_buffer_wf/metrics.json").read_text(encoding="utf-8"))["runs"]

out = {}
for years, label in ((CONFIRM_YEARS, "2021-2025"), (ALL_YEARS, "2016-2025")):
    ctx = Ctx(years)
    res, offs = {}, {}
    for sig, m in PICKS + REFS:
        snaps = ctx.sig_snaps(sig)
        if m == "M8·M9 미적용":
            s = ctx.evaluate(buffer_run(ctx, snaps, mode="off"), with_m9=False)
        else:
            pos = buffer_m8(ctx, snaps, M8[m])
            s = ctx.evaluate(pos, with_m9=True)
            if (sig, m) in PICKS:
                OUT_POS.mkdir(parents=True, exist_ok=True)
                pos.to_parquet(OUT_POS / f"buf_{sig}_E1_{label}.parquet", index=False)
        name = f"{sig or 'ml'} | {m}"
        res[name] = {**brief(s), "pick": (sig, m) in PICKS, "yearly": yearly(s["_rep"].nav)}
        if m == "M8·M9 미적용":
            offs[sig] = brief(s)
        if "bench" not in res:
            days = s["_rep"].nav.index
            bm = ctx.inputs.bm_close.reindex(days)
            first = float(ctx.inputs.bm_close[ctx.inputs.bm_close.index < days[0]].iloc[-1])  # 첫 체결 전날 종가
            res["bench"] = {**benchmark_summary(ctx.inputs.bm_close, days), "yearly": yearly(bm, start=first)}
    for sig, m in PICKS + REFS:
        name = f"{sig or 'ml'} | {m}"
        if m != "M8·M9 미적용" and sig in offs:
            g7 = g7_check(res[name], offs[sig], ctx.kpi)
            res[name].update(g7_mdd_red=g7["mdd_reduction_rel"], g7_cagr_gap=res[name]["cagr"] - offs[sig]["cagr"], g7_pass=g7["pass"])
        if (sig, m) in PICKS:
            ledger.result(f"buf|{sig}|{m}", years=label, **{k: v for k, v in res[name].items() if k not in ("yearly", "pick")})
        print(label, name, json.dumps({k: v for k, v in res[name].items() if k != "yearly"}, ensure_ascii=False, default=float), flush=True)
    if label == "2016-2025":
        res["control"] = {k: {x: res[f"ml | {m}"][x] - official[k][x] for x in ("ir", "cagr", "mdd", "annual_turnover")
                              if x in official[k]}
                          for k, m in (("buffer_full", "M8 사전등록"), ("buffer_off", "M8·M9 미적용"))}
        print("대조 (공식과의 차이)", json.dumps(res["control"], ensure_ascii=False, default=float))
    print(label, "코스피 200", json.dumps(res["bench"], ensure_ascii=False, default=float), flush=True)
    out[label] = res
    del ctx
(HERE / "confirm_perf.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("done")
