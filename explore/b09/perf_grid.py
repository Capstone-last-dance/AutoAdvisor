"""성능 트랙 2단계 (11/19): 신호 판 × 정책 × 운용 해석 × M8 격자를 선별 구간(2016~2020)에서 잰다. 학습 없음.

  python -I explore/b09/run.py perf_grid.py [정책 id ...]   (id: ml, P-b1, P-b1ml, P-b1-nocash. 없으면 처음 셋)

- 신호 판: v7 ML(공식 정책 체크포인트), b1(P-b1 학습 결과), b1ml(P-b1ml 학습 결과). 체크포인트가 없는 판은 건너뛴다.
- 시드 0·1 평균 (공식 정책도 0·1만 써서 맞춘다).
- 운용 해석: 그대로, I-smoothk-0.5, I-smoothk-0.7, I-band0.01-smooth0.5 (rl_infer.VARIANTS).
- M8: 사전등록, E1(노출 조절 끔), M-dd-0.15-f0.3, M-dd+tr+tag (m8x.M8). 버퍼도 같은 신호 판 × 같은 M8.
- 선별 기준 (실행 전 고정, 11/19): RL·버퍼를 같은 기준으로, 시드 평균 IR(코스피 200 대비) 상위. IR 차이가 0.02 이내면
  MDD가 작은 쪽. 상위 2개를 2021~2025·10년 확인 대상으로 한다. G5·G7은 참고로 적는다.
- 결과: perf_grid_search.csv, 원장.
"""
import json
import sys

from common import HERE, REPO, SEARCH_YEARS, Ctx, Ledger, brief, buffer_run, g5_judgment, g7_check
from m8x import M8, MEnv, buffer_m8, market
from rl_infer import VARIANTS as IV
from rl_screen import RUNS
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.portfolio.buffer_wf_report import benchmark_summary
from src.portfolio.params import RLParams
from src.portfolio.rl_train import DEVICE, rollout

SEEDS = (0, 1)
INTERP = {"그대로": {}, **{k: IV[k] for k in ("I-smoothk-0.5", "I-smoothk-0.7", "I-band0.01-smooth0.5")}}
SOURCES = {  # 정책 id → (신호 판, 정책 이름, 체크포인트 경로, 학습 때의 행동 해석)
    "ml": (None, "v7 공식 정책", lambda y, s: REPO / f"reports/20261117_B_rlwf_y{y}/seed_{s}/best.zip", {}),
    "P-b1": ("b1", "P-b1", lambda y, s: RUNS / "P-b1" / f"y{y}_s{s}" / "best.zip", {}),
    "P-b1ml": ("b1ml", "P-b1ml", lambda y, s: RUNS / "P-b1ml" / f"y{y}_s{s}" / "best.zip", {}),
    # 11/19 추가 (격자 1차 뒤): B1 신호에서 RL이 진 원인이 현금(25~36%)이라 현금 로짓 없는 판
    "P-b1-nocash": ("b1", "P-b1-nocash", lambda y, s: RUNS / "P-b1-nocash" / f"y{y}_s{s}" / "best.zip", {"nocash": True}),
}
PICK = sys.argv[1:] or ["ml", "P-b1", "P-b1ml"]
OUT = HERE / "perf_grid_search.csv"
OLD = pd.read_csv(OUT) if OUT.exists() and sys.argv[1:] else None  # id를 주면 기존 표에 더한다
KEYS = ("ir", "cagr", "annual_excess", "mdd", "annual_turnover", "mean_cash")

ctx = Ctx(SEARCH_YEARS)
rl = RLParams.from_prereg(ctx.cfg)
bm, ma200 = market(ctx)
ledger = Ledger(HERE / "ledger.jsonl")
rows = []
bench = None
for pid in PICK:
    sig, pname, ck, x0 = SOURCES[pid]
    if not all(ck(y, s).exists() for y in ctx.years for s in SEEDS):
        print("체크포인트가 없어 건너뜀:", pname, flush=True)
        continue
    snaps = ctx.sig_snaps(sig)
    sname = sig or "ml"
    off_rep = ctx.evaluate(buffer_run(ctx, snaps, mode="off"), with_m9=False)
    off = brief(off_rep)
    if bench is None:
        bench = benchmark_summary(ctx.inputs.bm_close, off_rep["_rep"].nav.index)
    pols = {(y, s): PPO.load(ck(y, s), device=DEVICE) for y in ctx.years for s in SEEDS}
    have_buf = OLD is not None and ((OLD["family"] == "버퍼") & (OLD["sig"] == sname)).any()
    if not have_buf:
        rows.append({"family": "버퍼", "sig": sname, "policy": "버퍼 규칙", "interp": "", "m8": "M8·M9 미적용", **off})
    for mname, spec in M8.items():
        cid = f"buf|{sname}|{mname}"
        if not have_buf:
            ledger.plan(cid, "성능", f"버퍼 규칙, 신호 {sname}, {mname}", "성능 트랙 2단계 격자 (2021~2025를 본 뒤)")
        buf = brief(ctx.evaluate(buffer_m8(ctx, snaps, spec), with_m9=True))
        g7 = g7_check(buf, off, ctx.kpi)
        row = {"family": "버퍼", "sig": sname, "policy": "버퍼 규칙", "interp": "", "m8": mname, **buf,
               "g7_mdd_red": g7["mdd_reduction_rel"], "g7_cagr_gap": buf["cagr"] - off["cagr"], "g7_pass": g7["pass"]}
        if not have_buf:
            rows.append(row)
            ledger.result(cid, years="2016-2020", **{k: row[k] for k in KEYS}, g7_pass=g7["pass"])
            print(json.dumps(row, ensure_ascii=False, default=float), flush=True)
        for iname, x in INTERP.items():
            rid = f"rl|{sname}|{iname}|{mname}" if not x0 else f"rl|{pname}|{iname}|{mname}"
            ledger.plan(rid, "성능", f"{pname} + 운용 {iname} + {mname}", "성능 트랙 2단계 격자 (2021~2025를 본 뒤)")
            per = {}
            for s in SEEDS:
                env = MEnv(snaps, ctx.engine(True, ledger=False), rl, ctx.risk, 0.0, "full", calendar=ctx.all_dates,
                           x={**x0, **x}, m8=spec, bm=bm, ma200=ma200)
                per[s] = ctx.evaluate(rollout(env, lambda d, s=s: pols[(int(d[:4]), s)]).positions, with_m9=True)
            mean = {k: float(np.mean([brief(v)[k] for v in per.values()])) for k in KEYS}
            g5 = g5_judgment(per, buf, ctx.kpi)
            r = {"family": "RL", "sig": sname, "policy": pname, "interp": iname, "m8": mname, **mean,
                 "ir_seeds": [float(per[s]["ir"]) for s in SEEDS], "g5_ir_gain": g5["ir_gain"],
                 "g5_turnover_ratio": g5["turnover_ratio"], "g5_pass": g5["pass"], "g7_pass_buffer": g7["pass"]}
            rows.append(r)
            ledger.result(rid, years="2016-2020", seeds=list(SEEDS), **{k: r[k] for k in KEYS}, g5_pass=g5["pass"])
            print(json.dumps(r, ensure_ascii=False, default=float), flush=True)
    del pols

out = pd.DataFrame(rows)
if OLD is not None:
    key = ["family", "policy", "interp", "m8"]
    new_k = set(map(tuple, out[key].fillna("").astype(str).values))
    keep = OLD[[tuple(r) not in new_k for r in OLD[key].fillna("").astype(str).values]]
    out = pd.concat([keep, out], ignore_index=True)
out.to_csv(OUT, index=False)
print("코스피 200 (같은 구간)", json.dumps(bench, default=float))
out["ir_seeds"] = out["ir_seeds"].astype(str)
cand = out[out["m8"] != "M8·M9 미적용"].sort_values("ir", ascending=False).reset_index(drop=True)
top = cand.iloc[0]
near = cand[cand["ir"] >= top["ir"] - 0.02].sort_values("mdd")
picks = [near.iloc[0]] + [r for _, r in cand.iterrows() if not r.equals(near.iloc[0])][:1]
print("IR 상위 10")
print(cand.head(10)[["family", "sig", "policy", "interp", "m8", "ir", "cagr", "mdd", "annual_turnover", "mean_cash"]].to_string())
print("선택 규칙 상위 2 (확인 대상 후보):")
for p in picks:
    print(" ", p["family"], p["sig"], p["policy"], p["interp"], p["m8"], "IR", round(p["ir"], 3), "CAGR", round(p["cagr"], 4), "MDD", round(p["mdd"], 3))
