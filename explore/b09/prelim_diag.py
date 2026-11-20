"""B-09 계획용 예비 진단 (11/18, 공식 결과 아님). v7 공식 결과를 읽기만 한다.

1. 무관심한 정책(로짓 모두 같음)의 현금 = 1 - min(20, N) / (N + 1), N = 거래 가능 슬롯 수
   (rank <= 40 ∪ 직전 보유) 중 crash_flag = 0. 쿨다운은 무시한다.
2. 학습한 정책의 현금: 공식 positions의 CASH, 노출 E_t로 되돌린 M8 전 현금 근사 1 - (1 - cash) / E_t
3. 보상에서 현금과 종목의 차이: 전부 현금 a_k = -R_BM (519주 산술 x 52) vs buffer_off a_k
4. 고른 체크포인트의 스텝 분포
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])

s = pd.read_parquet(ROOT / "signals/wf_20261117_A_wfnews.parquet")
s["date"] = pd.to_datetime(s["date"])
by = {d: g for d, g in s.groupby("date")}
dates = sorted(by)

print("== 1·2. 현금")
for seed in range(5):
    p = pd.read_parquet(ROOT / f"positions/wf_20261117_B_rl_wf_s{seed}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    pg = {d: g for d, g in p.groupby("date")}
    prev, rows = set(), []
    for d in dates:
        g0 = by[d]
        ok = set(g0.ticker[g0.crash_flag == 0])
        top = set(g0.ticker[g0["rank"] <= 40])
        n = len((top | prev) & ok)
        g = pg[d]
        st = g[g.ticker != "CASH"]
        cash = float(g.target_weight[g.ticker == "CASH"].sum())
        e = float(g.exposure.iloc[0])
        rows.append((n, 1 - min(20, n) / (n + 1), cash, e, 1 - (1 - cash) / e, int((st.target_weight > 0).sum())))
        prev = set(st.ticker[st.target_weight > 0])
    r = pd.DataFrame(rows, columns=["N", "flat", "cash", "E", "pre_m8", "n_hold"])
    print(f"s{seed}: N {r.N.mean():.1f}, 무관심 현금 {r.flat.mean():.3f}, 실제 현금 {r.cash.mean():.3f}, "
          f"E_t {r.E.mean():.3f}, M8 전 현금 근사 {r.pre_m8.mean():.3f}, 보유 수 {r.n_hold.min()}~{r.n_hold.max()}")

print("== 3. 보상에서 현금 vs 종목")
ix = pd.read_parquet(ROOT / "data/processed/index.parquet")
c = ix.set_index(pd.to_datetime(ix["date"]))["close"].sort_index()
days = c.index
fill = [days[days > x][0] for x in dates]
r_bm = np.array([c[fill[i + 1]] / c[fill[i]] - 1 for i in range(len(fill) - 1)])
dec = pd.read_csv(ROOT / "reports/20261118_B_g5g7/decomposition.csv").set_index("portfolio")
print(f"주 {len(r_bm)}개, 전부 현금 a_k = -R_BM 연 {-r_bm.mean() * 52:.4f}, 주간 R_BM 표준편차 {r_bm.std():.4f}")
print(f"buffer_off a_k 연 {dec.loc['buffer_off', 'excess']:.4f} (비용 {dec.loc['buffer_off', 'cost']:.4f}, "
      f"선택 {dec.loc['buffer_off', 'selection']:.4f})")

print("== 4. 체크포인트 스텝")
steps = []
for y in range(2016, 2026):
    m = json.loads((ROOT / f"reports/20261117_B_rlwf_y{y}/metrics.json").read_text(encoding="utf-8"))
    steps += [v["best_timesteps"] for v in m["main_training"]["per_seed"]]
print(f"{len(steps)}개 중 50,000스텝 이하 {sum(x <= 50000 for x in steps)}개, 300,000 {steps.count(300000)}개")
print(sorted(Counter(steps).items()))
