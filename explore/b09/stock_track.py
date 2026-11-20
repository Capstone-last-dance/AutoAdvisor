"""주식만으로 (11/19 사용자 결정): 분산 + 사고 방지 + 규칙 기반 종목 신호 + M8 변형. C 엔진으로 체결·비용·지표를 잰다.

  python -I explore/b09/run.py stock_track.py

설계 (실행 전 고정, 11/19)
- 종목군: U_t 중 시총 순위 200 이내 + 60일 평균 거래대금이 U_t 중앙값 이상 (조작이 쉬운 소형·저유동 종목을 뺀다,
  2023년 4월 폭락 종목 교훈). 지표는 모두 신호일 t 종가까지의 가격·시총·거래대금만 쓴다 (R-03).
- 종목 신호 (점수가 높을수록 rank가 좋다)
  · S0 대형주: 시가총액 (신호 없이 분산만, 지수 대용)
  · S1 저변동: −vol_60 (A 피처 함수)
  · S2 모멘텀: mom_12_1 (B1과 같다)
  · S3 저변동 + 모멘텀: 종목군 안 −vol_60 백분위와 mom_12_1 백분위의 평균
- 급락 플래그: A 베이스라인과 같다 (U_t에서 vol_60 상위 ceil(0.10 × N)). crash_prob = vol_60 백분위.
- 보유: 버퍼 규칙을 2배로 (진입 rank ≤ 40, 청산 rank > 80, 최대 40종목), 종목당 비중은 1/40로 고정하고 빈자리는 현금
  (보유가 적을 때 몇 종목에 몰리지 않게). 종목 한도 10%·업종 한도 30%·M9는 그대로.
- M8: 사전등록, E1(노출 조절 끔), M-dd+tr+tag, M-dd-0.15-f0.3 (m8x.M8). 참고로 M8·M9 미적용도 잰다 (G7 참고값).

구간과 판정 (실행 전 고정)
- 설계 2016~2025 (이미 본 구간), 시험 2011~2015 (처음 보는 구간).
- 선택: 설계 구간에서 MDD가 코스피 200보다 작은 구성 중 CAGR 최고.
- 교체 기준: 고른 구성이 시험 구간에서 (1) CAGR > 코스피 200, (2) MDD < 코스피 200, (3) 끝 잔고 ≥ 처음 투자한 돈,
  그리고 설계 구간에서 (4) CAGR > B-09 최종 모델(−0.53%). 모두 만족하면 B-09 최종 모델(탐색)을 이 구성으로 바꾼다.
- 결과: stock_track.json, stock_track.csv, 원장. 고른 구성의 positions는 runs/stock_track/ (git 제외).
"""
import json
import math
from dataclasses import replace

from common import CONFIRM_YEARS, HERE, Ctx, Ledger, brief, buffer_run, g7_check
from m8x import M8, buffer_m8
from rl_screen import RUNS
import numpy as np
import pandas as pd
from src.features.price_features import mom_12_1, vol_60
from src.portfolio.buffer_rule import buffer_rule_weights
from src.portfolio.buffer_wf_report import benchmark_summary

N_HOLD, CAP_RANK = 40, 200
B09_FINAL_CAGR = -0.0053  # docs/b09_explore.md 7절 조합 확인 (10년, 5시드 평균)
TEST_YEARS = (2011, 2012, 2013, 2014, 2015)
DESIGN_YEARS = (2016,) + tuple(range(2017, 2021)) + CONFIRM_YEARS
STRATS = ("S0 대형주", "S1 저변동", "S2 모멘텀", "S3 저변동 + 모멘텀")
OUT_POS = RUNS / "stock_track"


def factors(prices):
    pv = lambda c: prices.pivot(index="date", columns="ticker", values=c)
    close, mcap, value = pv("close"), pv("mcap"), pv("value")
    return {"vol": vol_60(close), "mom": mom_12_1(close), "mcap": mcap, "val60": value.rolling(60, min_periods=60).mean()}


def pct(s):
    return s.rank(pct=True, method="average")


def signals_for(ctx, F, strat):
    u = ctx.tables["universe"]
    u = u.loc[u["in_universe"].astype(bool)]
    by = {pd.Timestamp(d): g["ticker"].astype(str).tolist() for d, g in u.groupby(pd.to_datetime(u["date"]))}
    out = []
    for d in ctx.dates:
        t = pd.Timestamp(d)
        tick = by[t]
        df = pd.DataFrame({"ticker": tick})
        for k, f in F.items():
            df[k] = f.loc[t].reindex(tick).to_numpy() if t in f.index else np.nan
        cap_rank = df["mcap"].rank(ascending=False, method="first")
        elig = (cap_rank <= CAP_RANK) & (df["val60"] >= df["val60"].median()) & df["vol"].notna()
        if strat == "S0 대형주":
            score = df["mcap"]
        elif strat == "S1 저변동":
            score = -df["vol"]
        elif strat == "S2 모멘텀":
            score = df["mom"]
            elig &= df["mom"].notna()
        else:
            e = df[elig]
            score = pd.Series(np.nan, index=df.index)
            score[elig] = (pct(-e["vol"]) + pct(e["mom"].fillna(e["mom"].median()))) / 2.0
        df["elig"], df["score"] = elig, score
        df = df.sort_values(["elig", "score", "ticker"], ascending=[False, False, True], na_position="last", kind="stable")
        n = len(df)
        df["rank"] = np.arange(1, n + 1, dtype=np.int32)
        df["rank_score"] = 1.0 - (df["rank"] - 1) / n
        df["crash_prob"] = df["vol"].rank(pct=True, method="average").fillna(0.5)
        k = math.ceil(0.10 * n)
        flagged = set(df.dropna(subset=["vol"]).sort_values(["vol", "ticker"], ascending=[False, True])["ticker"].iloc[:k])
        df["crash_flag"] = df["ticker"].isin(flagged).astype("int8")
        df.insert(0, "date", t)
        out.append(df[["date", "ticker", "rank_score", "rank", "crash_prob", "crash_flag"]])
    return pd.concat(out, ignore_index=True)


def choose(snap, holdings, buffer, cooldown):
    sig = pd.DataFrame({"ticker": snap.tickers, "rank": snap.rank, "crash_flag": snap.crash_flag.astype(np.int8)})
    w = buffer_rule_weights(holdings, sig, snap.tickers, snap.tickers[cooldown], buffer)
    return w * 0.0 + min(1.0 / max(len(w), 1), 1.0 / N_HOLD) if len(w) else w


def yearly(nav, start=1.0):
    last = nav.groupby(nav.index.year).last()
    return {int(y): float(v) for y, v in (last / last.shift(1).fillna(start) - 1.0).items()}


ledger = Ledger(HERE / "ledger.jsonl")
for s in STRATS:
    for m in M8:
        ledger.plan(f"stock|{s}|{m}", "주식만", f"{s}, 종목군 시총 200·거래대금 중앙값 이상, 40종목 1/40, {m}",
                    "사용자 결정(11/19): 주식으로만. 설계 2016~2025, 시험 2011~2015")
out, rows = {}, []
for label, years in (("시험 2011-2015", TEST_YEARS), ("설계 2016-2025", DESIGN_YEARS)):
    ctx = Ctx(years)
    buf = replace(ctx.buffer, enter_rank_max=N_HOLD, exit_rank_greater_than=2 * N_HOLD, max_holdings=N_HOLD)
    F = factors(ctx.tables["prices"])
    bench = None
    for s in STRATS:
        snaps = ctx.snaps(signals=signals_for(ctx, F, s), key=f"stock-{s}")
        off_rep = ctx.evaluate(buffer_run(ctx, snaps, buffer=buf, mode="off", choose_fn=choose), with_m9=False)
        off = brief(off_rep)
        if bench is None:
            days = off_rep["_rep"].nav.index
            first = float(ctx.inputs.bm_close[ctx.inputs.bm_close.index < days[0]].iloc[-1])
            bench = {**benchmark_summary(ctx.inputs.bm_close, days), "yearly": yearly(ctx.inputs.bm_close.reindex(days), first)}
        rows.append({"window": label, "strat": s, "m8": "M8·M9 미적용", **off, "yearly": yearly(off_rep["_rep"].nav)})
        for m, spec in M8.items():
            pos = buffer_m8(ctx, snaps, spec, buffer=buf, choose_fn=choose)
            rep = ctx.evaluate(pos, with_m9=True)
            r = brief(rep)
            g7 = g7_check(r, off, ctx.kpi)
            held = pos[pos["ticker"] != "CASH"].groupby("date").size()
            row = {"window": label, "strat": s, "m8": m, **r, "final_nav": float(rep["final_nav"]),
                   "mean_holdings": float(held.reindex(pd.unique(pos["date"])).fillna(0).mean()),
                   "g7_mdd_red": g7["mdd_reduction_rel"], "g7_cagr_gap": r["cagr"] - off["cagr"], "g7_pass": g7["pass"],
                   "yearly": yearly(rep["_rep"].nav)}
            rows.append(row)
            OUT_POS.mkdir(parents=True, exist_ok=True)
            pos.to_parquet(OUT_POS / f"{s.split()[0]}_{m.split()[0]}_{label.split()[1]}.parquet", index=False)
            ledger.result(f"stock|{s}|{m}", years=label, **{k: v for k, v in row.items() if k not in ("yearly", "window", "strat", "m8")})
            print(label, s, m, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                           if k not in ("yearly",)}, ensure_ascii=False, default=float), flush=True)
    out[label] = {"bench": bench}
    print(label, "코스피 200", json.dumps(bench, ensure_ascii=False, default=float), flush=True)
    del ctx

df = pd.DataFrame(rows)
df["yearly"] = df["yearly"].map(json.dumps)
df.to_csv(HERE / "stock_track.csv", index=False)
(HERE / "stock_track.json").write_text(json.dumps({"bench": out, "rows": rows}, ensure_ascii=False, indent=1, default=float),
                                       encoding="utf-8")
cand = df[df["m8"] != "M8·M9 미적용"]
for w in ("설계 2016-2025", "시험 2011-2015"):
    b = out[w]["bench"]
    print(f"\n[{w}] 코스피 200: CAGR {b['cagr']*100:+.2f}%, MDD {b['mdd']*100:.1f}%")
    t = cand[cand.window == w][["strat", "m8", "cagr", "mdd", "ir", "final_nav", "mean_cash", "mean_holdings", "annual_turnover", "g7_pass"]].copy()
    for c in ("cagr", "mdd", "mean_cash"):
        t[c] = (t[c] * 100).round(1)
    print(t.round(3).to_string(index=False))
bd, bt = out["설계 2016-2025"]["bench"], out["시험 2011-2015"]["bench"]
design = cand[(cand.window == "설계 2016-2025") & (cand.mdd < bd["mdd"])]
if design.empty:
    print("\n선택: 설계 구간에서 MDD가 지수보다 작은 구성이 없다 → 교체하지 않음")
    ledger.result("stock|선택", pick=None, replace=False)
else:
    p = design.sort_values("cagr", ascending=False).iloc[0]
    tp = cand[(cand.window == "시험 2011-2015") & (cand.strat == p.strat) & (cand.m8 == p.m8)].iloc[0]
    checks = {"시험 CAGR > 코스피 200": bool(tp.cagr > bt["cagr"]), "시험 MDD < 코스피 200": bool(tp.mdd < bt["mdd"]),
              "시험 끝 잔고 ≥ 원금": bool(tp.final_nav >= 1.0), "설계 CAGR > B-09 최종 모델": bool(p.cagr > B09_FINAL_CAGR)}
    print(f"\n선택 (설계 구간 규칙): {p.strat} + {p.m8} (설계 CAGR {p.cagr*100:+.2f}%, MDD {p.mdd*100:.1f}%)")
    print(f"  시험 구간: CAGR {tp.cagr*100:+.2f}%, MDD {tp.mdd*100:.1f}%, 끝 잔고 {tp.final_nav:.3f}")
    for k, v in checks.items():
        print(f"  {k}: {'만족' if v else '미달'}")
    print("교체 기준:", "모두 만족 → B-09 최종 모델 교체" if all(checks.values()) else "미달 → 교체하지 않음")
    ledger.result("stock|선택", pick=f"{p.strat} + {p.m8}", checks=checks, replace=all(checks.values()))
print("\n연도별 수익률 % (설계·시험, M8 사전등록과 E1)")
for w in ("시험 2011-2015", "설계 2016-2025"):
    sub = [r for r in rows if r["window"] == w and r["m8"] in ("M8 사전등록", "E1 노출 조절 끔")]
    yt = pd.DataFrame({f"{r['strat']} | {r['m8']}": r["yearly"] for r in sub}).T
    yt.loc["코스피 200"] = pd.Series(out[w]["bench"]["yearly"])
    print(w); print((yt * 100).round(1).to_string())
print("done")
