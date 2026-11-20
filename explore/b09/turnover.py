r"""교체 줄이기: 지금 포트폴리오 종목(ML 순위 + 버퍼 규칙)은 그대로 두고 매주 교체만 줄이면 수익이 늘어나나 (11/19 사용자 결정).

배경 (docs/b09_explore.md 15절): 실전 바구니(버퍼 규칙 20종목, 엔진 밖)는 2016~2025 비용 전 CAGR +5.2%, 비용 후 −4.5%다.
매주 보유의 약 45%를 바꿔 연 9.7%p 안팎을 비용으로 잃는다. 투자 비중 타이밍만으로는 수익을 만들지 못했다.

실행 (저장소 루트, PowerShell, 사용자가 돌린다, 5~10분, 창 1개):
  .\.venv\Scripts\python.exe -I explore\b09\run.py turnover.py
  끊기면 같은 명령을 다시 돌린다 (끝난 실행은 건너뛴다).

구성 (실행 전 고정 11/19). 모두 A v7 시험 신호의 ML 순위, U_t, crash_flag 차단, 종목 10%·업종 30% 한도, M9, C 엔진,
노출 조절 끔(E = 1). 바꾸는 것은 순위를 얼마나 오래 보나, 언제 파나, 얼마나 자주 맞추나뿐이다.
- T0 기준: 버퍼 20/40/20, 매주 (지금 포트폴리오 = 바구니 타이밍의 'B 100%')
- T1 청산 80: 버퍼 20/80/20
- T2 4주 평균 순위: rank_score를 종목별 최근 4개 신호일 평균으로 바꿔 순위를 다시 매긴다 (buffer_batch.py S-rank-sm4와 같다)
- T3 4주 평균 순위 + 청산 80 (6절 2016~2020 선별에서 회전율이 가장 낮았던 조합, S-sm4+x80)
- T4 격주: 2주마다 리밸런싱 (사이 주는 보유 유지, crash_flag·M9는 그대로 적용)
- T5 4주마다
- T6 8주 평균 순위 + 청산 80
- T7 4주 평균 순위 + 청산 80 + 격주
- 참고 (판정에 쓰지 않음): 매주 리밸런싱 구성(T0·T1·T2·T3·T6)에 바구니 타이밍 에이전트(bt_agent '바구니만' 앙상블)의
  비중 E_t를 그대로 얹은 판, 그리고 그 평균 비중으로 고정한 판. 에이전트는 T0 바구니로 배웠으므로 다른 구성에는 옮겨 쓰는 셈이다.

판정 (실행 전 고정 11/19)
- (1) 수익 증가: 10년(2016-01-11 ~ 2025-12-30) CAGR > T0
- (2) 두 구간 모두: 2016~2020과 2021~2025 CAGR이 각각 T0보다 높음 (10년 연속 실행을 2020년 마지막 거래일에서 나눠 잰다)
- (3) 원금보다 늘어남: 10년 끝 잔고 > 1
- 확인 대상: (1)·(2)를 만족한 구성 중 10년 CAGR이 가장 높은 것.
- 2016~2020은 T1·T2·T3·T4가 6절(사전등록 M8을 켠 판)에서 이미 본 구간이다. 2021~2025는 이 구성들로 처음 본다.
  10년 전체도 다른 실험에서 여러 번 본 구간이라, 넘더라도 forward test(PM이 허락하면 2026 홀드아웃)로 확인해야 한다.
- 참고 지표: MDD, IR, 연 회전율, 비용 추정(연 회전율 × (2 × 한쪽 비용 + 매도세)), 연도별, 코스피 200.
- 탐색 결과다 (R-15). 결과: tov_runs.csv, tov_judgment.json, runs/tov/, 원장 tov|*.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

from common import ALL_YEARS, HERE, Ctx, Ledger, buffer_run, rank_from_score
import numpy as np
import pandas as pd
from src.portfolio.buffer_wf_report import benchmark_summary

OUT = HERE / "runs" / "tov"
AGENT = HERE / "runs" / "bt" / "eval" / "bkt_ens.json"  # 바구니 타이밍 '바구니만' 앙상블의 E_t
BT_B100 = HERE / "runs" / "bt" / "eval" / "bench_e1.json"  # 대조: T0과 같아야 한다
SPLIT = pd.Timestamp("2020-12-31")
# (id, 이름, 순위 평균 주 수, 청산 순위, 리밸런싱 주기)
VARIANTS = [
    ("T0", "기준 (버퍼 20/40, 매주)", None, None, 1),
    ("T1", "청산 80", None, 80, 1),
    ("T2", "4주 평균 순위", 4, None, 1),
    ("T3", "4주 평균 순위 + 청산 80", 4, 80, 1),
    ("T4", "격주", None, None, 2),
    ("T5", "4주마다", None, None, 4),
    ("T6", "8주 평균 순위 + 청산 80", 8, 80, 1),
    ("T7", "4주 평균 순위 + 청산 80 + 격주", 4, 80, 2),
]


def smooth_signals(sig: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """rank_score를 종목별 최근 weeks개 신호일 평균으로 바꾸고 U_t 안 순위를 다시 매긴다 (buffer_batch.smooth_score와 같다)."""
    s = sig[["date", "ticker", "rank_score"]].sort_values(["ticker", "date"])
    sm = s.groupby("ticker")["rank_score"].transform(lambda x: x.rolling(weeks, min_periods=1).mean())
    out = sig.copy()
    out["rank_score"] = sm.reindex(sig.index).to_numpy()
    out["rank"] = rank_from_score(out, "rank_score").to_numpy()
    return out


def run(ctx, snaps, b, every: int, es=None):
    """es가 없으면 노출 조절 끔(E = 1), 있으면 그 E_t를 넣는다 (M8 Step 1~3 + M9는 같다)."""
    use = snaps if es is None else [replace(s, exposure=float(e)) for s, e in zip(snaps, es)]
    pos = buffer_run(ctx, use, buffer=b, mode="no_exposure" if es is None else "full", with_m9=True, trade_every=every)
    s = ctx.evaluate(pos, with_m9=True)
    rep = s.pop("_rep")
    m = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v) for k, v in s.items()}
    return m, rep.nav


def split_cagr(nav: pd.Series) -> tuple[float, float]:
    """10년 연속 실행을 2020년 마지막 거래일에서 나눈 두 구간의 CAGR (첫 구간은 체결 전 잔고 1에서 시작)."""
    cut = nav[nav.index <= SPLIT].index[-1]
    d0 = nav.index[0] - pd.Timedelta(days=1)
    c1 = (nav[cut] / 1.0) ** (365.25 / (cut - d0).days) - 1.0
    c2 = (nav.iloc[-1] / nav[cut]) ** (365.25 / (nav.index[-1] - cut).days) - 1.0
    return float(c1), float(c2)


def yearly(nav: pd.Series) -> dict:
    last = nav.groupby(nav.index.year).last()
    return {int(y): float(v) for y, v in (last / last.shift(1).fillna(1.0) - 1.0).items()}


def save(name: str, rec: dict, nav: pd.Series) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": nav.index, "nav": nav.to_numpy()}).to_parquet(OUT / f"nav_{name}.parquet", index=False)
    (OUT / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, default=float), encoding="utf-8")


def load(name: str):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ledger_has(event: str, vid: str) -> bool:
    lines = (HERE / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    return any(r.get("event") == event and r.get("id") == vid for r in map(json.loads, filter(None, lines)))


def main() -> None:
    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("plan", "tov|T0"):  # 처음 돌릴 때만 계획 행을 쓴다 (실행 전)
        for vid, label, sm, x, every in VARIANTS:
            ledger.plan(f"tov|{vid}", "교체 줄이기 (지금 포트폴리오 종목 그대로)",
                        f"{label}: 순위 평균 {sm or 1}주, 청산 rank > {x or 40}, {every}주마다 리밸런싱. 노출 조절 끔, M9, C 엔진, 2016~2025",
                        "사용자 결정(11/19): 바구니 타이밍 뒤 1번 '교체 줄이기'. 비용 전 +5.2%·비용 후 −4.5%, 매주 45% 교체")
    t0 = time.time()
    ctx = Ctx(ALL_YEARS)
    base = ctx.snaps()
    cost_unit = 2 * (float(ctx.cfg["costs"]["commission_per_side"]) + float(ctx.cfg["costs"]["slippage_per_side"])) \
        + float(ctx.cfg["costs"]["sell_tax"])
    print(f"입력 읽음 ({time.time() - t0:.0f}초)", flush=True)
    snaps_by = {None: base}
    agent = json.loads(AGENT.read_text(encoding="utf-8")) if AGENT.exists() else None
    if agent is not None and list(agent["dates"]) != list(ctx.dates):
        raise RuntimeError("에이전트 E_t의 신호일이 평가 신호일과 다르다")

    for vid, label, sm, x, every in VARIANTS:
        if sm not in snaps_by:
            t = time.time()
            snaps_by[sm] = ctx.snaps(signals=smooth_signals(ctx.signals, sm), key=f"sm{sm}")
            print(f"  {sm}주 평균 순위 스냅샷 ({time.time() - t:.0f}초)", flush=True)
        b = ctx.buffer if x is None else replace(ctx.buffer, exit_rank_greater_than=x)
        names = [(vid, None)]
        if agent is not None and every == 1:
            w = float(np.mean(agent["e_path"]))
            names += [(f"{vid}+agent", agent["e_path"]), (f"{vid}+fixed", [w] * len(base))]
        for name, es in names:
            if load(name) is not None:
                continue
            t = time.time()
            m, nav = run(ctx, snaps_by[sm], b, every, es)
            c1, c2 = split_cagr(nav)
            rec = {**m, "variant": vid, "label": label, "kind": name[len(vid):].lstrip("+") or "100%",
                   "cagr_2016_2020": c1, "cagr_2021_2025": c2, "cost_est": m["annual_turnover"] * cost_unit,
                   "mean_e": float(np.mean(es)) if es is not None else 1.0, "yearly": yearly(nav)}
            save(name, rec, nav)
            print(f"  {name} {label}: CAGR {m['cagr']:+.2%} (2016~2020 {c1:+.2%}, 2021~2025 {c2:+.2%}), MDD {m['mdd']:.1%}, "
                  f"연 회전율 {m['annual_turnover']:.1f} ({time.time() - t:.0f}초)", flush=True)

    # 대조: T0 = 바구니 타이밍의 B 100%, T0+agent = 바구니 타이밍의 '바구니만' 앙상블
    t0r = load("T0")
    if BT_B100.exists():
        ref = json.loads(BT_B100.read_text(encoding="utf-8"))
        d = abs(t0r["cagr"] - ref["cagr"])
        print(f"대조 1: T0 CAGR {t0r['cagr']:+.6%} vs 바구니 타이밍 B 100% {ref['cagr']:+.6%} (차이 {d:.1e})")
        if d > 1e-12:
            raise RuntimeError("T0이 바구니 타이밍의 B 100%와 다르다")
    if agent is not None:
        d = abs(load("T0+agent")["cagr"] - agent["cagr"])
        print(f"대조 2: T0 + 에이전트 CAGR {load('T0+agent')['cagr']:+.6%} vs 바구니 타이밍 '바구니만' 앙상블 {agent['cagr']:+.6%} (차이 {d:.1e})")
        if d > 1e-12:
            raise RuntimeError("T0 + 에이전트가 바구니 타이밍 결과와 다르다")

    nav0 = pd.read_parquet(OUT / "nav_T0.parquet")
    days = pd.DatetimeIndex(pd.to_datetime(nav0["date"]))
    k = benchmark_summary(ctx.inputs.bm_close, days)
    kc1, kc2 = split_cagr(pd.Series(ctx.inputs.bm_close.reindex(days).to_numpy() / float(ctx.inputs.bm_close.reindex(days).iloc[0]),
                                    index=days))
    rows, judg = [], {}
    for vid, label, sm, x, every in VARIANTS:
        r = load(vid)
        j = {"(1) 수익 증가 (10년 CAGR > T0)": bool(r["cagr"] > t0r["cagr"]) if vid != "T0" else None,
             "(2) 두 구간 모두 T0보다 높음": bool(r["cagr_2016_2020"] > t0r["cagr_2016_2020"] and r["cagr_2021_2025"] > t0r["cagr_2021_2025"])
             if vid != "T0" else None,
             "(3) 원금보다 늘어남 (끝 잔고 > 1)": bool(r["final_nav"] > 1.0)}
        judg[vid] = {"label": label, **j, **{c: r[c] for c in ("cagr", "mdd", "ir", "annual_turnover", "cost_est", "final_nav",
                                                               "cagr_2016_2020", "cagr_2021_2025")}}
        for name in (vid, f"{vid}+agent", f"{vid}+fixed"):
            q = load(name)
            if q is not None:
                rows.append({"name": name, "variant": vid, "label": label, "kind": q["kind"], **{c: q[c] for c in (
                    "cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "ir", "annual_turnover", "cost_est", "final_nav", "mean_cash",
                    "mean_e")}, **{f"y{y}": v for y, v in q["yearly"].items()}})
    rd = pd.DataFrame(rows)
    rd.to_csv(HERE / "tov_runs.csv", index=False)
    cands = [(v, judg[v]["cagr"]) for v in judg if v != "T0" and judg[v]["(1) 수익 증가 (10년 CAGR > T0)"]
             and judg[v]["(2) 두 구간 모두 T0보다 높음"]]
    pick = max(cands, key=lambda z: z[1])[0] if cands else None
    agent_ref = {}
    for vid, *_ in VARIANTS:
        a, f, h = load(f"{vid}+agent"), load(f"{vid}+fixed"), load(vid)
        if a is not None and f is not None:
            agent_ref[vid] = {"agent_cagr": a["cagr"], "agent_mdd": a["mdd"], "fixed_cagr": f["cagr"], "fixed_mdd": f["mdd"],
                              "full_cagr": h["cagr"], "mean_e": a["mean_e"],
                              "beats_fixed": bool(a["cagr"] > f["cagr"] and a["mdd"] < f["mdd"]), "beats_100": bool(a["cagr"] > h["cagr"])}
    (HERE / "tov_judgment.json").write_text(json.dumps({
        "period": [str(days[0].date()), str(days[-1].date())], "kospi200": {"cagr": float(k["cagr"]), "mdd": float(k["mdd"]),
                                                                            "cagr_2016_2020": kc1, "cagr_2021_2025": kc2},
        "variants": judg, "confirm_candidate": pick, "agent_reference": agent_ref, "cost_unit": cost_unit},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not ledger_has("result", "tov|판정"):
        for vid in judg:
            ledger.result(f"tov|{vid}", **judg[vid])
        ledger.result("tov|판정", confirm_candidate=pick, agent_reference=agent_ref)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    show = rd.copy()
    for c in ("cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "cost_est", "mean_cash", "mean_e") + tuple(
            c for c in show.columns if c.startswith("y") and c[1:].isdigit()):
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(1)
    for c in ("ir", "annual_turnover", "final_nav"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(3)
    print(f"\n[2016-01-11 ~ {days[-1].date()}, ML 순위 + 버퍼 규칙 변형, 노출 조절 끔, M9, C 엔진] 코스피 200 CAGR {k['cagr']:+.1%} "
          f"(2016~2020 {kc1:+.1%}, 2021~2025 {kc2:+.1%}), MDD {k['mdd']:.1%}")
    print(show[["name", "label", "cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "ir", "annual_turnover", "cost_est",
                "final_nav", "mean_e"]].to_string(index=False))
    yc = [c for c in show.columns if c.startswith("y") and c[1:].isdigit()]
    print("\n[연도별 수익률 %, 100% 투자 구성]")
    print(show[show.kind == "100%"].set_index("name")[yc].to_string())
    print("\n[판정] (실행 전 고정)")
    for vid, jv in judg.items():
        marks = ", ".join(f"{q} {'만족' if jv[q] else '미달'}" for q in jv if q.startswith("(") and jv[q] is not None)
        print(f"  {vid} {jv['label']}: {marks}")
    print(f"  확인 대상: {pick or '없음'}")
    if agent_ref:
        print("\n[참고: 바구니 타이밍 에이전트 비중을 얹은 판 (판정에 안 씀)]")
        for vid, a in agent_ref.items():
            print(f"  {vid}: 에이전트 {a['agent_cagr']:+.2%}/MDD {a['agent_mdd']:.1%} vs 같은 평균 {a['mean_e']:.0%} 고정 "
                  f"{a['fixed_cagr']:+.2%}/{a['fixed_mdd']:.1%} vs 100% {a['full_cagr']:+.2%}")
    print("done")


if __name__ == "__main__":
    main()
