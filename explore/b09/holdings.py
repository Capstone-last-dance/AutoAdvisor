r"""보유 종목 수 늘리기: T7 규칙 그대로 두고 보유 수만 20 → 40·60·80·120으로 늘린다 (11/20 사용자 결정).

배경 (docs 21절): 시험 신호의 전체 예측력(rank IC)은 학습용 신호와 같지만, 상위 20종목의 초과 수익은 낮다
(5일 초과 연 환산, 비용 없음, 2016~2022: 상위 20 +3.8%, 상위 40 +4.5%, 상위 80 +5.7%, 상위 120 +4.4%).
지금 구조(최대 20종목)는 시험 신호가 가장 약한 자리만 고른다.

실행 (저장소 루트, PowerShell, 사용자가 돌린다, 3~6분, 창 1개):
  .\.venv\Scripts\python.exe -I explore\b09\run.py holdings.py
  끊기면 같은 명령을 다시 돌린다 (끝난 실행은 건너뛴다).

구성 (실행 전 고정 11/20). 모두 T7과 같다: 4주 평균 순위, 격주 리밸런싱, 같은 비중, crash_flag 차단, 종목 10%·업종 30% 한도,
M9, 노출 조절 끔(E = 1), C 엔진, 2016~2025 연속. 바꾸는 것은 보유 수 N과 그에 맞춘 순위 기준뿐이다.
- 진입 rank ≤ N, 청산 rank > N + 60 (보유 수 밖으로 60자리 여유, N = 20이면 T7의 20/80과 같다), 최대 N종목.
- N = 20 (T7, 기준. turnover.py T7과 같아야 한다 — 다르면 멈춘다), 40, 60, 80, 120.

판정 (실행 전 고정 11/20)
- 고르기: N = 40·60·80·120 중 2016~2020 CAGR이 가장 높은 N을 고른다 (2016~2020만 본다).
- 확인 (2021~2025): (a) 고른 N의 2021~2025 CAGR > T7의 2021~2025 CAGR
                    (b) 고른 N의 2021~2025 MDD ≤ T7의 2021~2025 MDD + 5%p
                    (c) 10년 CAGR > T7 (참고 조건)
  (a)·(b)를 만족하면 '확인 통과'. (c)는 같이 적는다.
- 주의: 상위 k 진단(21절)이 2016~2022를 봤으므로 2021~2022는 완전히 처음 보는 구간이 아니다. 2023~2025는 이 구성으로 처음이다.
  넘어도 forward test(PM이 허락하면 2026 홀드아웃)로 확인해야 한다.
- 참고 지표: MDD, IR, 연 회전율, 비용 추정, 평균 보유 수, 코스피 종합(18절)·코스피 200 대비 연 초과수익과 IR, 연도별.
- 탐색 결과다 (R-15). 결과: hold_runs.csv, hold_judgment.json, runs/hold/, 원장 hold|*.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

from bench_kospi import KOSPI_PATH, stats as kospi_stats
from common import ALL_YEARS, HERE, Ctx, Ledger
import exitrule as ex
import numpy as np
import pandas as pd
from turnover import ledger_has, run, smooth_signals, split_cagr, yearly

OUT = HERE / "runs" / "hold"
TOV = HERE / "runs" / "tov"
NS = (20, 40, 60, 80, 120)
BAND = 60
EVERY = 2


def save(name: str, rec: dict, nav: pd.Series) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": nav.index, "nav": nav.to_numpy()}).to_parquet(OUT / f"nav_{name}.parquet", index=False)
    (OUT / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, default=float), encoding="utf-8")


def load(name: str):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("plan", "hold|N20"):  # 실행 전 계획 행 (처음 한 번)
        for n in NS:
            ledger.plan(f"hold|N{n}", "보유 종목 수 늘리기 (T7 규칙)",
                        f"보유 {n}종목: 진입 rank ≤ {n}, 청산 rank > {n + BAND}, 4주 평균 순위, 격주, E = 1, M9, C 엔진, 2016~2025",
                        "사용자 결정(11/20): 시험 신호는 상위 20보다 상위 80의 초과 수익이 높다 (21절). 2016~2020으로 고르고 2021~2025로 확인")
    t0 = time.time()
    ctx = Ctx(ALL_YEARS)
    cost_unit = 2 * (float(ctx.cfg["costs"]["commission_per_side"]) + float(ctx.cfg["costs"]["slippage_per_side"])) \
        + float(ctx.cfg["costs"]["sell_tax"])
    snaps = ctx.snaps(signals=smooth_signals(ctx.signals, 4), key="sm4")
    k = pd.read_parquet(KOSPI_PATH)
    kospi = pd.Series(k["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(k["date"])).as_unit("ns"))
    k200 = ctx.inputs.bm_close.copy()
    k200.index = pd.DatetimeIndex(pd.to_datetime(k200.index)).as_unit("ns")
    cal = k200.index[k200.index <= pd.Timestamp("2025-12-30")]
    fills = pd.DatetimeIndex([cal[cal.searchsorted(d, side="right")] for d in pd.to_datetime(ctx.dates)])
    print(f"입력 읽음 ({time.time() - t0:.0f}초)", flush=True)

    for n in NS:
        name = f"N{n}"
        if load(name) is not None:
            continue
        t = time.time()
        b = replace(ctx.buffer, enter_rank_max=n, exit_rank_greater_than=n + BAND, max_holdings=n)
        m, nav = run(ctx, snaps, b, EVERY)
        c1, c2 = split_cagr(nav)
        d1, d2 = ex.half_mdd(nav)
        ks = kospi_stats(nav, {"kospi": kospi, "k200": k200}, fills)
        rec = {**m, "n": n, "enter": n, "exit": n + BAND, "cagr_2016_2020": c1, "cagr_2021_2025": c2, "mdd_2016_2020": d1,
               "mdd_2021_2025": d2, "cost_est": m["annual_turnover"] * cost_unit, "excess_kospi": ks["excess_kospi"],
               "ir_kospi": ks["ir_kospi"], "excess_k200": ks["excess_k200"], "ir_k200": ks["ir_k200"],
               "windows": ex.windows(nav), "yearly": yearly(nav)}
        save(name, rec, nav)
        print(f"  보유 {n}: CAGR {m['cagr']:+.2%} (2016~2020 {c1:+.2%}, 2021~2025 {c2:+.2%}), MDD {m['mdd']:.1%} "
              f"(전반 {d1:.1%}, 후반 {d2:.1%}), 회전율 {m['annual_turnover']:.1f}, 코스피 대비 {ks['excess_kospi']:+.1%}·IR "
              f"{ks['ir_kospi']:+.2f} ({time.time() - t:.0f}초)", flush=True)

    ref = json.loads((TOV / "T7.json").read_text(encoding="utf-8"))
    d = abs(load("N20")["cagr"] - ref["cagr"])
    print(f"대조: 보유 20 CAGR {load('N20')['cagr']:+.6%} vs turnover.py T7 {ref['cagr']:+.6%} (차이 {d:.1e})")
    if d > 1e-12:
        raise RuntimeError("보유 20이 turnover.py T7과 다르다")

    base = load("N20")
    cands = {n: load(f"N{n}")["cagr_2016_2020"] for n in NS if n != 20}
    pick = max(cands, key=cands.get)
    p = load(f"N{pick}")
    j = {"pick_by_2016_2020": pick,
         "(a) 2021~2025 CAGR > T7": bool(p["cagr_2021_2025"] > base["cagr_2021_2025"]),
         "(b) 2021~2025 MDD ≤ T7 + 5%p": bool(p["mdd_2021_2025"] <= base["mdd_2021_2025"] + 0.05),
         "(c) 10년 CAGR > T7 (참고)": bool(p["cagr"] > base["cagr"])}
    j["확인 통과"] = bool(j["(a) 2021~2025 CAGR > T7"] and j["(b) 2021~2025 MDD ≤ T7 + 5%p"])
    cols = ("cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "mdd_2016_2020", "mdd_2021_2025", "ir", "annual_turnover", "cost_est",
            "final_nav", "mean_cash", "excess_kospi", "ir_kospi", "excess_k200", "ir_k200")
    rows = [{"n": n, **{c: load(f"N{n}")[c] for c in cols}, **{f"w:{w}": v for w, v in load(f"N{n}")["windows"].items()},
             **{f"y{y}": v for y, v in load(f"N{n}")["yearly"].items()}} for n in NS]
    rd = pd.DataFrame(rows)
    rd.to_csv(HERE / "hold_runs.csv", index=False)
    (HERE / "hold_judgment.json").write_text(json.dumps({"band": BAND, "every": EVERY, "judgment": j,
                                                         "runs": {int(r["n"]): {c: r[c] for c in cols} for r in rows}},
                                                        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not ledger_has("result", "hold|판정"):
        for r in rows:
            ledger.result(f"hold|N{int(r['n'])}", **{c: r[c] for c in cols})
        ledger.result("hold|판정", **j)

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    show = rd.copy()
    for c in [c for c in show.columns if c.startswith(("cagr", "mdd", "cost_est", "mean_cash", "excess", "w:"))
              or (c.startswith("y") and c[1:].isdigit())]:
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(1)
    for c in ("ir", "ir_kospi", "ir_k200", "annual_turnover", "final_nav"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    print("\n[2016-01-11 ~ 2025-12-30, T7 규칙, 보유 수만 바꿈]")
    print(show[["n", "cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "mdd_2016_2020", "mdd_2021_2025", "annual_turnover",
                "cost_est", "final_nav", "excess_kospi", "ir_kospi", "excess_k200", "ir_k200"]].to_string(index=False))
    print("\n[하락·상승 구간 수익률 %]")
    print(show[["n"] + [c for c in show.columns if c.startswith("w:")]].to_string(index=False))
    print("\n[연도별 수익률 %]")
    print(show[["n"] + [c for c in show.columns if c.startswith("y") and c[1:].isdigit()]].to_string(index=False))
    print(f"\n[판정] (실행 전 고정) 2016~2020으로 고른 보유 수: {pick}")
    for q, v in j.items():
        if q.startswith("("):
            print(f"  {q}: {'만족' if v else '미달'}")
    print(f"  확인: {'통과' if j['확인 통과'] else '미달'}")
    print("done")


if __name__ == "__main__":
    main()
