r"""같은 현금 비교: 공식 정책(B-07 PPO)의 작은 MDD는 현금 때문인가 (11/20 사용자 결정: 24절 뒤 "엔진 돌려서 정식으로 확인하자").

배경 (docs 24절 참고 줄, 엔진 밖 근사): 공식 정책은 RL 세 판 중에서만 가장 높다. 공식 정책과 같은 평균 현금 57%를 두고
나머지를 매일 같은 비중으로 들면 2020~2025 코스피 종합 +5.6%·MDD 17.0%, T7 보유 80 +4.9%·19.7%로, 공식 정책
(24절 평가 −1.1%·28.8%, 10년 실행 부분 −2.0%·30.2%)보다 CAGR이 높고 MDD가 작았다.
이것을 C 엔진(비용·매도세·체결 제약 포함)으로 10년 전체에서 확인한다.

실행 (저장소 루트, PowerShell, 사용자가 돌린다, 5분 안팎, 창 1개):
  .\.venv\Scripts\python.exe -I explore\b09\run.py samecash.py
  끊기면 같은 명령을 다시 돌린다 (끝난 실행은 건너뛴다).

구성 (실행 전 고정 11/20). 2016-01-11 ~ 2025-12-30 연속, A v7 시험 신호, C 엔진, M9.
- 공식 정책: B-07 PPO 5시드의 공식 10년 실행 (reports/20261118_B_buffer_wf, 사전등록 M8). 다시 돌리지 않고 저장된
  일별 NAV와 지표를 쓴다 (11/20 확인: 일별 NAV로 잰 CAGR·MDD·끝 잔고가 metrics.json과 같다).
- c* = 공식 5시드의 10년 평균 현금 (metrics.json runs.rl_s0~4.mean_cash의 평균 = 0.533893). 실행 때 다시 계산해 다르면 멈춘다.
- T7 보유 80 + 현금: 22절 보유 80 (4주 평균 순위, 진입 rank ≤ 80, 청산 rank > 140, 격주, 같은 비중, crash_flag 차단,
  종목 10%·업종 30% 한도, M9) 그대로 두고, 리밸런싱 주의 목표 비중에 E = 1 − c* (= 0.466107)를 곱한다.
  사이 주는 들고 있는 비중을 그대로 둔다 (crash_flag·M9 쿨다운 종목은 판다).
- 대조: 같은 경로(mode full)에 E = 1을 넣은 판이 22절 보유 80과 같아야 한다 (10년 CAGR 차 > 1e-12면 멈춘다).
  common.buffer_run의 사이 주 처리를 11/20에 고쳤으므로(E_t를 두 번 곱하던 것), E = 1 실행이 그대로인지도 이 대조로 본다.
- 참고 (엔진 밖, 판정에 안 씀): 코스피 종합 100%, 코스피 종합 (1 − c*) + 현금 (매일 같은 비중, 현금 수익 0, 비용 없음).

판정 (실행 전 고정 11/20). 공식 = 5시드 값의 평균.
- (1) 10년: T7 보유 80 + 현금의 CAGR > 공식이고 MDD < 공식
- (2) 2016~2020과 2021~2025 각각 (1)과 같은 조건 (10년 연속 실행을 2020년 마지막 거래일에서 나눔)
- (3) 비교 성립: T7 보유 80 + 현금의 실현 평균 현금이 c* ± 2%p 안. 밖이면 (1)·(2) 대신 '비교 불성립'으로 적는다.
- (3)이 성립하고 (1)·(2)를 모두 만족하면 '같은 현금이면 T7 보유 80이 공식 정책보다 CAGR·MDD 모두 낫다'를 확인한 것이다.
- 참고: 공식 시드별 비교(CAGR·MDD 모두 이긴 시드 수), 2020~2025 구간(24절과 같은 기간), 연도별, 회전율, 비용 추정,
  코스피 종합·200 대비 연 초과와 IR.
- 주의: T7과 보유 80은 2016~2025를 보고 고른 구성이라(16·22절) 사전등록된 공식 정책보다 유리한 쪽으로 치우친다.
  확인해도 forward test(PM이 허락하면 2026 홀드아웃)가 필요하다.
- 탐색 결과다 (R-15). 결과: samecash_runs.csv, samecash_judgment.json, runs/samecash/, 원장 samecash|*.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

from bench_kospi import END, KOSPI_PATH, stats as kospi_stats
from common import ALL_YEARS, HERE, REPO, Ctx, Ledger
import exitrule as ex
import numpy as np
import pandas as pd
from turnover import ledger_has, run, smooth_signals, split_cagr, yearly

OUT = HERE / "runs" / "samecash"
HOLD = HERE / "runs" / "hold"
OFFICIAL = REPO / "reports" / "20261118_B_buffer_wf"
C_STAR = 0.533893  # 공식 5시드 10년 평균 현금 (11/20 실행 전 계산)
CASH_TOL = 0.02
N, BAND, EVERY = 80, 60, 2
SEEDS = tuple(range(5))
FROM_2020 = pd.Timestamp("2020-01-06")  # 24절 평가와 같은 기간의 첫 체결일
TRACK = "같은 현금 비교 (T7 보유 80 vs 공식 정책)"
NUM = ("cagr", "mdd", "cagr_2016_2020", "cagr_2021_2025", "mdd_2016_2020", "mdd_2021_2025", "cagr_2020_2025",
       "mdd_2020_2025", "mean_cash", "annual_turnover", "cost_est", "final_nav", "excess_kospi", "ir_kospi",
       "excess_k200", "ir_k200")


def save(name: str, rec: dict, nav: pd.Series) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": nav.index, "nav": nav.to_numpy()}).to_parquet(OUT / f"nav_{name}.parquet", index=False)
    (OUT / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, default=float), encoding="utf-8")


def load(name: str):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def as_ns(s: pd.Series) -> pd.Series:
    s = s.astype("float64").copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).as_unit("ns")
    return s


def official_c_star() -> float:
    m = json.loads((OFFICIAL / "metrics.json").read_text(encoding="utf-8"))
    return float(np.mean([m["runs"][f"rl_s{s}"]["mean_cash"] for s in SEEDS]))


def describe(nav: pd.Series, bms: dict, fills: pd.DatetimeIndex, fills20: pd.DatetimeIndex) -> dict:
    """10년 연속 실행의 일별 NAV(체결 전 잔고 1)로 10년·두 구간·2020~2025 값을 잰다 (18·24절과 같은 함수)."""
    k10 = kospi_stats(nav, bms, fills)
    c1, c2 = split_cagr(nav)
    d1, d2 = ex.half_mdd(nav)
    span = nav[(nav.index >= fills20[0]) & (nav.index <= END)] / nav[fills20[0]]
    k20 = kospi_stats(span, bms, fills20)
    return {"cagr": k10["cagr"], "mdd": k10["mdd"], "final_nav": k10["final_nav"],
            "cagr_2016_2020": c1, "cagr_2021_2025": c2, "mdd_2016_2020": d1, "mdd_2021_2025": d2,
            "cagr_2020_2025": k20["cagr"], "mdd_2020_2025": k20["mdd"],
            "excess_kospi": k10["excess_kospi"], "ir_kospi": k10["ir_kospi"],
            "excess_k200": k10["excess_k200"], "ir_k200": k10["ir_k200"],
            "windows": ex.windows(nav), "yearly": yearly(nav)}


def main() -> None:
    c_star = official_c_star()
    if abs(c_star - C_STAR) > 1e-6:
        raise RuntimeError(f"공식 평균 현금 {c_star:.6f}이 실행 전에 정한 {C_STAR}와 다르다")
    w = 1.0 - C_STAR
    variants = {"e1": ("T7 보유 80, E = 1 (대조, 22절과 같아야 함)", 1.0),
                "cash": (f"T7 보유 80 + 현금 {C_STAR:.1%} (E = {w:.3f})", w)}
    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("plan", "samecash|cash"):  # 실행 전 계획 행 (처음 한 번)
        why = ("사용자 결정(11/20): 24절 뒤 '공식 정책의 작은 MDD는 현금 때문'(엔진 밖 근사)을 C 엔진으로 확인. "
               "c* = 공식 5시드 10년 평균 현금")
        ledger.plan("samecash|e1", TRACK, "대조: T7 보유 80, mode full 경로에 E = 1, 22절 보유 80과 같아야 함", why)
        ledger.plan("samecash|cash", TRACK,
                    f"T7 보유 80 (진입 ≤ {N}, 청산 > {N + BAND}, 4주 평균 순위, 격주, M9, C 엔진, 2016~2025), "
                    f"리밸런싱 주 E = 1 − c* = {w:.6f} (c* = {C_STAR})", why)

    t0 = time.time()
    ctx = Ctx(ALL_YEARS)
    cost_unit = 2 * (float(ctx.cfg["costs"]["commission_per_side"]) + float(ctx.cfg["costs"]["slippage_per_side"])) \
        + float(ctx.cfg["costs"]["sell_tax"])
    snaps = ctx.snaps(signals=smooth_signals(ctx.signals, 4), key="sm4")
    k = pd.read_parquet(KOSPI_PATH)
    kospi = pd.Series(k["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(k["date"])).as_unit("ns"))
    k200 = as_ns(ctx.inputs.bm_close)
    cal = k200.index[k200.index <= END]
    fills = pd.DatetimeIndex([cal[cal.searchsorted(d, side="right")] for d in pd.to_datetime(ctx.dates)])
    fills20 = fills[fills >= FROM_2020]
    bms = {"kospi": kospi, "k200": k200}
    print(f"입력 읽음 ({time.time() - t0:.0f}초). c* = {C_STAR}, E = {w:.6f}", flush=True)

    b = replace(ctx.buffer, enter_rank_max=N, exit_rank_greater_than=N + BAND, max_holdings=N)
    for name, (label, e) in variants.items():
        if load(name) is not None:
            continue
        t = time.time()
        m, nav = run(ctx, snaps, b, EVERY, [e] * len(snaps))
        nav = as_ns(nav)
        rec = {"name": name, "label": label, "e": e, "ir": m["ir"], "annual_turnover": m["annual_turnover"],
               "mean_cash": m["mean_cash"], "cost_est": m["annual_turnover"] * cost_unit, "summary_cagr": m["cagr"],
               **describe(nav, bms, fills, fills20)}
        save(name, rec, nav)
        print(f"  {label}: CAGR {rec['cagr']:+.2%} (2016~2020 {rec['cagr_2016_2020']:+.2%}, 2021~2025 "
              f"{rec['cagr_2021_2025']:+.2%}), MDD {rec['mdd']:.1%}, 평균 현금 {rec['mean_cash']:.1%}, "
              f"회전율 {rec['annual_turnover']:.1f} ({time.time() - t:.0f}초)", flush=True)

    ref = json.loads((HOLD / "N80.json").read_text(encoding="utf-8"))
    d = abs(load("e1")["summary_cagr"] - ref["cagr"])
    print(f"대조: E = 1 판 CAGR {load('e1')['summary_cagr']:+.6%} vs 22절 보유 80 {ref['cagr']:+.6%} (차이 {d:.1e})")
    if d > 1e-12:
        raise RuntimeError("E = 1 판이 22절 보유 80과 다르다")

    # 공식 정책 5시드 (저장된 10년 실행)
    dn = pd.read_csv(OFFICIAL / "daily_nav.csv", parse_dates=["date"]).set_index("date")
    dn.index = pd.DatetimeIndex(dn.index).as_unit("ns")
    mo = json.loads((OFFICIAL / "metrics.json").read_text(encoding="utf-8"))
    off = []
    for s in SEEDS:
        r = mo["runs"][f"rl_s{s}"]
        q = {"name": f"official_s{s}", "label": f"공식 정책 시드 {s}", "ir": r["ir"], "annual_turnover": r["annual_turnover"],
             "mean_cash": r["mean_cash"], "cost_est": r["annual_turnover"] * cost_unit,
             **describe(as_ns(dn[f"rl_s{s}"]), bms, fills, fills20)}
        if abs(q["cagr"] - r["cagr"]) > 1e-8 or abs(q["mdd"] - r["mdd"]) > 1e-8:
            raise RuntimeError(f"공식 시드 {s}: 일별 NAV로 잰 값이 metrics.json과 다르다")
        off.append(q)
    om = {"name": "official_mean", "label": "공식 정책 5시드 평균", **{c: float(np.mean([q[c] for q in off])) for c in NUM + ("ir",)},
          "windows": {kk: float(np.mean([q["windows"][kk] for q in off])) for kk in off[0]["windows"]},
          "yearly": {y: float(np.mean([q["yearly"][y] for q in off])) for y in off[0]["yearly"]}}

    # 참고 (엔진 밖): 코스피 종합 100%와 (1 − c*) + 현금, 첫 체결일 종가 = 1
    kn = kospi[(kospi.index >= fills[0]) & (kospi.index <= END)]
    kn = kn / kn.iloc[0]
    kmix = (1.0 + w * kn.pct_change().fillna(0.0)).cumprod()
    idx_rows = [{"name": "kospi", "label": "코스피 종합 100% (참고, 엔진 밖)", "mean_cash": 0.0, **describe(kn, bms, fills, fills20)},
                {"name": "kospi_mix", "label": f"코스피 종합 {w:.1%} + 현금 (참고, 엔진 밖, 비용 없음)", "mean_cash": C_STAR,
                 **describe(kmix, bms, fills, fills20)}]

    T = load("cash")
    valid = bool(abs(T["mean_cash"] - C_STAR) <= CASH_TOL)
    j = {"c_star": C_STAR, "e": w, "실현 평균 현금": T["mean_cash"], "(3) 비교 성립 (실현 평균 현금 c* ± 2%p)": valid}
    if valid:
        j["(1) 10년 CAGR > 공식, MDD < 공식"] = bool(T["cagr"] > om["cagr"] and T["mdd"] < om["mdd"])
        j["(2) 2016~2020 CAGR > 공식, MDD < 공식"] = bool(T["cagr_2016_2020"] > om["cagr_2016_2020"]
                                                      and T["mdd_2016_2020"] < om["mdd_2016_2020"])
        j["(2) 2021~2025 CAGR > 공식, MDD < 공식"] = bool(T["cagr_2021_2025"] > om["cagr_2021_2025"]
                                                      and T["mdd_2021_2025"] < om["mdd_2021_2025"])
        j["확인"] = bool(j["(1) 10년 CAGR > 공식, MDD < 공식"] and j["(2) 2016~2020 CAGR > 공식, MDD < 공식"]
                       and j["(2) 2021~2025 CAGR > 공식, MDD < 공식"])
    else:
        j["확인"] = "비교 불성립"
    j["참고: 10년 CAGR·MDD 모두 이긴 공식 시드 수"] = int(sum(T["cagr"] > q["cagr"] and T["mdd"] < q["mdd"] for q in off))
    j["참고: 2020~2025 (24절 기간)"] = {"T7 보유 80 + 현금": [T["cagr_2020_2025"], T["mdd_2020_2025"]],
                                   "공식 5시드 평균 (10년 실행 부분)": [om["cagr_2020_2025"], om["mdd_2020_2025"]]}

    rows = idx_rows + [load("e1"), T] + off + [om]
    rd = pd.DataFrame([{"name": r["name"], "label": r["label"], **{c: r.get(c) for c in NUM},
                        **{f"w:{kk}": v for kk, v in r["windows"].items()},
                        **{f"y{y}": v for y, v in r["yearly"].items()}} for r in rows])
    rd.to_csv(HERE / "samecash_runs.csv", index=False)
    (HERE / "samecash_judgment.json").write_text(json.dumps({
        "period": [str(fills[0].date()), str(END.date())], "judgment": j,
        "runs": {r["name"]: {c: r.get(c) for c in NUM} for r in rows}}, ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    if not ledger_has("result", "samecash|판정"):
        for r in (load("e1"), T, om):
            ledger.result(f"samecash|{r['name']}", **{c: r.get(c) for c in NUM})
        ledger.result("samecash|판정", **j)

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    show = rd.copy()
    for c in [c for c in show.columns if c.startswith(("cagr", "mdd", "mean_cash", "cost_est", "excess", "w:"))
              or (c.startswith("y") and c[1:].isdigit())]:
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(1)
    for c in ("ir_kospi", "ir_k200", "annual_turnover", "final_nav"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    print(f"\n[{fills[0].date()} ~ {END.date()}, 10년 연속, C 엔진 (코스피 행은 엔진 밖 참고)] %")
    print(show[["label", "cagr", "mdd", "cagr_2016_2020", "cagr_2021_2025", "mdd_2016_2020", "mdd_2021_2025",
                "cagr_2020_2025", "mdd_2020_2025", "mean_cash", "annual_turnover", "final_nav", "excess_kospi",
                "ir_kospi"]].to_string(index=False))
    print("\n[하락·상승 구간 수익률 %]")
    print(show[["label"] + [c for c in show.columns if c.startswith("w:")]].to_string(index=False))
    print("\n[연도별 수익률 %]")
    print(show[["label"] + [c for c in show.columns if c.startswith("y") and c[1:].isdigit()]].to_string(index=False))
    print(f"\n[판정] (실행 전 고정) c* = {C_STAR:.1%}, 실현 평균 현금 {T['mean_cash']:.1%}")
    for q, v in j.items():
        if q.startswith("("):
            print(f"  {q}: {'만족' if v is True else ('미달' if v is False else v)}")
    print(f"  확인: {j['확인'] if isinstance(j['확인'], str) else ('통과' if j['확인'] else '미달')}")
    print(f"  참고: 10년 CAGR·MDD 모두 이긴 공식 시드 {j['참고: 10년 CAGR·MDD 모두 이긴 공식 시드 수']}/5")
    print("done")


if __name__ == "__main__":
    main()
