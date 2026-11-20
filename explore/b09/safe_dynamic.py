"""동적 비율 규칙의 15년 확인 (11/19 사용자 질문 '동적으로 비율 정하는 방법은?').

  python -I explore/b09/run.py safe_dynamic.py

무엇을 보나
- index_overlay.py(11/19)의 지수 규칙 7개는 2016~2025에서만 돌렸다. 2011~2015에서는 한 번도 재지 않았다.
  규칙 정의를 그대로 두고(새 파라미터 없음) 2011~2015와 15년 전체에서 버티는지 본다.
  다만 2011~2015 지수 흐름 자체는 이미 봤다 (다른 트랙에서). 그래서 완전히 처음 보는 구간은 아니다.
- 비교: 주식 비중 고정(30%·50%·70%, 매주 맞춤). 새로 넣은 참고 행이라 고르는 데 쓰지 않는다.

규칙 (index_overlay.py와 같은 정의)
- 지수 그대로: E = 1
- 사전등록 M8: E_t = clip(0.157263 / σ_t, 0.3, 1), 모멘텀 붕괴 국면이면 ≤ 0.7 (부록 1 값.
  0.157263은 2011~2015 평균 변동성으로 정한 값이다. 수익이 아니라 변동성만 쓴다)
- M-tr-b: 200일선 아래면 min(E_t, 0.3), 위면 E_t
- M-tr-c: 200일선 아래면 0.3, 위면 1 (국면이면 0.7)
- M-dd-0.15-f0.3: E_t, 자기 NAV 낙폭 > 15%면 × 0.3
- M-dd+tr: M-tr-b 다음 낙폭 > 15%면 × 0.5
- M-tr-0: 200일선 위면 1, 아래면 0
- E_t·국면은 risk_layer.market_state_table(지수 종가만 쓴다)로 계산한다. index_overlay는 같은 값을 스냅샷에서 읽었다.
- 자산 코스피 200(ETF) + 현금(수익 0). 신호일 종가로 정하고 다음 거래일 종가에 맞춘다. ETF 비용(수수료 0.015% + 슬리피지 0.10%).

대조: 2016-2025를 따로 돌려 index_overlay.csv(ETF 비용, 이자 0)의 CAGR·MDD·평균 노출과 같은지 본다 (차이 < 1e-9).

판정 (실행 전 고정, 11/19)
- 실행: safe_drawdown과 같은 17개 (전체 2011-2025, 5년 3개, 시작 연도별 3년 13개).
- 버팀 (규칙 7개 중 '지수 그대로' 제외 6개): 2011-2015에서 최대 낙폭 < 지수, 그리고 CAGR ≥ 지수 (지수보다 덜 잃음).
- 선택: 버팀을 만족하고 전체 15년 최대 낙폭 < 지수인 규칙 중 15년 CAGR ÷ 최대 낙폭이 가장 큰 것.
- 탐색 결과다 (R-15). 운용은 PM 결정이 필요하다. 결과: safe_dynamic.csv, safe_dynamic.json, 원장.
"""
import json

from common import HERE, Ledger
import numpy as np
import pandas as pd
from safe_drawdown import DATES, DAYS, MAIN, PX, RATE, RUNS, STARTS, risk, P, window
from src.backtest import metrics as m5
from src.portfolio.risk_layer import market_state_table

MS = market_state_table(P, risk).reindex(DAYS)
EXP = dict(zip(DAYS, MS["exposure"].to_numpy(dtype=float)))
REG = dict(zip(DAYS, MS["regime_mc"].to_numpy(dtype=float)))
MA = dict(zip(DAYS, P.rolling(200, min_periods=200).mean().to_numpy(dtype=float)))
CAP = float(risk.momentum_crash_regime.exposure_cap)


def below(t):
    return bool(PX[t] < MA[t])  # MA가 NaN이면 False (index_overlay와 같다)


RULES = {
    "지수 그대로": lambda t, st: 1.0,
    "사전등록 M8": lambda t, st: EXP[t],
    "M-tr-b": lambda t, st: min(EXP[t], 0.3) if below(t) else EXP[t],
    "M-tr-c": lambda t, st: 0.3 if below(t) else (1.0 if REG[t] == 0 else min(1.0, CAP)),
    "M-dd-0.15-f0.3": lambda t, st: EXP[t] * (0.3 if st["dd"] > 0.15 else 1.0),
    "M-dd+tr": lambda t, st: (min(EXP[t], 0.3) if below(t) else EXP[t]) * (0.5 if st["dd"] > 0.15 else 1.0),
    "M-tr-0": lambda t, st: 0.0 if below(t) else 1.0,
}
FIXED = {f"고정 {w:.0%}": (lambda w: (lambda t, st: w))(w) for w in (0.3, 0.5, 0.7)}
ALL = {**RULES, **FIXED}


def simulate(rule, dates, end_day, fn=None, series=False):
    """index_overlay.simulate와 같은 체결·비용 (이자 0, ETF 비용).
    fn: 노출 함수 (없으면 ALL[rule]). series: 일별 잔고(_nav)·주식 비중(_w)도 돌려준다 (safe_phases.py)."""
    fn = fn or ALL[rule]
    idx = DAYS.get_indexer(dates)
    fills = {DAYS[i + 1]: d for i, d in zip(idx, dates) if i + 1 < len(DAYS) and DAYS[i + 1] <= end_day}
    sig = set(dates)
    units, cash, peak = 0.0, 1.0, 1.0
    target, nav_d, w_d, exps, turns = None, {}, {}, [], []
    for day in DAYS[(DAYS >= dates[0]) & (DAYS <= end_day)]:
        px = PX[day]
        nav = units * px + cash
        if day in fills and target is not None:
            w = units * px / nav
            trade = (target - w) * nav
            nav -= abs(trade) * RATE
            units = target * nav / px
            cash = nav - units * px
            turns.append(abs(target - w))
        peak = max(peak, nav)
        nav_d[day], w_d[day] = nav, units * px / nav
        if day in sig:
            e = fn(day, {"nav": nav, "dd": 1.0 - nav / peak})
            if not np.isfinite(e):
                raise ValueError(f"{rule} {day.date()} 노출이 NaN (시장 상태 이력 부족)")
            target = float(np.clip(e, 0.0, 1.0))
            exps.append(target)
    nav = pd.Series(nav_d)
    rep = nav[nav.index >= min(fills)]
    n = len(rep) - 1
    last = rep.groupby(rep.index.year).last()
    yearly = last / last.shift(1).fillna(1.0) - 1.0
    res = {
        "cagr": m5.cagr(np.r_[1.0, rep.to_numpy()[1:]], trading_days=n),
        "mdd_io": m5.max_drawdown(rep.to_numpy()),  # index_overlay와 같은 정의 (대조용)
        "mdd": m5.max_drawdown(np.r_[1.0, nav.to_numpy()]),  # 시작 잔고 1 포함 (safe_* 표와 같은 정의)
        "final_nav": float(nav.iloc[-1]), "min_nav": float(nav.min()),
        "negative_years": int((yearly < -1e-12).sum()), "worst_year": float(yearly.min()),
        "mean_exposure": float(np.mean(exps)), "annual_turnover": float(np.sum(turns) / (n / 252.0)),
        "yearly": {int(y): float(v) for y, v in yearly.items()},
    }
    if series:
        res["_nav"], res["_w"] = nav, pd.Series(w_d)
    return res


def control():
    old = pd.read_csv(HERE / "index_overlay.csv", encoding="utf-8")
    rate = old["cash_rate"].fillna(0.0) if "cash_rate" in old else 0.0
    old = old[(old.window == "2016-2025") & (rate == 0.0) & (old.cost == "ETF 비용")].set_index("rule")
    dates, end_day = window(2016, 2025)
    worst = 0.0
    for rule in RULES:
        r = simulate(rule, dates, end_day)
        o = old.loc[rule]
        diff = max(abs(r["cagr"] - o.cagr), abs(r["mdd_io"] - o.mdd), abs(r["mean_exposure"] - o.mean_exposure))
        worst = max(worst, diff)
        print(f"  대조 {rule}: 최대 차이 {diff:.2e}")
    ok = worst < 1e-9
    print("대조:", "index_overlay와 같다" if ok else f"다르다 (최대 {worst:.2e}) → 결과를 믿기 전에 원인을 본다")
    return ok, worst


def main():
    ledger = Ledger(HERE / "ledger.jsonl")
    for rule in ALL:
        ledger.plan(f"safedyn|{rule}", "동적 비율 15년 확인", "index_overlay 규칙 정의 그대로 (고정 비중은 참고 행)",
                    "사용자 질문(11/19) '동적으로 비율 정하는 방법': 2016~2025에서만 잰 규칙을 2011~2015·15년에서 확인")
    ok, worst = control()
    ledger.result("safedyn|대조", same_as_index_overlay=ok, max_diff=worst)

    out, rows = {}, []
    for wname, (y0, y1) in RUNS.items():
        dates, end_day = window(y0, y1)
        for rule in ALL:
            res = simulate(rule, dates, end_day)
            out[(wname, rule)] = res
            rows.append({"window": wname, "rule": rule, **{k: v for k, v in res.items() if k != "yearly"},
                         **{f"y{y}": v for y, v in res["yearly"].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "safe_dynamic.csv", index=False)
    (HERE / "safe_dynamic.json").write_text(
        json.dumps({f"{w} | {m}": r for (w, m), r in out.items()}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    table = []
    for rule in ALL:
        full, test = out[("전체 2011-2025", rule)], out[("2011-2015", rule)]
        three = [out[(w, rule)] for w in STARTS]
        row = {"rule": rule, "cagr15": full["cagr"], "mdd15": full["mdd"], "calmar15": full["cagr"] / full["mdd"] if full["mdd"] > 0 else None,
               "final15": full["final_nav"], "mean_exposure": full["mean_exposure"], "loss_years": full["negative_years"],
               "worst_year": full["worst_year"], "turnover": full["annual_turnover"],
               **{f"cagr {w}": out[(w, rule)]["cagr"] for w in MAIN if w != "전체 2011-2025"},
               **{f"mdd {w}": out[(w, rule)]["mdd"] for w in MAIN if w != "전체 2011-2025"},
               "3y_above_principal": f"{sum(r['final_nav'] >= 1.0 for r in three)}/{len(three)}",
               "3y_worst_final": min(r["final_nav"] for r in three), "3y_max_mdd": max(r["mdd"] for r in three)}
        table.append(row)
        ledger.result(f"safedyn|{rule}", **{k: v for k, v in row.items() if k != "rule"})
    t = pd.DataFrame(table)
    show = t.copy()
    for c in show.columns:
        if c.startswith(("cagr", "mdd", "3y_max")) or c in ("mean_exposure", "worst_year"):
            show[c] = (show[c] * 100).round(2)
    for c in ("calmar15", "final15", "turnover", "3y_worst_final"):
        show[c] = show[c].astype(float).round(3)
    pd.set_option("display.width", 280)
    pd.set_option("display.max_columns", 40)
    print("\n[요약] % 단위: 수익률·낙폭·비중. calmar15 = 15년 CAGR ÷ 15년 최대 낙폭")
    print(show.to_string(index=False))

    yc = [c for c in df.columns if c.startswith("y") and c[1:].isdigit()]
    print("\n[전체 2011-2025] 연도별 수익률 %")
    print((df[df.window == "전체 2011-2025"].set_index("rule")[yc] * 100).round(1).to_string())

    ix_test, ix_full = out[("2011-2015", "지수 그대로")], out[("전체 2011-2025", "지수 그대로")]
    print("\n버팀 (2011-2015에서 지수보다 낙폭 작고 덜 잃음) · 15년 낙폭 < 지수:")
    elig = []
    for rule in list(RULES)[1:]:
        te, fu = out[("2011-2015", rule)], out[("전체 2011-2025", rule)]
        hold = te["mdd"] < ix_test["mdd"] and te["cagr"] >= ix_test["cagr"]
        lower = fu["mdd"] < ix_full["mdd"]
        print(f"  {rule}: 버팀 {'만족' if hold else '미달'} (2011-15 CAGR {te['cagr']:+.2%}, 낙폭 {te['mdd']:.1%}), "
              f"15년 낙폭 {fu['mdd']:.1%} {'<' if lower else '≥'} 지수 {ix_full['mdd']:.1%}")
        if hold and lower:
            elig.append(rule)
    if not elig:
        print("선택: 조건을 만족한 규칙이 없다")
        ledger.result("safedyn|선택", pick=None)
    else:
        pick = max(elig, key=lambda r: out[("전체 2011-2025", r)]["cagr"] / out[("전체 2011-2025", r)]["mdd"])
        f = out[("전체 2011-2025", pick)]
        print(f"선택: {pick} (15년 CAGR {f['cagr']:+.2%}, 최대 낙폭 {f['mdd']:.1%}, 끝 잔고 {f['final_nav']:.3f}; "
              f"지수 {ix_full['cagr']:+.2%}, {ix_full['mdd']:.1%})")
        ledger.result("safedyn|선택", pick=pick, eligible=elig)
    print("done")


if __name__ == "__main__":
    main()
