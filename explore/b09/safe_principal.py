"""원금 보존 에이전트 2판 (11/19 사용자 결정): 원금 = 처음 투자한 돈, 바닥 = 그 95%, 현금 수익 0 (부록 1, 이자 가정 없음).

  python -I explore/b09/run.py safe_principal.py

구조 (모든 에이전트 공통)
- 자산: 코스피 200(지수 종가, 배당 제외)과 현금(수익 0). 매주 신호일 종가까지의 정보로 노출을 정하고 다음 거래일 종가에 맞춘다.
- 바닥 F = 0.95 × 처음 투자한 돈 (구간 내내 고정, 해마다 바꾸지 않는다). 처음에는 여유 5%로 시작하고, 이익이 쌓이면 여유가 커진다.
- 노출: E_t = clip(m_t × (NAV_t − F) / NAV_t, 0, 1). 에이전트는 배수 m_t만 정한다.
- 비용: 수수료 0.015% + 슬리피지 0.10% (매매 금액, 한쪽). ETF라 거래세 0. '주식 비용' 판은 매도세 0.2%를 더한다.

에이전트 (safe_agent.py와 같은 정의, 실행 전 고정)
- A0 고정 배수 m = 4 / A1 추세: 200일선 위 8, 아래 2 / A2 변동성: clip(√52 / (4σ_t), 2, 10) / A3 추세 + 변동성: 200일선 위 A2, 아래 2

구간과 판정 (실행 전 고정, 11/19)
- 설계 구간 2016~2025 (이미 본 구간), 시험 구간 2011~2015 (처음 보는 구간). 구간마다 처음 투자한 돈 1에서 시작.
- 선택: 설계 구간에서 바닥을 한 번도 뚫지 않은 것(일간 NAV ≥ F) 중 A1~A3의 CAGR 최고.
- 교체 기준: 고른 에이전트가 시험 구간에서 (1) 바닥을 한 번도 뚫지 않음, (2) 끝 잔고 ≥ 처음 투자한 돈 (CAGR ≥ 0), (3) CAGR > A0,
  그리고 설계 구간에서 (4) CAGR > A0. 모두 만족하면 B-09 최종 모델(탐색)을 이 에이전트로 바꾼다.
- 결과: safe_principal.json, safe_principal.csv, 원장.
"""
import json
import math

from common import HERE, Ledger
import numpy as np
import pandas as pd
from src.backtest import metrics as m5
from src.common import io
from src.portfolio.env_check_report import FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE, kospi200, signal_dates
from src.portfolio.params import PREREG_PATH, RiskLayerParams, load_prereg
from src.portfolio.risk_layer import market_state_table

F0 = 0.95
cfg = load_prereg(PREREG_PATH)
CASH_RETURN = float(cfg["costs"]["cash_return"])
assert CASH_RETURN == 0.0, "부록 1 cash_return이 0이 아니다"
risk = RiskLayerParams.from_prereg(cfg)
P = kospi200(io.read_table("index")).sort_index()
DAYS = P.index
MA = P.rolling(200, min_periods=200).mean()
SIGMA = market_state_table(P, risk)["sigma"]
DATES = [pd.Timestamp(d) for d in signal_dates(io.read_table("universe"), DAYS, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)]
RATE = float(cfg["costs"]["commission_per_side"]) + float(cfg["costs"]["slippage_per_side"])
TAX = float(cfg["costs"]["sell_tax"])


def m_vol(t):
    s = SIGMA.get(t, np.nan)
    return 4.0 if not np.isfinite(s) or s <= 0 else float(np.clip(math.sqrt(52) / (4.0 * s), 2.0, 10.0))


def up(t):
    return None if not np.isfinite(MA.get(t, np.nan)) else bool(P[t] >= MA[t])


AGENTS = {
    "A0 고정 배수": lambda t: 4.0,
    "A1 추세 배수": lambda t: 4.0 if up(t) is None else (8.0 if up(t) else 2.0),
    "A2 변동성 배수": m_vol,
    "A3 추세 + 변동성": lambda t: 4.0 if up(t) is None else (m_vol(t) if up(t) else 2.0),
}
BENCH = ("지수 그대로", "현금만")


def simulate(name, dates, end_day, tax):
    idx = DAYS.get_indexer(dates)
    fills = {DAYS[i + 1]: d for i, d in zip(idx, dates) if 0 <= i and i + 1 < len(DAYS) and DAYS[i + 1] <= end_day}
    sig = set(dates)
    units, cash, floor = 0.0, 1.0, F0
    target, nav_d, exps, turns = None, {}, [], []
    for day in DAYS[(DAYS >= dates[0]) & (DAYS <= end_day)]:
        nav = units * P[day] + cash  # 현금 수익 0
        if day in fills and target is not None:
            w = units * P[day] / nav
            trade = (target - w) * nav
            nav -= abs(trade) * RATE + max(0.0, -trade) * tax
            units = target * nav / P[day]
            cash = nav - units * P[day]
            turns.append(abs(target - w))
        nav_d[day] = nav
        if day in sig:
            if name == "지수 그대로":
                target = 1.0
            elif name == "현금만":
                target = 0.0
            else:
                target = float(np.clip(AGENTS[name](day) * (nav - floor) / nav, 0.0, 1.0))
            exps.append(target)
    nav = pd.Series(nav_d)
    rep = nav[nav.index >= min(fills)]
    n = len(rep) - 1
    last = rep.groupby(rep.index.year).last()
    yearly = last / last.shift(1).fillna(1.0) - 1.0
    bl = P.reindex(rep.index)
    b_last = bl.groupby(bl.index.year).last()
    b_year = b_last / b_last.shift(1).fillna(float(P[P.index < rep.index[0]].iloc[-1])) - 1.0
    upy = b_year > 0
    return {
        "cagr": m5.cagr(np.r_[1.0, rep.to_numpy()[1:]], trading_days=n), "mdd": m5.max_drawdown(rep.to_numpy()),
        "final_nav": float(rep.iloc[-1]), "min_nav": float(nav.min()),
        "floor_breach_days": int((nav < F0 - 1e-12).sum()),  # 일간 NAV가 바닥(처음 투자한 돈의 95%) 아래였던 날 수
        "below_principal_days": int((nav < 1.0 - 1e-12).sum()),  # 처음 투자한 돈 아래였던 날 수 (참고)
        "negative_years": int((yearly < -1e-12).sum()), "worst_year": float(yearly.min()),
        "mean_exposure": float(np.mean(exps)), "annual_turnover": float(np.sum(turns) / (n / 252.0)),
        "up_capture": float(yearly[upy].sum() / b_year[upy].sum()) if upy.any() else None,
        "yearly": {int(y): float(v) for y, v in yearly.items()},
    }


WINDOWS = {"시험 2011-2015": (2011, 2015), "설계 2016-2025": (2016, 2025), "설계 2016-2020": (2016, 2020),
           "설계 2021-2025": (2021, 2025)}
ledger = Ledger(HERE / "ledger.jsonl")
for a in AGENTS:
    ledger.plan(f"safe2|{a}", "원금 보존 에이전트", "코스피 200 + 바닥 0.95 × 처음 투자한 돈(고정) + 배수 규칙, 현금 수익 0",
                "사용자 결정(11/19): 원금 = 처음 투자한 돈, 손실 한도 5%, 이자 가정 없음. 시험 구간 2011~2015")
out, rows = {}, []
for wname, (y0, y1) in WINDOWS.items():
    dates = [d for d in DATES if y0 <= d.year <= y1]
    end_day = DAYS[DAYS <= pd.Timestamp(f"{y1}-12-31")][-1]
    for cost, tax in (("ETF 비용", 0.0), ("주식 비용", TAX)):
        for name in list(BENCH) + list(AGENTS):
            res = simulate(name, dates, end_day, tax)
            out[f"{wname} | {name} | {cost}"] = res
            rows.append({"window": wname, "agent": name, "cost": cost, **{k: v for k, v in res.items() if k != "yearly"},
                         **{f"y{y}": v for y, v in res["yearly"].items()}})
            if name in AGENTS and cost == "ETF 비용":
                ledger.result(f"safe2|{name}", years=wname, **{k: v for k, v in res.items() if k != "yearly"})
df = pd.DataFrame(rows)
df.to_csv(HERE / "safe_principal.csv", index=False)
(HERE / "safe_principal.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

main = df[df.cost == "ETF 비용"]
cols = ["agent", "cagr", "final_nav", "mdd", "min_nav", "floor_breach_days", "negative_years", "worst_year", "up_capture", "mean_exposure"]
for w in WINDOWS:
    print(f"\n[{w}] ETF 비용 (바닥 = 처음 투자한 돈의 95%)")
    t = main[main.window == w][cols].copy()
    for c in ("cagr", "mdd", "worst_year", "up_capture", "mean_exposure"):
        t[c] = (t[c] * 100).round(1)
    for c in ("final_nav", "min_nav"):
        t[c] = t[c].round(3)
    print(t.to_string(index=False))
yc = [c for c in df.columns if c.startswith("y") and c[1:].isdigit()]
for w in ("설계 2016-2025", "시험 2011-2015"):
    print(f"\n[{w}] 연도별 수익률 % (ETF 비용)")
    print((main[main.window == w].set_index("agent")[yc].dropna(axis=1, how="all") * 100).round(1).to_string())

design = main[(main.window == "설계 2016-2025") & main.agent.isin(list(AGENTS)[1:]) & (main.floor_breach_days == 0)]
get = lambda w, a: main[(main.window == w) & (main.agent == a)].iloc[0]
if design.empty:
    print("\n선택: 설계 구간에서 바닥을 안 뚫은 A1~A3가 없다 → 교체하지 않음")
    ledger.result("safe2|선택", pick=None, replace=False)
else:
    pick = design.sort_values("cagr", ascending=False).iloc[0]["agent"]
    tp, t0 = get("시험 2011-2015", pick), get("시험 2011-2015", "A0 고정 배수")
    dp, d0 = get("설계 2016-2025", pick), get("설계 2016-2025", "A0 고정 배수")
    checks = {"시험 바닥 안 뚫음": bool(tp.floor_breach_days == 0), "시험 끝 잔고 ≥ 원금": bool(tp.final_nav >= 1.0),
              "시험 CAGR > A0": bool(tp.cagr > t0.cagr), "설계 CAGR > A0": bool(dp.cagr > d0.cagr)}
    print(f"\n선택 (설계 구간 규칙): {pick}")
    for k, v in checks.items():
        print(f"  {k}: {'만족' if v else '미달'}")
    print("교체 기준:", "모두 만족 → B-09 최종 모델 교체" if all(checks.values()) else "미달 → 교체하지 않음")
    ledger.result("safe2|선택", pick=pick, checks=checks, replace=all(checks.values()))
print("done")
