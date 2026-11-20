"""안전 제약 에이전트 (11/19 사용자 결정 ①): 투자는 하되 해마다 원금 바닥을 구조로 지키고, 시장 상태로 위험 예산을 쓴다.

  python -I explore/b09/run.py safe_agent.py

구조 (모든 에이전트 공통)
- 자산: 코스피 200(지수 종가, 배당 제외)과 현금(이자 r). 매주 신호일 종가까지의 정보로 노출을 정하고 다음 거래일 종가에 맞춘다.
- 바닥: F_t = 연초 NAV / (1 + r)^(연말까지 남은 기간). 연초 NAV는 전년 마지막 거래일 종가 NAV(첫해 1).
- 노출: E_t = clip(m_t × (NAV_t − F_t) / NAV_t, 0, 1). 에이전트는 배수 m_t만 정한다 (위험 예산을 얼마나 쓰는가).
- 비용: 수수료 0.015% + 슬리피지 0.10% (매매 금액, 한쪽). ETF라 거래세 0. '주식 비용' 판은 매도세 0.2%를 더한다.

에이전트 (실행 전 고정, 11/19)
- A0 고정 배수: m = 4 (index_overlay의 CPPI-1.0-4와 같다)
- A1 추세 배수: 코스피 200 ≥ 200일선이면 m = 8, 아래면 2
- A2 변동성 배수: m = clip(√52 / (4 σ_t), 2, 10). σ_t는 M8 Step 4의 sigma(20일 로그수익률 표준편차 × √252,
  risk_layer.market_state_table). '한 주에 4σ만큼 빠져도 바닥을 뚫지 않을 만큼만 건다'는 뜻이다.
- A3 추세 + 변동성: 200일선 위면 A2의 m, 아래면 2
- 이력이 모자라 200일선·σ가 없으면 m = 4.

구간과 판정 (실행 전 고정)
- 설계 구간 2016~2025 (이미 본 구간), 시험 구간 2011~2015 (지수 운용 규칙으로 처음 보는 구간). 구간마다 NAV 1에서 시작.
- 현금 이자는 연 2% 가정(주), 1%·3%(민감도). 실제 금리 자료가 아니다.
- 선택: 설계 구간 10년(이자 2%, ETF 비용)에서 A1~A3 중 손실 연도가 가장 적고, 같으면 CAGR이 높은 것.
- 교체 기준: 고른 에이전트가 시험 구간(이자 2%)에서 (1) 손실 연도 0, (2) CAGR이 A0보다 높음, (3) CAGR이 현금(2%)보다 높음을
  모두 만족하고, 설계 구간에서도 CAGR이 A0보다 높으면 B-09 최종 모델(탐색)을 이 에이전트로 바꾼다.
- 결과: safe_agent.json, safe_agent.csv, 원장.
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

cfg = load_prereg(PREREG_PATH)
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
BENCH = {"지수 그대로": lambda t: None, "현금만": lambda t: None}


def simulate(name, dates, end_day, r, tax):
    idx = DAYS.get_indexer(dates)
    fills = {DAYS[i + 1]: d for i, d in zip(idx, dates) if 0 <= i and i + 1 < len(DAYS) and DAYS[i + 1] <= end_day}
    sig = set(dates)
    units, cash = 0.0, 1.0
    y0, year, prev = 1.0, dates[0].year, 1.0
    target, nav_d, exps, turns, cush = None, {}, [], [], []
    for day in DAYS[(DAYS >= dates[0]) & (DAYS <= end_day)]:
        if day.year != year:
            y0, year = prev, day.year
        cash *= (1.0 + r) ** (1.0 / 252.0)
        nav = units * P[day] + cash
        if day in fills and target is not None:
            w = units * P[day] / nav
            trade = (target - w) * nav
            nav -= abs(trade) * RATE + max(0.0, -trade) * tax
            units = target * nav / P[day]
            cash = nav - units * P[day]
            turns.append(abs(target - w))
        nav_d[day] = nav
        prev = nav
        if day in sig:
            tau = max(0.0, (pd.Timestamp(f"{day.year}-12-31") - day).days / 365.0)
            floor = y0 / (1.0 + r) ** tau
            cush.append((nav - floor) / floor)
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
    b_first = float(P[P.index < rep.index[0]].iloc[-1])
    b_year = b_last / b_last.shift(1).fillna(b_first) - 1.0
    upy = b_year > 0
    return {
        "cagr": m5.cagr(np.r_[1.0, rep.to_numpy()[1:]], trading_days=n), "mdd": m5.max_drawdown(rep.to_numpy()),
        "negative_years": int((yearly < -1e-12).sum()), "worst_year": float(yearly.min()),
        "mean_exposure": float(np.mean(exps)), "annual_turnover": float(np.sum(turns) / (n / 252.0)),
        "min_cushion": float(np.min(cush)),  # 신호일 기준 (NAV − 바닥) / 바닥의 최솟값. 음수면 바닥 아래로 내려간 적이 있다
        "up_capture": float(yearly[upy].sum() / b_year[upy].sum()) if upy.any() else None,  # 지수 상승 해 수익의 몇 %를 땄나
        "yearly": {int(y): float(v) for y, v in yearly.items()},
        "index_yearly": {int(y): float(v) for y, v in b_year.items()},
    }


WINDOWS = {"시험 2011-2015": (2011, 2015), "설계 2016-2025": (2016, 2025), "설계 2016-2020": (2016, 2020),
           "설계 2021-2025": (2021, 2025)}
RATES = (0.02, 0.01, 0.03)
ledger = Ledger(HERE / "ledger.jsonl")
for a in AGENTS:
    ledger.plan(f"safe|{a}", "원금 보존 에이전트", "코스피 200 + 연 단위 원금 바닥 + 배수 규칙 (머리말)",
                "사용자 결정 ①: 투자하면서 원금 보존, 시험 구간 2011~2015 (설계 구간은 이미 본 2016~2025)")
out, rows = {}, []
for wname, (y0, y1) in WINDOWS.items():
    dates = [d for d in DATES if y0 <= d.year <= y1]
    end_day = DAYS[DAYS <= pd.Timestamp(f"{y1}-12-31")][-1]
    for r in RATES:
        for cost, tax in (("ETF 비용", 0.0), ("주식 비용", TAX)):
            for name in list(BENCH) + list(AGENTS):
                res = simulate(name, dates, end_day, r, tax)
                out[f"{wname} | {name} | 이자 {r:.0%} | {cost}"] = res
                rows.append({"window": wname, "agent": name, "cash_rate": r, "cost": cost,
                             **{k: v for k, v in res.items() if k not in ("yearly", "index_yearly")},
                             **{f"y{y}": v for y, v in res["yearly"].items()}})
                if name in AGENTS and r == 0.02 and cost == "ETF 비용":
                    ledger.result(f"safe|{name}", years=wname, **{k: v for k, v in res.items() if k not in ("yearly", "index_yearly")})
df = pd.DataFrame(rows)
df.to_csv(HERE / "safe_agent.csv", index=False)
(HERE / "safe_agent.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

main = df[(df.cash_rate == 0.02) & (df.cost == "ETF 비용")]
cols = ["agent", "cagr", "mdd", "negative_years", "worst_year", "up_capture", "mean_exposure", "min_cushion"]
for w in WINDOWS:
    print(f"\n[{w}] 이자 2%, ETF 비용")
    t = main[main.window == w][cols].copy()
    for c in ("cagr", "mdd", "worst_year", "up_capture", "mean_exposure", "min_cushion"):
        t[c] = (t[c] * 100).round(1)
    print(t.to_string(index=False))
for w in ("설계 2016-2025", "시험 2011-2015"):
    print(f"\n[{w}] 연도별 수익률 % (이자 2%, ETF 비용)")
    yc = [c for c in df.columns if c.startswith("y") and c[1:].isdigit()]
    t = main[main.window == w].set_index("agent")[yc].dropna(axis=1, how="all")
    print((t * 100).round(1).to_string())

design = main[(main.window == "설계 2016-2025") & main.agent.isin(list(AGENTS)[1:])]
pick = design.sort_values(["negative_years", "cagr"], ascending=[True, False]).iloc[0]["agent"]
get = lambda w, a: main[(main.window == w) & (main.agent == a)].iloc[0]
tp, t0, d0, dp = get("시험 2011-2015", pick), get("시험 2011-2015", "A0 고정 배수"), get("설계 2016-2025", "A0 고정 배수"), get("설계 2016-2025", pick)
checks = {"시험 손실 연도 0": bool(int(tp.negative_years) == 0), "시험 CAGR > A0": bool(tp.cagr > t0.cagr),
          "시험 CAGR > 현금 2%": bool(tp.cagr > get("시험 2011-2015", "현금만").cagr), "설계 CAGR > A0": bool(dp.cagr > d0.cagr)}
print(f"\n선택 (설계 구간 규칙): {pick}")
for k, v in checks.items():
    print(f"  {k}: {'만족' if v else '미달'}")
print("교체 기준:", "모두 만족 → B-09 최종 모델 교체" if all(checks.values()) else "미달 → 교체하지 않음")
ledger.result("safe|선택", pick=pick, checks=checks, replace=all(checks.values()))
print("done")
