"""지수 + 위험 규칙 (11/19, 사용자 목표: 원금 대비 해마다 늘어나는 모델). 종목을 고르지 않는다.

  python -I explore/b09/run.py index_overlay.py

- 수익원: 코스피 200 (ETF로 보유한다고 보고 지수 종가로 계산, 배당 제외). 나머지는 현금 (부록 1 cash_return = 0).
- 노출 E_t는 신호일 t 종가까지의 정보로 정하고 다음 거래일 종가에 맞춘다 (C 엔진과 같은 체결 시점).
- 비용: 부록 1의 수수료 0.015% + 슬리피지 0.10% (매매 금액 기준, 한쪽). ETF는 증권거래세가 없어 sell_tax는 0으로 두고,
  주식처럼 sell_tax 0.20%를 물린 판을 '비용 가정 보수적'으로 같이 적는다. C 엔진을 거치지 않는 계산이라 공식 조건과 다르다.
- 규칙 (모두 이 실행 전에 정의가 있던 것. 새로 맞춘 파라미터 없음)
  · 지수 그대로: E = 1
  · 사전등록 M8: E_t = clip(0.157263 / σ_KOSPI20d, 0.3, 1), 모멘텀 붕괴 국면이면 ≤ 0.7 (부록 1 값 그대로)
  · M-tr-b: 200일선 아래면 min(E_t, 0.3), 위면 사전등록 E_t (G7 트랙)
  · M-tr-c: 200일선 아래면 0.3, 위면 1 (국면이면 0.7) (G7 트랙)
  · M-dd-0.15-f0.3: 사전등록 E_t, 자기 NAV 낙폭 > 15%면 × 0.3 (G7 확인 변형)
  · M-dd+tr: M-tr-b 다음 낙폭 > 15%면 × 0.5 (G7 확인 변형 M-dd+tr+tag에서 종목 차단만 뺀 것)
  · M-tr-0: 200일선 위면 1, 아래면 0 (교과서식 추세 규칙. 부록 1의 exposure_min 0.3 아래로 내려가므로 따로 표시)
- 연 단위 원금 보존 (CPPI, 포트폴리오 보험, 11/19 사용자 목표 '손실 해는 원금 보존, 흑자 해는 최대화')
  · 해마다 바닥 F = f × 연초 NAV(전년 마지막 거래일 종가 기준, 첫해는 1). E_t = clip(m × (NAV_t − F) / NAV_t, 0, 1).
  · 주 설정은 실행 전에 정한다: f = 0.98(한 해 최대 손실 약 2%), m = 4 (교과서에서 흔히 쓰는 3~5의 가운데).
  · 민감도: f ∈ {0.97, 0.98, 0.99} × m ∈ {3, 4, 5}. 고르는 데 쓰지 않고 주 설정이 얼마나 흔들리는지만 본다.
  · CPPI-0.98-4 + M8: 위 E_t와 사전등록 M8 E_t 중 작은 값.
  · 한 주 안에 지수가 1/m 넘게 빠지면 바닥을 뚫을 수 있다 (주 1회 조정이라). 바닥을 뚫은 해를 따로 센다.
- 11/19 2차 (1차 CPPI 결과를 본 뒤 더함, 따로 표시): 나쁜 해 허용 손실을 키운 바닥 0.95·0.90, 그리고 현금 이자 가정.
  · 현금 이자: 부록 1은 0이다. '이자 연 2% 가정'(2016~2025 한국 기준금리 평균 약 1.8%에 가까운 단순 가정, 자료 아님)을 따로 잰다.
  · 이자가 있으면 바닥을 연말 기준으로 할인한다: F_t = f × 연초 NAV / (1 + r)^(연말까지 남은 기간). f = 1.0이면 '원금 100% 보존형'.
- 구간: 2016~2020, 2021~2025, 10년 (구간마다 NAV 1에서 새로 시작). 결과: index_overlay.json, index_overlay.csv, 원장.
"""
import json

from common import ALL_YEARS, HERE, Ctx, Ledger
import numpy as np
import pandas as pd
from src.backtest import metrics as m5

ctx = Ctx(ALL_YEARS)
P = ctx.inputs.bm_close.sort_index()
DAYS = P.index
MA = P.rolling(200, min_periods=200).mean()
SNAP = {pd.Timestamp(s.date): s for s in ctx.snaps()}
cost_cfg = ctx.cfg["costs"]
RATE = float(cost_cfg["commission_per_side"]) + float(cost_cfg["slippage_per_side"])
TAX = float(cost_cfg["sell_tax"])
CAP = float(ctx.risk.momentum_crash_regime.exposure_cap)


def below(t):
    return bool(P[t] < MA[t])


def cppi(f, m):
    return lambda s, t, st: m * (st["nav"] - f * st["y0"] / (1.0 + st["r"]) ** st["tau"]) / st["nav"]


RULES = {  # 인자: 신호일 스냅샷 s, 신호일 t, 상태 st (nav, dd = 고점 대비 낙폭, y0 = 연초 NAV)
    "지수 그대로": lambda s, t, st: 1.0,
    "사전등록 M8": lambda s, t, st: float(s.exposure),
    "M-tr-b": lambda s, t, st: min(float(s.exposure), 0.3) if below(t) else float(s.exposure),
    "M-tr-c": lambda s, t, st: 0.3 if below(t) else (1.0 if s.regime_mc == 0 else min(1.0, CAP)),
    "M-dd-0.15-f0.3": lambda s, t, st: float(s.exposure) * (0.3 if st["dd"] > 0.15 else 1.0),
    "M-dd+tr": lambda s, t, st: (min(float(s.exposure), 0.3) if below(t) else float(s.exposure)) * (0.5 if st["dd"] > 0.15 else 1.0),
    "M-tr-0": lambda s, t, st: 0.0 if below(t) else 1.0,
    **{f"CPPI-{f}-{m}": cppi(f, m) for f in (0.97, 0.98, 0.99) for m in (3, 4, 5)},
    "CPPI-0.98-4 + M8": lambda s, t, st: min(cppi(0.98, 4)(s, t, st), float(s.exposure)),
    **{f"CPPI-{f}-{m}": cppi(f, m) for f in (1.0, 0.95, 0.9) for m in (3, 4, 5)},  # 2차
}
MAIN = "CPPI-0.98-4"
FLOOR = {k: float(k.split("-")[1].split(" ")[0]) for k in RULES if k.startswith("CPPI")}
RATES = (0.0, 0.02)


def simulate(rule, dates, end_day, tax, r=0.0):
    dates = [pd.Timestamp(d) for d in dates]
    idx = DAYS.get_indexer(dates)
    fills = {DAYS[i + 1]: d for i, d in zip(idx, dates) if i + 1 < len(DAYS) and DAYS[i + 1] <= end_day}
    sig = set(dates)
    units, cash, peak = 0.0, 1.0, 1.0
    target, nav_d, exps, turns = None, {}, [], []
    y0, year, prev_nav = 1.0, dates[0].year, 1.0
    for day in DAYS[(DAYS >= dates[0]) & (DAYS <= end_day)]:
        if day.year != year:  # 연초 NAV = 전년 마지막 거래일 종가 NAV
            y0, year = prev_nav, day.year
        cash *= (1.0 + r) ** (1.0 / 252.0)  # 현금 이자 (부록 1은 0)
        nav = units * P[day] + cash
        if day in fills and target is not None:
            w = units * P[day] / nav if nav > 0 else 0.0
            trade = (target - w) * nav
            cost = abs(trade) * RATE + max(0.0, -trade) * tax
            nav -= cost
            units = target * nav / P[day]
            cash = nav - units * P[day]
            turns.append(abs(target - w))
        peak = max(peak, nav)
        nav_d[day] = nav
        prev_nav = nav
        if day in sig:  # 신호일 종가까지의 정보로 다음 체결 목표를 정한다
            st = {"nav": nav, "dd": 1.0 - nav / peak, "y0": y0, "r": r,
                  "tau": max(0.0, (pd.Timestamp(f"{day.year}-12-31") - day).days / 365.0)}
            target = float(np.clip(RULES[rule](SNAP[day], day, st), 0.0, 1.0))
            exps.append(target)
    nav = pd.Series(nav_d)
    first_fill = min(fills)
    rep = nav[nav.index >= first_fill]
    n = len(rep) - 1
    wk = nav.reindex(dates).dropna()
    r_net = wk.pct_change().dropna().to_numpy()
    r_bm = P.reindex(wk.index).pct_change().dropna().to_numpy()
    last = rep.groupby(rep.index.year).last()
    yearly = (last / last.shift(1).fillna(1.0) - 1.0)
    years = n / 252.0
    return {
        "cagr": m5.cagr(np.r_[1.0, rep.to_numpy()[1:]], trading_days=n), "mdd": m5.max_drawdown(rep.to_numpy()),
        "ir": m5.information_ratio(r_net - r_bm), "mean_exposure": float(np.mean(exps)),
        "annual_turnover": float(np.sum(turns) / years), "min_nav": float(rep.min()),
        "negative_years": int((yearly < 0).sum()), "worst_year": float(yearly.min()),
        "floor_breach_years": int((yearly < FLOOR[rule] - 1.0 - 1e-12).sum()) if rule in FLOOR else None,
        "yearly": {int(y): float(v) for y, v in yearly.items()},
    }


WINDOWS = {"2016-2020": (2016, 2020), "2021-2025": (2021, 2025), "2016-2025": (2016, 2025)}
ledger = Ledger(HERE / "ledger.jsonl")
_prev = HERE / "index_overlay.csv"
DONE = set()  # 이미 원장에 결과가 있는 (규칙, 이자) 조합은 다시 적지 않는다
if _prev.exists():
    _old = pd.read_csv(_prev)
    _rate = _old["cash_rate"] if "cash_rate" in _old else pd.Series(0.0, index=_old.index)
    DONE = set(zip(_old["rule"], _rate.astype(float)))


def lid(rule, r):
    return f"idx|{rule}" if r == 0.0 else f"idx|{rule}|이자{r:.0%}"


for rule in RULES:
    for r in RATES:
        if (rule, r) not in DONE:
            ledger.plan(lid(rule, r), "원금 증가", f"코스피 200 + 노출 규칙 {rule}, 현금 이자 {r:.0%} (종목 선택 없음)",
                        "사용자 목표: 원금 대비 해마다 늘어나는 모델 (2021~2025를 본 뒤 실험)")
out, rows = {}, []
for wname, (y0, y1) in WINDOWS.items():
    dates = [d for d in ctx.dates if y0 <= int(d[:4]) <= y1]
    end_day = DAYS[DAYS <= pd.Timestamp(f"{y1}-12-31")][-1]
    res = {}
    for rule in RULES:
        for rate in RATES:
            for tax_name, tax in (("ETF 비용", 0.0), ("주식 비용", TAX)):
                r = simulate(rule, dates, end_day, tax, rate)
                res[f"{rule} | {tax_name} | 이자 {rate:.0%}"] = r
                rows.append({"window": wname, "rule": rule, "cost": tax_name, "cash_rate": rate,
                             **{k: v for k, v in r.items() if k != "yearly"}, **{f"y{y}": v for y, v in r["yearly"].items()}})
                if tax_name == "ETF 비용" and (rule, rate) not in DONE:
                    ledger.result(lid(rule, rate), years=wname, **{k: v for k, v in r.items() if k != "yearly"})
    out[wname] = res
pd.DataFrame(rows).to_csv(HERE / "index_overlay.csv", index=False)
(HERE / "index_overlay.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
for rate in RATES:
    print(f"10년 연도별 수익률 % (ETF 비용, 현금 이자 {rate:.0%})")
    show = ["지수 그대로", "M-tr-0", "M-dd+tr", MAIN, "CPPI-1.0-4", "CPPI-0.95-4", "CPPI-0.9-4"]
    yr = pd.DataFrame({rule: out["2016-2025"][f"{rule} | ETF 비용 | 이자 {rate:.0%}"]["yearly"] for rule in show})
    print((yr * 100).round(1).to_string())
    print(f"10년 요약 (ETF 비용, 현금 이자 {rate:.0%}): 손실 연도 수, 최악 연도, 바닥 뚫은 해, CAGR, MDD, 평균 노출")
    for rule in RULES:
        r = out["2016-2025"][f"{rule} | ETF 비용 | 이자 {rate:.0%}"]
        print(f"  {rule:20s} 손실 {r['negative_years']}해, 최악 {r['worst_year']*100:+.1f}%, 바닥 뚫음 {r['floor_breach_years']}, "
              f"CAGR {r['cagr']*100:+.2f}%, MDD {r['mdd']*100:.1f}%, 노출 {r['mean_exposure']*100:.0f}%")
print("done")
