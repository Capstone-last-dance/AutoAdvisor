"""안전 투자 모델 (11/19 사용자 결정): 고점 대비 낙폭 한도 5%, 현금 수익 0, 매주 점검 vs 매일 점검.

  python -I explore/b09/run.py safe_drawdown.py

목표: 폭락장에서 살아남는 것. IR·지수 초과 수익은 따지지 않는다.

구조 (실행 전 고정, 11/19)
- 자산: 코스피 200(지수 종가, 배당 제외, ETF로 가정)과 현금(수익 0, 부록 1).
- 바닥 F_t = 0.95 × 지금까지의 최고 잔고 (일간 종가 NAV, 시작 1 포함). 바닥은 올라가기만 한다.
  바닥을 지키면 고점 대비 낙폭이 5% 이내이고, 최고 잔고 ≥ 1이므로 원금 손실도 5% 이내다.
- 목표 비중 E_t = clip(m_t × (NAV_t − F_t) / NAV_t, 0, 1). 에이전트는 배수 m_t만 정한다.
  A0 고정 4 / A1 200일선 위 8, 아래 2 / A2 clip(√52 / (4σ_t), 2, 10) / A3 200일선 위 A2, 아래 2 (safe_principal과 같다)
- 체결: 그날 종가까지의 정보로 정하고 다음 거래일 종가에 맞춘다.
- 점검 방식
  - 매주: 신호일(매주)에만 목표 비중을 정한다.
  - 매일 점검: 매주 리밸런싱은 같고, 신호일이 아닌 날에도 종가에서 현재 비중이 목표보다 2%p 넘게 크면
    다음 거래일 종가에 목표까지 줄인다 (줄이기만 하고 늘리지는 않는다).
- 비용: 수수료 0.015% + 슬리피지 0.10% (매매 금액, 한쪽). ETF라 거래세 0.
- 참고 행: 지수 그대로, 현금만, A0 원금 고정 바닥(F = 0.95 × 처음 돈, safe_principal과 같은 방식).

판정 (실행 전 고정, 11/19)
- 실행: 전체 2011-2025(한 번에), 2011-2015 / 2016-2020 / 2021-2025(각각 1에서 시작),
  시작 연도별 3년(2011~2023년 첫 신호일에 시작, 13개).
- 합격: 위 모든 실행에서 바닥 아래로 내려간 날이 0 (일간 기준 고점 대비 낙폭 ≤ 5%).
- 선택: 합격한 A0~A3 × 점검 방식 중 전체 2011-2025 CAGR이 가장 높은 것 (같으면 회전율이 낮은 쪽).
- 보고: 고른 모델이 각 실행에서 원금 이상으로 끝났는지, 비중 2% 미만으로 묶인 날 비율.
- 탐색 결과다 (R-15). 운용 모델로 쓰려면 PM 결정이 필요하다.
- 결과: safe_drawdown.csv, safe_drawdown.json, 원장.
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

X = 0.05  # 낙폭 한도
GUARD_BAND = 0.02  # 매일 점검: 현재 비중 − 목표 비중이 이보다 크면 줄인다
LOCK = 0.02  # 비중이 이보다 작으면 '묶인 날'로 센다
cfg = load_prereg(PREREG_PATH)
assert float(cfg["costs"]["cash_return"]) == 0.0, "부록 1 cash_return이 0이 아니다"
risk = RiskLayerParams.from_prereg(cfg)
P = kospi200(io.read_table("index")).sort_index()
DAYS = P.index
PX = dict(zip(DAYS, P.to_numpy(dtype=float)))
MA = dict(zip(DAYS, P.rolling(200, min_periods=200).mean().to_numpy(dtype=float)))
SIG = market_state_table(P, risk)["sigma"].reindex(DAYS)
SIG = dict(zip(DAYS, SIG.to_numpy(dtype=float)))
DATES = [pd.Timestamp(d) for d in signal_dates(io.read_table("universe"), DAYS, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)]
RATE = float(cfg["costs"]["commission_per_side"]) + float(cfg["costs"]["slippage_per_side"])


def m_vol(t):
    s = SIG.get(t, np.nan)
    return 4.0 if not np.isfinite(s) or s <= 0 else float(np.clip(math.sqrt(52) / (4.0 * s), 2.0, 10.0))


def up(t):
    ma = MA.get(t, np.nan)
    return None if not np.isfinite(ma) else bool(PX[t] >= ma)


AGENTS = {
    "A0 고정 배수": lambda t: 4.0,
    "A1 추세 배수": lambda t: 4.0 if up(t) is None else (8.0 if up(t) else 2.0),
    "A2 변동성 배수": m_vol,
    "A3 추세 + 변동성": lambda t: 4.0 if up(t) is None else (m_vol(t) if up(t) else 2.0),
}
MODES = ("매주", "매일 점검")
CANDS = [(a, "고점", md) for a in AGENTS for md in MODES]
REFS = [("지수 그대로", "-", "매주"), ("현금만", "-", "매주"), ("A0 고정 배수", "원금", "매주")]


def label(agent, floor_type, mode):
    if floor_type == "-":
        return agent
    return f"{agent} · {'고점 바닥' if floor_type == '고점' else '원금 고정 바닥'} · {mode}"


def simulate(agent, floor_type, mode, dates, end_day, x=X, series=False):
    """x: 낙폭 한도 (기본 X = 5%). safe_frontier.py가 다른 값으로 부른다."""
    days = DAYS[(DAYS >= dates[0]) & (DAYS <= end_day)]
    nxt = dict(zip(days[:-1], days[1:]))
    sig = set(dates)
    units, cash, peak = 0.0, 1.0, 1.0
    pending = {}  # 체결일 → 목표 비중
    nav_d, w_d, f_d, turns, n_guard = {}, {}, {}, [], 0
    for day in days:
        px = PX[day]
        nav = units * px + cash  # 현금 수익 0
        if day in pending:
            target = pending.pop(day)
            w = units * px / nav
            trade = (target - w) * nav
            nav -= abs(trade) * RATE
            units = target * nav / px
            cash = nav - units * px
            turns.append(abs(target - w))
        peak = max(peak, nav)
        floor = (1.0 - x) * (peak if floor_type == "고점" else 1.0)
        w = units * px / nav
        nav_d[day], w_d[day], f_d[day] = nav, w, floor
        if day not in nxt:
            continue
        if agent == "지수 그대로":
            tgt = 1.0
        elif agent == "현금만":
            tgt = 0.0
        else:
            tgt = float(np.clip(AGENTS[agent](day) * (nav - floor) / nav, 0.0, 1.0))
        if day in sig:
            pending[nxt[day]] = tgt
        elif mode == "매일 점검" and agent in AGENTS and w - tgt > GUARD_BAND:
            pending[nxt[day]] = tgt
            n_guard += 1
    nav, wser, fser = pd.Series(nav_d), pd.Series(w_d), pd.Series(f_d)
    first_fill = nxt[days[0]]
    rep = nav[nav.index >= first_fill]
    n = len(rep) - 1
    last = rep.groupby(rep.index.year).last()
    yearly = last / last.shift(1).fillna(1.0) - 1.0
    gap = (fser - nav) / fser
    res = {
        "cagr": m5.cagr(np.r_[1.0, rep.to_numpy()[1:]], trading_days=n),
        "mdd": m5.max_drawdown(np.r_[1.0, nav.to_numpy()]),
        "final_nav": float(nav.iloc[-1]), "min_nav": float(nav.min()),
        "breach_days": int((gap > 1e-12).sum()),  # 일간 NAV가 바닥 아래였던 날 수
        "worst_breach": float(max(gap.max(), 0.0)),  # 바닥 아래로 가장 깊이 내려간 정도 (바닥 대비)
        "below_principal_days": int((nav < 1.0 - 1e-12).sum()),
        "negative_years": int((yearly < -1e-12).sum()), "worst_year": float(yearly.min()),
        "mean_exposure": float(wser[rep.index].mean()),
        "lock_share": float((wser[rep.index] < LOCK).mean()),  # 비중 2% 미만이었던 날 비율
        "annual_turnover": float(np.sum(turns) / (n / 252.0)), "guard_trades": n_guard,
        "yearly": {int(y): float(v) for y, v in yearly.items()},
    }
    if series:  # safe_phases.py: 일별 잔고·주식 비중
        res["_nav"], res["_w"] = nav, wser
    return res


def window(y0, y1):
    dates = [d for d in DATES if y0 <= d.year <= y1]
    return dates, DAYS[DAYS <= pd.Timestamp(f"{y1}-12-31")][-1]


MAIN = {"전체 2011-2025": (2011, 2025), "2011-2015": (2011, 2015), "2016-2020": (2016, 2020), "2021-2025": (2021, 2025)}
STARTS = {f"3년 {y}-{y + 2}": (y, y + 2) for y in range(2011, 2024)}
RUNS = {**MAIN, **STARTS}

def main():
    ledger = Ledger(HERE / "ledger.jsonl")
    for c in CANDS:
        ledger.plan(f"safedd|{label(*c)}", "안전 투자 모델", "코스피 200 + 바닥 0.95 × 최고 잔고(올라가기만) + 배수 규칙, 현금 수익 0",
                    "사용자 결정(11/19): 고점 대비 낙폭 한도 5%, 매주 vs 매일 점검 비교, IR 기준 없음")
    out, rows = {}, []
    for wname, (y0, y1) in RUNS.items():
        dates, end_day = window(y0, y1)
        for c in REFS + CANDS:
            res = simulate(*c, dates, end_day)
            out[(wname, label(*c))] = res
            rows.append({"window": wname, "model": label(*c), **{k: v for k, v in res.items() if k != "yearly"},
                         **{f"y{y}": v for y, v in res["yearly"].items()}})
            if c in CANDS and wname in MAIN:
                ledger.result(f"safedd|{label(*c)}", years=wname, **{k: v for k, v in res.items() if k != "yearly"})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "safe_drawdown.csv", index=False)
    (HERE / "safe_drawdown.json").write_text(
        json.dumps({f"{w} | {m}": r for (w, m), r in out.items()}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["model", "cagr", "mdd", "final_nav", "min_nav", "breach_days", "worst_breach", "negative_years", "worst_year",
            "mean_exposure", "lock_share", "guard_trades"]
    for w in MAIN:
        print(f"\n[{w}] (바닥 = 최고 잔고의 95%, 현금 수익 0, ETF 비용)")
        t = df[df.window == w][cols].copy()
        for c in ("cagr", "mdd", "worst_breach", "worst_year", "mean_exposure", "lock_share"):
            t[c] = (t[c] * 100).round(1)
        for c in ("final_nav", "min_nav"):
            t[c] = t[c].round(3)
        print(t.to_string(index=False))

    print("\n[시작 연도별 3년] 끝 잔고, 그리고 13개 실행의 바닥 아래 날 수 합")
    st = df[df.window.isin(list(STARTS))]
    print(st.pivot(index="model", columns="window", values="final_nav").round(3).reindex(
        [label(*c) for c in REFS + CANDS])[list(STARTS)].to_string())
    print(st.groupby("model")[["breach_days"]].sum().reindex([label(*c) for c in REFS + CANDS]).to_string())

    yc = [c for c in df.columns if c.startswith("y") and c[1:].isdigit()]
    print("\n[전체 2011-2025] 연도별 수익률 %")
    print((df[df.window == "전체 2011-2025"].set_index("model")[yc] * 100).round(1).to_string())

    names = [label(*c) for c in CANDS]
    passed = {m: all(out[(w, m)]["breach_days"] == 0 for w in RUNS) for m in names}
    print("\n합격 (모든 실행에서 바닥 아래 날 0):")
    for m in names:
        print(f"  {m}: {'합격' if passed[m] else '불합격'}")
    elig = [m for m in names if passed[m]]
    if not elig:
        print("선택: 합격한 모델이 없다")
        ledger.result("safedd|선택", pick=None, passed=passed)
    else:
        full = lambda m: out[("전체 2011-2025", m)]
        pick = max(elig, key=lambda m: (round(full(m)["cagr"], 10), -full(m)["annual_turnover"]))
        above = {w: bool(out[(w, pick)]["final_nav"] >= 1.0) for w in RUNS}
        print(f"선택: {pick} (전체 CAGR {full(pick)['cagr']:.2%}, 최대 낙폭 {full(pick)['mdd']:.2%}, "
              f"끝 잔고 {full(pick)['final_nav']:.3f}, 묶인 날 {full(pick)['lock_share']:.0%})")
        print(f"  원금 이상으로 끝난 실행: {sum(above.values())}/{len(above)}")
        for w, v in above.items():
            if not v:
                print(f"    원금 아래로 끝남: {w} ({out[(w, pick)]['final_nav']:.3f})")
        ledger.result("safedd|선택", pick=pick, passed=passed, above_principal=above)
    print("done")


if __name__ == "__main__":
    main()
