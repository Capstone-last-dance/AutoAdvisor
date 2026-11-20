"""낙폭 한도별 교환표 (11/19 사용자 결정): 고점 대비 낙폭 한도 X를 바꾸면 수익이 얼마나 생기나.

  python -I explore/b09/run.py safe_frontier.py

구조 (실행 전 고정, 11/19): safe_drawdown.py의 시뮬레이션을 그대로 쓰고 X만 바꾼다.
- 모델: A0 고정 배수(m = 4) · 고점 바닥(F = (1 − X) × 최고 잔고, 올라가기만) · 매주 점검.
  safe_drawdown에서 고른 모델이다. 배수 4는 X와 상관없이 고정한다 (X ≥ 25%면 고점에서 비중 1).
- X: 5% · 10% · 15% · 20% (제안한 값) + 25% · 30% (지수 최대 낙폭 41% 쪽으로 곡선을 잇는 참고).
- 현금 수익 0, ETF 비용 (수수료 0.015% + 슬리피지 0.10%, 거래세 0).
- 실행: safe_drawdown과 같은 17개 (전체 2011-2025, 5년 3개, 시작 연도별 3년 13개). 참고 행: 지수 그대로, 현금만.

판정: 고르는 규칙은 두지 않는다. 사용자가 감당할 손실 한도로 X를 고르는 표다.
- 확인 항목: 각 X에서 17개 실행 모두 바닥 아래 날이 0인가 (한도가 구조대로 지켜졌나).
- 수익 숫자는 지나간 구간(2011~2025, 모두 이미 본 구간)의 값이다. 고른 X도 탐색 결과이고, 운용은 PM 결정이 필요하다.
- 결과: safe_frontier.csv, safe_frontier.json, 원장.
"""
import json

from common import HERE, Ledger
import pandas as pd
from safe_drawdown import MAIN, RUNS, STARTS, simulate, window

XS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
MODEL = ("A0 고정 배수", "고점", "매주")
REFS = {"지수 그대로": ("지수 그대로", "-", "매주"), "현금만": ("현금만", "-", "매주")}


def main():
    ledger = Ledger(HERE / "ledger.jsonl")
    for x in XS:
        ledger.plan(f"safefr|X={x:.0%}", "안전 투자 모델 교환표", f"A0 배수 4 · 바닥 {1 - x:.2f} × 최고 잔고 · 매주, 현금 수익 0",
                    "사용자 결정(11/19): 낙폭 한도별 수익 교환표 (선택 규칙 없음, 사용자가 손실 한도로 고른다)")
    models = {**{k: (v, 0.05) for k, v in REFS.items()}, **{f"X={x:.0%}": (MODEL, x) for x in XS}}
    out, rows = {}, []
    for wname, (y0, y1) in RUNS.items():
        dates, end_day = window(y0, y1)
        for name, (spec, x) in models.items():
            res = simulate(*spec, dates, end_day, x=x)
            out[(wname, name)] = res
            rows.append({"window": wname, "model": name, "x": None if name in REFS else x,
                         **{k: v for k, v in res.items() if k != "yearly"}, **{f"y{y}": v for y, v in res["yearly"].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "safe_frontier.csv", index=False)
    (HERE / "safe_frontier.json").write_text(
        json.dumps({f"{w} | {m}": r for (w, m), r in out.items()}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    idx_cagr = out[("전체 2011-2025", "지수 그대로")]["cagr"]
    table = []
    for name in models:
        full = out[("전체 2011-2025", name)]
        three = [out[(w, name)] for w in STARTS]
        row = {
            "model": name, "cagr15": full["cagr"], "share_of_index": full["cagr"] / idx_cagr, "mdd15": full["mdd"],
            "final15": full["final_nav"], "min_nav15": full["min_nav"], "loss_years": full["negative_years"],
            "worst_year": full["worst_year"], "mean_exposure": full["mean_exposure"], "lock_share": full["lock_share"],
            **{f"cagr {w}": out[(w, name)]["cagr"] for w in MAIN if w != "전체 2011-2025"},
            "3y_above_principal": f"{sum(r['final_nav'] >= 1.0 for r in three)}/{len(three)}",
            "3y_worst_final": min(r["final_nav"] for r in three), "3y_max_mdd": max(r["mdd"] for r in three),
            "breach_days_all": sum(out[(w, name)]["breach_days"] for w in RUNS),
            "max_mdd_all": max(out[(w, name)]["mdd"] for w in RUNS),
        }
        table.append(row)
        if name not in REFS:
            ledger.result(f"safefr|{name}", **{k: v for k, v in row.items() if k != "model"})
    t = pd.DataFrame(table)
    pct = ["cagr15", "share_of_index", "mdd15", "worst_year", "mean_exposure", "lock_share", "3y_max_mdd", "max_mdd_all"]
    pct += [c for c in t.columns if c.startswith("cagr ")]
    for c in pct:
        t[c] = (t[c] * 100).round(2)
    for c in ("final15", "min_nav15", "3y_worst_final"):
        t[c] = t[c].round(3)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print("\n[교환표] A0 배수 4 · 고점 바닥 · 매주, 현금 수익 0 (% 단위: 수익률·낙폭·비중)")
    print(t.to_string(index=False))

    yc = [c for c in df.columns if c.startswith("y") and c[1:].isdigit()]
    print("\n[전체 2011-2025] 연도별 수익률 %")
    print((df[df.window == "전체 2011-2025"].set_index("model")[yc] * 100).round(1).to_string())

    print("\n한도 확인 (17개 실행 모두 바닥 아래 날 0):")
    for x in XS:
        name = f"X={x:.0%}"
        bd = sum(out[(w, name)]["breach_days"] for w in RUNS)
        print(f"  {name}: {'지켜짐' if bd == 0 else f'바닥 아래 {bd}일'} (17개 실행 중 최대 낙폭 "
              f"{max(out[(w, name)]['mdd'] for w in RUNS):.1%})")
    print("done")


if __name__ == "__main__":
    main()
