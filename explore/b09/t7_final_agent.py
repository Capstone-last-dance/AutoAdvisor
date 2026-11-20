r"""T7 비중 최종 에이전트 (11/20 사용자 목표):
"MDD는 낮추고 CAGR·수익은 높이는 보상으로 RL 에이전트를 만든다. 리밸런싱 결정마다 투자 비중을 시장 상황에 따라 0~100%로 정한다.
수익이 나면 보상, 손실이 나면 벌점을 준다. 현금은 안정자산 금리로 수익을 준다 (주식 20%면 나머지 80%가 안정자산 수익)."

이 목표는 v8(`t7_agent8.py`, docs 30절)의 설계와 같다. 이 파일은 v8에서 실행 전에 정한 규칙(검증 2021~2023 점수)이 고른
λ 0.5, 시드 5개 앙상블을 최종 에이전트로 불러 쓴다. 학습은 사용자가 11/20에 `t7_agent8.py run`으로 했다 (runs/t7h/).
여기서는 다시 학습하지 않는다.

목표 → 구현
- 결정: 리밸런싱일(2주)마다 주식 비중 E ∈ {0, 0.1, …, 1.0}을 정한다. 최종 E는 시드 5개의 평균이다 (0~1, 0.02 단위).
- 시장 상황(관측): 지수 특성 14개(코스피 200 수익·변동성·이동평균 괴리·낙폭, 시장 폭) + 바구니 특성 6개 + 지금 비중.
- 보상 (결정마다) = 100 × ln(다음 결정까지 잔고 변화) − λ × 100 × (그 구간에 새로 깊어진 MDD), λ = 0.5.
  - 잔고가 늘면(수익) 보상이 +이고, 줄면(손실) −다. 손실로 MDD가 깊어지면 그만큼 더 뺀다.
  - 1년 에피소드의 보상 합 = 100 × (그해 로그수익 − 0.5 × 그해 MDD).
- 현금: 잔고 = 주식 E + 현금 (1 − E). 현금은 매일 CD 91일물 금리로 불어난다 (`cash_yield.py`, docs 29절).
  주식 20%면 나머지 80%가 CD 91일물 이자를 받는다. 학습·검증 장부와 평가(엔진 NAV 보정)가 모두 같다.

명령 (저장소 루트, PowerShell):
  .\.venv\Scripts\python.exe -I explore\b09\run.py t7_final_agent.py report             # 보상 예시, 2021~2025 결정, 성과
  .\.venv\Scripts\python.exe -I explore\b09\run.py t7_final_agent.py decide 2024-07-19   # 그 신호일에 정한 비중
- 탐색 결과다 (R-15). 결과: t7_final_decisions.csv.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd

import t7_agent8 as v8

LAM = 0.5  # docs 30절 v8: 검증 2021~2023 앙상블 점수 CAGR ÷ max(MDD, 2%)가 가장 높은 λ (실행 전에 정한 규칙)
OUT = v8.HERE / "t7_final_decisions.csv"
SHOW = ["r126", "ma200gap", "vol20", "b_dd", "b_vol20"]


def load():
    data = v8.Data()
    models, mu, sd = v8.load_best(LAM)
    return data, models, mu, sd


def decisions(data, models, mu, sd) -> pd.DataFrame:
    """2021년 첫 리밸런싱일(시작 비중 0.466107)부터 2025년 마지막 리밸런싱일까지 차례로 정한 E (평가와 같은 경로)."""
    es, _ = v8.path_from(data, models, mu, sd, v8.ACTIVE[0], v8.ACTIVE[-1])
    f = data.f
    rows = f[(f["year"] >= v8.ACTIVE[0]) & (f["year"] <= v8.ACTIVE[-1])].reset_index(drop=True)
    if len(rows) != len(es):
        raise RuntimeError("결정 수가 리밸런싱일 수와 다르다")
    out = rows[["date"] + SHOW].copy()
    out.insert(1, "e", np.asarray(es, dtype=float))
    return out


def reward_examples() -> pd.DataFrame:
    """학습 환경의 보상 그대로 잰다: 2주(거래일 10일, 달력일 14일), 현금 금리 연 3%, 주식이 매일 같은 폭으로 움직임,
    시작 비중 = 정한 비중 (거래 비용 없음)."""
    env_cls = v8.make_env_class()
    day = (1.03) ** (14 / 365 / 10) - 1.0
    rows = []
    for e, move in ((0.2, 0.05), (0.2, -0.05), (1.0, 0.05), (1.0, -0.05), (0.0, 0.05), (0.0, -0.05)):
        px = np.concatenate([np.linspace(1.0, 1.0 + move, 11), np.full(10, 1.0 + move)])
        rc = np.concatenate([[0.0], np.full(10, day), np.zeros(10)])
        dec = np.array([0, 10, 20])
        env = env_cls(px, np.zeros((len(dec), len(v8.FEATS))), dec, LAM, rc)
        env.p, env.t = 0, 0
        env.book = v8.Book(px, 0, e, rc=rc)
        a = int(np.argmin(np.abs(v8.LEVELS - e)))
        _, r, *_ = env.step(a)
        lv = 100.0 * math.log(env.book.nav)
        rows.append({"주식 비중": f"{e:.0%}", "주식 2주 수익": f"{move:+.0%}",
                     "주식 몫": f"{e * move:+.3%}", "현금 몫 (CD 연 3%)": f"{(1 - e) * ((1.03) ** (14 / 365) - 1):+.3%}",
                     "잔고 변화": f"{env.book.nav - 1:+.3%}", "수익 보상": round(lv, 3), "MDD 벌점": round(r - lv, 3), "보상": round(r, 3)})
    return pd.DataFrame(rows)


def check_path(d: pd.DataFrame) -> float:
    """이 파일이 다시 낸 결정이 v8 평가(C 엔진에 넣은 E)와 같은지."""
    es = json.loads((v8.EVAL / f"{v8.lam_id(LAM)}_ens.json").read_text(encoding="utf-8"))["es"]
    act = np.asarray(es, dtype=float)[-len(d):]
    diff = float(np.max(np.abs(act - d["e"].to_numpy())))
    if diff > 1e-12:
        raise RuntimeError(f"평가 때 E와 다르다 (최대 차이 {diff})")
    return diff


def report() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(f"[최종 에이전트] v8 λ {LAM:g}, 시드 {len(v8.SEEDS)}개 앙상블, 학습 2011~2020 (runs/t7h/)")
    print("\n1) 보상 예시 (학습 환경 그대로, λ 0.5): 수익이면 +, 손실이면 − 보상이고, MDD가 깊어지면 더 뺀다")
    print(reward_examples().to_string(index=False))

    data, models, mu, sd = load()
    d = decisions(data, models, mu, sd)
    diff = check_path(d)
    d.to_csv(OUT, index=False)
    print(f"\n2) 2021~2025 결정 {len(d)}번 (평가 때 C 엔진에 넣은 E와 같음, 최대 차이 {diff:.0e}) → {OUT.name}")
    bins = pd.cut(d["e"], [-0.001, 0.0, 0.3, 0.7, 0.999, 1.0], labels=["0% (전부 현금)", "1~30%", "31~70%", "71~99%", "100%"])
    print("  비중 분포: " + ", ".join(f"{k} {v}번" for k, v in bins.value_counts(sort=False).items()))
    print("  해마다 평균 비중: " + ", ".join(f"{y} {g['e'].mean():.2f}" for y, g in d.groupby(d["date"].dt.year)))
    q = d["r126"].quantile([1 / 3, 2 / 3]).to_numpy()
    print(f"  지수 6개월 수익이 하위 1/3인 날 평균 비중 {d.loc[d['r126'] <= q[0], 'e'].mean():.2f}, "
          f"상위 1/3인 날 {d.loc[d['r126'] >= q[1], 'e'].mean():.2f}")
    px = pd.Series(data.px, index=data.days)
    k = d["date"].map(lambda t: int(data.days.searchsorted(t)))
    nxt = list(k.iloc[1:]) + [len(px) - 1]
    d["바구니 다음 2주"] = [px.iloc[min(b + 1, len(px) - 1)] / px.iloc[a + 1] - 1 for a, b in zip(k, nxt)]
    show = d.loc[(d["e"] >= 0.999) | (d["e"] <= 0.0)].copy()
    show["date"] = show["date"].dt.date
    print("  비중 0% 또는 100%로 정한 날 (그 뒤 2주 바구니 수익):")
    print(show[["date", "e", "r126", "b_dd", "바구니 다음 2주"]].round(3).to_string(index=False))

    j = json.loads((v8.HERE / "t7h_judgment.json").read_text(encoding="utf-8"))
    P = v8.lam_id(LAM)
    rows = [("최종 에이전트 (λ 0.5 앙상블)", j["results"][f"{P}_ens"]), ("T7-80-C53 (비중 0.466 고정)", j["results"]["c53"]),
            ("같은 평균 비중 고정", j["results"][f"{P}_ens_fixed"]), ("코스피 종합 46.6% + CD 91일물", j["refs"]["kospi_mix"]),
            ("T7 바구니 100%", j["refs"]["e1"]), ("코스피 종합", j["refs"]["kospi"])]
    print("\n3) 성과 (CAGR·MDD, 현금 = CD 91일물, C 엔진). 검증 구간은 λ·체크포인트를 고른 구간이라 부풀어 있다")
    for name, r in rows:
        print(f"  {name:28s} 검증 2021~2023 {r['cagr_val']:+.1%}·{r['mdd_val']:.1%} | 테스트 2024~2025 {r['cagr_test']:+.1%}·{r['mdd_test']:.1%}"
              f" | 2021~2025 {r['cagr_all']:+.1%}·{r['mdd_all']:.1%}")
    print("  판정 (실행 전 고정): " + ", ".join(f"{k_.split(' (')[0]} {'만족' if v else '미달'}" for k_, v in j["judgment"].items()
                                       if isinstance(v, bool)))


def decide(day: str) -> None:
    data, models, mu, sd = load()
    d = decisions(data, models, mu, sd)
    t = pd.Timestamp(day)
    if t < d["date"].iloc[0] or t > d["date"].iloc[-1] + pd.Timedelta(days=14):
        raise SystemExit(f"{d['date'].iloc[0].date()} ~ {d['date'].iloc[-1].date()} 리밸런싱일만 낼 수 있다")
    r = d[d["date"] <= t].iloc[-1]
    print(f"{r['date'].date()} 리밸런싱일 결정: 주식 {r['e']:.0%}, 현금(CD 91일물) {1 - r['e']:.0%} "
          f"(지수 6개월 수익 {r['r126']:+.1%}, 200일선 괴리 {r['ma200gap']:+.1%}, 바구니 낙폭 {r['b_dd']:+.1%})")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "report":
        report()
    elif cmd == "decide" and len(sys.argv) > 2:
        decide(sys.argv[2])
    else:
        raise SystemExit("명령: report | decide YYYY-MM-DD")


if __name__ == "__main__":
    main()
