r"""실험 G3 (docs 34절, 사후 확인, 재학습 없음, 실행 전 고정 11/20): 실험 G 모델로 고르기 없는 15개 평균 앙상블 → C 엔진.

- 폴드마다 λ 3 × 시드 5 = 15개 PPO 정책의 비중을 평균한다. 체크포인트도 λ도 고르지 않는다.
- 주 판정 대상: v12 (G1) last.zip 15개 평균. 검증 구간을 전혀 쓰지 않는다.
  - 판정: G1 (2016~2025 CAGR > C53, MDD < C53), G2 (2016~2020·2021~2025 각각 같은 조건),
    타이밍 (같은 평균 고정보다 CAGR·MDD 모두 낫고, 시드별 λ 3개 평균 판 중 자기 고정보다 CAGR 높은 시드 4/5 이상).
- 참고 판 (판정 안 함): v12 best.zip 15개, v13 last·best 15개.
- 이유: 검증 점수 CAGR / max(MDD, 2%)가 11개 폴드 모두 E 0~0.2를 가장 높게 쳤다 (검증 자료만, docs 34절).

실행 (저장소 루트, PowerShell): .\.venv\Scripts\python.exe -I -X utf8 explore\b09\run.py t7_g3.py 12   (그다음 13)
- 탐색 결과다 (R-15). 결과: t7g3_v12.json·t7g3_v13.json, runs/t7l|t7m/eval_g3/, 원장 auto|G3.
"""
from __future__ import annotations

import importlib
import json
import sys
import time

import numpy as np
import pandas as pd

VER = sys.argv[1] if len(sys.argv) > 1 else "12"
m = importlib.import_module(f"t7_agent{VER}")  # 경로·특성·행동 단계를 그 판 것으로 바꾼다
import t7_agent8 as base  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

PRIMARY = VER == "12"
OUT = m.HERE / f"t7g3_v{VER}.json"
EVAL = m.RUNS / "eval_g3"


def load_fold(y: int, ck: str):
    """폴드 y의 15개 정책과 표준화 값. 표준화는 폴드 학습 자료로 정해져 λ·시드와 무관하다 (같은지 확인)."""
    models, st0 = {}, None
    for lam in m.LAMBDAS:
        for s in m.SEEDS:
            d = m.run_dir(lam, y, s)
            st = json.loads((d / "stats.json").read_text(encoding="utf-8"))
            if st["cols"] != m.FEATS:
                raise RuntimeError(f"폴드 {y} λ {lam:g} 시드 {s}: 특성 열이 다르다")
            if st0 is None:
                st0 = st
            elif not (np.allclose(st["mu"], st0["mu"]) and np.allclose(st["sd"], st0["sd"])):
                raise RuntimeError(f"폴드 {y}: λ·시드마다 표준화 값이 다르다")
            models[(lam, s)] = PPO.load(d / ck, device="cpu")
    return models, np.array(st0["mu"]), np.array(st0["sd"])


def main() -> None:
    t_all = time.time()
    led = m.Ledger(m.HERE / "ledger.jsonl")
    if not m.ledger_has("plan", "auto|G3"):  # 결과보다 먼저 적는다
        led.plan("auto|G3", "RL 스스로 비중 (docs 34절)",
                 "G3 (사후, 재학습 없음): 실험 G 모델, 폴드마다 λ 3 × 시드 5 = 15개 평균, 체크포인트·λ 고르기 없음. "
                 "주 판정 v12 last.zip, 참고 v12 best·v13 last·best",
                 "검증 점수 CAGR/max(MDD,2%)가 11개 폴드 모두 E 0~0.2를 가장 높게 침 (검증 자료만)")
    data = base.Data()
    print(f"G3 v{VER} ({'주 판정' if PRIMARY else '참고'}): 행동 {len(m.LEVELS)}단계, 특성 {len(m.FEATS)}개", flush=True)

    # C 엔진 준비 (t7_agent12.evaluate와 같다)
    dates = [d for d in m.signal_dates(m.io.read_table("universe"), m.kospi200(m.io.read_table("index")).index, m.FIRST_SIGNAL_DATE,
                                       m.LAST_SIGNAL_DATE) if 2016 <= int(d[:4]) <= 2025]
    ctx = m.Ctx(m.ALL_YEARS)
    if list(ctx.dates) != dates:
        raise RuntimeError("Ctx 신호일이 다르다")
    snaps = ctx.snaps(signals=m.smooth_signals(ctx.signals, 4), key="sm4")
    b = m.replace(ctx.buffer, enter_rank_max=m.N, exit_rank_greater_than=m.N + m.BAND, max_holdings=m.N)
    k200 = ctx.inputs.bm_close.copy()
    k200.index = m._ns(k200.index)
    cal = k200.index[k200.index <= m.END]
    fills = pd.DatetimeIndex([cal[cal.searchsorted(d, side="right")] for d in m._ns(dates)])
    yr_reb = data.lv["year"].to_numpy()
    if len(yr_reb) != (len(dates) + 1) // 2:
        raise RuntimeError("리밸런싱일 수가 신호일 수와 맞지 않다")
    EVAL.mkdir(parents=True, exist_ok=True)
    CY, results = m.CashYield(), {}

    def engine(name: str, es_reb: list[float]) -> dict:
        t = time.time()
        mm, nav_raw, cfrac = m.run_cash(ctx, snaps, b, m.full_es(es_reb, len(dates)))
        nav_raw = nav_raw.copy()
        nav_raw.index = m._ns(nav_raw.index)
        nav = CY.adjust(nav_raw, cfrac)
        pd.DataFrame({"date": nav.index, "nav": nav.to_numpy()}).to_parquet(EVAL / f"nav_{name}.parquet", index=False)
        es_a = np.array(es_reb)
        r = {"name": name, "mean_e": float(es_a.mean()), "turnover10": float(mm["annual_turnover"]),
             "switches": int((np.abs(np.diff(es_a)) > 1e-9).sum())}
        for p, (y0, y1) in m.PERIODS.items():
            r[f"cagr_{p}"], r[f"mdd_{p}"] = m.period_stats(nav, y0, y1, fills)
        yl = nav.groupby(nav.index.year).last()
        r["yearly"] = {int(y): float(v) for y, v in (yl / yl.shift(1) - 1).items() if y >= 2017}
        r["yearly"][2016] = float(nav[nav.index.year == 2016].iloc[-1] / nav.iloc[0] - 1)
        r["mean_e_by_year"] = {int(y): float(es_a[yr_reb == y].mean()) for y in m.TEST_YEARS}
        results[name] = r
        print(f"  {name}: 2016~2025 {r['cagr_all']:+.2%}·{r['mdd_all']:.1%}, 2016~2020 {r['cagr_y16_20']:+.2%}·{r['mdd_y16_20']:.1%}, "
              f"2021~2025 {r['cagr_y21_25']:+.2%}·{r['mdd_y21_25']:.1%}, 평균 E {r['mean_e']:.2f} ({time.time() - t:.0f}초)", flush=True)
        return r

    c53 = engine("c53", [m.E_C53] * len(yr_reb))
    ref = json.loads((m.RUNS.parent.parent / f"{'t7l' if VER == '12' else 't7m'}_judgment.json").read_text(encoding="utf-8"))
    d_ = abs(c53["cagr_all"] - ref["results"]["c53"]["cagr_all"])
    print(f"  대조: C53 2016~2025 CAGR {c53['cagr_all']:+.6%} vs 실험 G 평가 {ref['results']['c53']['cagr_all']:+.6%} (차이 {d_:.1e})", flush=True)
    if d_ > 1e-12:
        raise RuntimeError("C53 값이 실험 G 평가와 다르다")

    for ck in ("last.zip", "best.zip"):
        t0 = time.time()
        pols = {y: load_fold(y, ck) for y in m.TEST_YEARS}
        tag = ck[:-4]
        es = m.wf_path(data, {y: (list(ms.values()), mu, sd) for y, (ms, mu, sd) in pols.items()})
        print(f" {tag}: 모델 읽기·경로 {time.time() - t0:.0f}초", flush=True)
        engine(f"{tag}_all15", es)
        engine(f"{tag}_all15_fixed", [float(np.mean(es))] * len(es))
        if PRIMARY and ck == "last.zip":
            for s in m.SEEDS:
                es_s = m.wf_path(data, {y: ([ms[(lam, s)] for lam in m.LAMBDAS], mu, sd) for y, (ms, mu, sd) in pols.items()})
                engine(f"{tag}_s{s}", es_s)
                engine(f"{tag}_s{s}_fixed", [float(np.mean(es_s))] * len(es_s))

    def beats(r, q, periods):
        return all(r[f"cagr_{p}"] > q[f"cagr_{p}"] and r[f"mdd_{p}"] < q[f"mdd_{p}"] for p in periods)

    judg = {}
    for tag in ("last", "best"):
        r, fx = results[f"{tag}_all15"], results[f"{tag}_all15_fixed"]
        judg[tag] = {"G1": beats(r, c53, ("all",)), "G2": beats(r, c53, ("y16_20", "y21_25")), "vs_fixed": beats(r, fx, ("all",))}
    if PRIMARY:
        sb = int(sum(results[f"last_s{s}"]["cagr_all"] > results[f"last_s{s}_fixed"]["cagr_all"] for s in m.SEEDS))
        judg["last"]["seeds_beat_fixed"] = sb
        judg["last"]["timing"] = bool(judg["last"]["vs_fixed"] and sb >= 4)

    fin = {"fold": m.FINAL_FOLD, "rule": "λ 3 × 시드 5 = 15개 last.zip 평균 (고르기 없음)",
           "models": [str(m.run_dir(lam, m.FINAL_FOLD, s) / "last.zip") for lam in m.LAMBDAS for s in m.SEEDS],
           "note": "2026 홀드아웃 테스트는 12/11 개봉 뒤 PM 승인 기록이 있을 때만 (명세 4.7절, PM-14)"}
    (m.RUNS / "final_2026_g3.json").write_text(json.dumps(fin, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT.write_text(json.dumps({"version": VER, "primary": PRIMARY, "judgment": judg, "results": results, "final_2026": fin},
                              ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    fmt = lambda r: f"{r['cagr_all']:+.1%}/{r['mdd_all']:.1%}"  # noqa: E731
    led.result(f"auto|G3|v{VER}", **{f"{tag}_all15": fmt(results[f"{tag}_all15"]) for tag in ("last", "best")},
               **{f"{tag}_y16_20": f"{results[f'{tag}_all15']['cagr_y16_20']:+.1%}/{results[f'{tag}_all15']['mdd_y16_20']:.1%}" for tag in ("last", "best")},
               **{f"{tag}_y21_25": f"{results[f'{tag}_all15']['cagr_y21_25']:+.1%}/{results[f'{tag}_all15']['mdd_y21_25']:.1%}" for tag in ("last", "best")},
               c53=fmt(c53), judgment=json.dumps(judg, ensure_ascii=False))

    print("\n[연도별 수익률 %·평균 E]")
    for n_, r in results.items():
        if n_.endswith("_fixed") and n_ != "last_all15_fixed":
            continue
        print(f"  {n_:18s} " + " ".join(f"{y % 100:02d}:{r['yearly'][y] * 100:+5.1f}" for y in sorted(r["yearly"]))
              + " | E " + " ".join(f"{r['mean_e_by_year'][y]:.2f}" for y in m.TEST_YEARS))
    print(f"\n[판정] {'주 판정 (v12 last.zip 15개)' if PRIMARY else '참고 판 (판정 안 함)'}: {json.dumps(judg, ensure_ascii=False)}")
    print(f"done ({(time.time() - t_all) / 60:.1f}분)")


if __name__ == "__main__":
    main()
