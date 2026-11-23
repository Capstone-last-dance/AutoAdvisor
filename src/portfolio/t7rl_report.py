r"""T7-80-RL 고정 기록의 2016~2025 재현 (docs/b09_explore.md 35절). 탐색 결과다 (R-15).

실행 (저장소 루트, 5분 안팎): .\.venv\Scripts\python.exe -m src.portfolio.t7rl_report
- 고정 파일(reports/20261120_B_t7rl)의 SHA-256을 manifest.json으로 확인한다.
- 탐색에서 잰 값을 main 코드(src/portfolio/t7rl.py)로 다시 만들고, 차이가 0이어야 한다 (일별 NAV는 상대 1e-12).
  (1) T7 보유 80 E = 1 실행 일별 NAV = reference/e1_nav.csv
  (2) T7-80-C53 일별 NAV = reference/c53_nav.csv (26절 고정 모델)
  (3) 고정 바구니 가격의 2016-01-11 뒤 = 상수 × (1)
  (4) 결정일 특성 10개 = reference/wf_features.csv
  (5) walk-forward E 경로 260개 (해마다 그 폴드 정책) = reference/wf_decisions.csv
  (6) T7-80-RL 일별 NAV (현금 수익 0) = reference/wf_nav.csv의 nav_raw
  (7) 폴드 2026 정책의 검증 구간(2024~2025) E 경로 = reference/val2026_decisions.csv
- 입력은 탐색과 같다: A v7 시험 신호 20261117_A_wfnews, PM M9 경보 20261112_PM_m9_b07_v1, C 엔진, 판정 기간 끝 처리.
- 결과(같은 폴더): metrics.json (지표·대조·지문), decisions_wf.csv, daily_nav.csv, git_commit.txt.
  지표는 공식 현금 수익 0과 탐색 기준 CD 91일물(엔진 일별 현금 비중으로 보정) 두 가지다.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.common import io
from src.portfolio import t7rl
from src.portfolio.buffer_wf_report import period_end, replay, summary
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.env_check_report import (FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE, TABLES, feature_table, kospi200,
                                            members_by_date, signal_dates)
from src.portfolio.params import PREREG_PATH, BufferRuleParams, RiskLayerParams, load_prereg
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_fold_report import ENGINE_TABLES, m9_schedule

SIG_RUN = "20261117_A_wfnews"
M9_RUN = "20261112_PM_m9_b07_v1"
YEARS = tuple(range(2016, 2026))
PERIODS = {"all": (2016, 2025), "y16_20": (2016, 2020), "y21_25": (2021, 2025), "y24_25": (2024, 2025)}
NAV_TOL = 1e-12


def cash_fraction(ledger: Sequence[Mapping[str, Any]]) -> pd.Series:
    """엔진 원장 → 날마다 마지막 기록의 현금 ÷ 체결 뒤 NAV."""
    f = pd.DataFrame({"date": [r["date"] for r in ledger], "cash": [float(r["cash"]) for r in ledger],
                      "nav": [float(r["nav_after"]) for r in ledger]})
    last = f.groupby("date", sort=False).last()
    out = (last["cash"] / last["nav"].where(last["nav"] > 0)).fillna(1.0)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).as_unit("ns")
    return out


def cd91_adjust(nav: pd.Series, cash_frac: pd.Series, monthly: pd.Series) -> pd.Series:
    """엔진 NAV(현금 수익 0)에 CD 91일물 현금 수익을 더한다: 보정 일간 수익 = 엔진 일간 수익 + 전날 현금 비중 × 현금 수익."""
    days = nav.index
    r_eng = nav.pct_change().fillna(0.0).to_numpy()
    c_prev = cash_frac.reindex(days).ffill().shift(1).fillna(0.0).to_numpy()
    out = float(nav.iloc[0]) * np.cumprod(1.0 + r_eng + c_prev * t7rl.cash_daily(days, monthly))
    return pd.Series(out, index=days)


def period_stats(nav: pd.Series, y0: int, y1: int, fills: pd.DatetimeIndex) -> tuple[float, float]:
    """y0 첫 체결일 종가부터 y1 마지막 날까지의 CAGR·MDD (탐색 결과표와 같은 식)."""
    start = fills[fills.year == y0][0]
    s = nav[(nav.index >= start) & (nav.index.year <= y1)]
    s = s / s.iloc[0]
    n = len(s) - 1
    return float(s.iloc[-1] ** (252 / n) - 1), float((1 - s / s.cummax()).max())


def as_ns(s: pd.Series) -> pd.Series:
    s = s.astype("float64").copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).as_unit("ns")
    return s


def read_nav(path) -> pd.Series:
    f = pd.read_csv(path, parse_dates=["date"])
    return pd.Series(f["nav"].to_numpy(dtype=float), index=pd.DatetimeIndex(f["date"]).as_unit("ns"))


def rel_diff(a: pd.Series, b: pd.Series) -> float:
    if not a.index.equals(b.index):
        return float("inf")
    return float(np.max(np.abs(a.to_numpy() / b.to_numpy() - 1.0)))


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    t_all = time.time()
    run_dir = t7rl.RUN_DIR
    man = t7rl.manifest(run_dir)
    t7rl.verify_files(sorted(man["files"]), run_dir)
    print(f"고정 파일 {len(man['files'])}개 지문 확인 ({run_dir.relative_to(io.ROOT).as_posix()})", flush=True)

    cfg = load_prereg(PREREG_PATH)
    risk = RiskLayerParams.from_prereg(cfg)
    buffer = t7rl.t7_buffer(BufferRuleParams.from_prereg(cfg))
    rate, tax = t7rl.cost_rates(cfg)
    tables = {n: io.read_table(n) for n in TABLES}
    bm = kospi200(tables["index"])
    all_dates = signal_dates(tables["universe"], bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)
    dates = [d for d in all_dates if int(d[:4]) in YEARS]
    sig = io.read_table("signals", run_id=SIG_RUN)
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig[sig["date"].isin(pd.to_datetime(dates))].reset_index(drop=True)
    members = members_by_date(tables["universe"], dates)
    feats = feature_table(tables["prices"], bm, members, risk)
    inputs = engine_inputs(*(tables[n] for n in ENGINE_TABLES))
    m9 = m9_schedule(M9_RUN, all_dates[0], all_dates[-1])
    last_period = period_end(inputs.bm_close.index, dates[-1], YEARS[-1])
    snaps = build_snapshots(dates, t7rl.smooth_signals(sig), feats, tables["universe"], tables["sector"], inputs.bm_close,
                            m9.cooldown_at, risk)
    print(f"입력 읽음: 신호일 {len(dates)}개 ({dates[0]}~{dates[-1]}), {time.time() - t_all:.0f}초", flush=True)

    def engine(ledger: bool):
        return build_engine(inputs, cfg, record_ledger=ledger, cooldowns=m9.cooldowns, event_liquidations=m9.liquidations)

    n = len(dates)
    k200 = as_ns(inputs.bm_close)
    cal = k200.index[k200.index <= pd.Timestamp(last_period[1])]
    fills = pd.DatetimeIndex([cal[cal.searchsorted(d, side="right")] for d in pd.to_datetime(dates)])
    px, raw, cd91 = t7rl.frozen_inputs(run_dir)
    ref = run_dir / "reference"
    runs: dict[str, dict[str, Any]] = {}
    checks: dict[str, Any] = {}

    def execute(name: str, es_all: list[float]) -> tuple[pd.Series, pd.DataFrame]:
        t = time.time()
        pos = t7rl.build_positions(snaps, engine(False), buffer, risk, es_all)
        eng = engine(True)
        rep = replay(pos, dates, eng, last_period)
        nav = as_ns(rep.nav)
        sm = summary(rep, inputs.bm_close, pos)
        adj = cd91_adjust(nav, cash_fraction(eng.ledger), cd91)
        r: dict[str, Any] = {"official": {k: sm[k] for k in ("cagr", "mdd", "ir", "annual_excess", "annual_turnover", "mean_cash",
                                                             "first_fill", "last_day", "final_nav")}}
        for tag, s in (("cash0", nav), ("cd91", adj)):
            r[tag] = {p: dict(zip(("cagr", "mdd"), period_stats(s, y0, y1, fills))) for p, (y0, y1) in PERIODS.items()}
        yl = adj.groupby(adj.index.year).last()
        r["yearly_cd91"] = {int(y): float(v) for y, v in (yl / yl.shift(1) - 1).items() if y >= 2017}
        r["yearly_cd91"][2016] = float(adj[adj.index.year == 2016].iloc[-1] / adj.iloc[0] - 1)
        runs[name] = r
        a, c = r["cash0"]["all"], r["cd91"]["all"]
        print(f"  {name}: 2016~2025 현금 0 {a['cagr']:+.2%}·{a['mdd']:.1%}, CD 91일물 {c['cagr']:+.2%}·{c['mdd']:.1%} "
              f"({time.time() - t:.0f}초)", flush=True)
        return nav, pos

    # (1)·(2) E = 1과 T7-80-C53
    nav_e1, _ = execute("t7e1", [1.0] * n)
    checks["e1_nav_rel_diff"] = rel_diff(nav_e1, read_nav(ref / "e1_nav.csv"))
    nav_c53, _ = execute("t7c53", [t7rl.E_C53] * n)
    checks["c53_nav_rel_diff"] = rel_diff(nav_c53, read_nav(ref / "c53_nav.csv"))
    # (3) 고정 바구니 가격 = 상수 × E = 1 NAV (2016-01-11부터)
    tail = px[px.index >= nav_e1.index[0]]
    c = float(tail.iloc[0] / nav_e1.iloc[0])
    checks["basket_px_rel_diff"] = rel_diff(tail, (c * nav_e1).reindex(tail.index))

    # (4)·(5) 특성과 walk-forward E 경로
    reb = dates[0::t7rl.EVERY]
    fm = t7rl.feature_matrix(px, t7rl.us_table(raw), reb)
    rf = pd.read_csv(ref / "wf_features.csv", parse_dates=["date"])
    checks["features_same_dates"] = bool((pd.DatetimeIndex(rf["date"]) == pd.DatetimeIndex(fm["date"])).all())
    checks["features_max_abs_diff"] = float(np.max(np.abs(fm[list(t7rl.FEATS)].to_numpy() - rf[list(t7rl.FEATS)].to_numpy())))
    rc = t7rl.cash_daily(px.index, cd91)
    pols = {y: t7rl.load_policy(y, run_dir, verify=False) for y in YEARS}
    dec = t7rl.exposure_path(px, fm, lambda d: pols[d.year], rc, rate, tax)
    rd = pd.read_csv(ref / "wf_decisions.csv", parse_dates=["date"])
    checks["decisions_max_abs_diff"] = float(np.max(np.abs(dec["exposure"].to_numpy() - rd["exposure"].to_numpy())))
    nav_rl, pos_rl = execute("t7rl", t7rl.expand(dec["exposure"].tolist(), n))
    rn = pd.read_csv(ref / "wf_nav.csv", parse_dates=["date"])
    checks["rl_nav_rel_diff"] = rel_diff(nav_rl, pd.Series(rn["nav_raw"].to_numpy(float), index=pd.DatetimeIndex(rn["date"]).as_unit("ns")))

    # (7) 폴드 2026 정책: 검증 구간 2024~2025
    rv = pd.read_csv(ref / "val2026_decisions.csv", parse_dates=["date"])
    final = t7rl.load_policy(t7rl.FINAL_FOLD, run_dir, verify=False)
    fv = t7rl.feature_matrix(px, t7rl.us_table(raw), rv["date"])
    dv = t7rl.exposure_path(px, fv, lambda d: final, rc, rate, tax)
    checks["val2026_max_abs_diff"] = float(np.max(np.abs(dv["exposure"].to_numpy() - rv["exposure"].to_numpy())))

    # 리허설: 2025년을 홀드아웃처럼 (현금 100%에서 새로 시작, 4주 평균도 2025 신호로만, 폴드 2025 정책) build_period로 돌린다.
    # 2026 홀드아웃 실행(run_holdout)과 같은 코드 경로다. walk-forward 2025와는 시작 보유·장부가 달라 같을 필요는 없다.
    t = time.time()
    d25 = [d for d in dates if d.startswith("2025")]
    s25 = sig[sig["date"].dt.year == 2025].reset_index(drop=True)
    snaps25 = build_snapshots(d25, t7rl.smooth_signals(s25), feats, tables["universe"], tables["sector"], inputs.bm_close,
                              m9.cooldown_at, risk)
    lp25 = period_end(inputs.bm_close.index, d25[-1], 2025)
    reh = t7rl.build_period(d25, snaps25, engine, inputs.bm_close.index, cfg, px[px.index.year <= 2024], raw, cd91, pols[2025])
    rehearsal = {"record": reh.record}
    for name in ("t7rl", "t7c53", "t7e1"):
        rp = replay(reh.positions[name], d25, engine(True), lp25)
        s = as_ns(rp.nav)
        rehearsal[name] = {"return_2025": float(s.iloc[-1] / s.iloc[0] - 1.0), "mdd_2025": float((1 - s / s.cummax()).max())}
    wf25 = dec[pd.DatetimeIndex(dec["date"]).year == 2025]["exposure"]
    rehearsal["mean_e_walk_forward_2025"] = float(wf25.mean())
    rehearsal["chain_check"] = bool(reh.basket_px[reh.basket_px.index <= "2024-12-30"].equals(px[px.index <= "2024-12-30"]))
    print(f"  리허설 2025 (새 시작): T7-80-RL {rehearsal["t7rl"]["return_2025"]:+.2%}·{rehearsal['t7rl']['mdd_2025']:.1%} "
          f"(평균 E {reh.record['mean_exposure']:.2f}, walk-forward 2025 {rehearsal['mean_e_walk_forward_2025']:.2f}), "
          f"C53 {rehearsal["t7c53"]["return_2025"]:+.2%}·{rehearsal['t7c53']['mdd_2025']:.1%} ({time.time() - t:.0f}초)", flush=True)

    ok = (checks["e1_nav_rel_diff"] <= NAV_TOL and checks["c53_nav_rel_diff"] <= NAV_TOL and checks["basket_px_rel_diff"] <= NAV_TOL
          and checks["features_same_dates"] and checks["features_max_abs_diff"] == 0.0 and checks["decisions_max_abs_diff"] == 0.0
          and checks["rl_nav_rel_diff"] <= NAV_TOL and checks["val2026_max_abs_diff"] == 0.0)
    checks["all_same"] = bool(ok)

    dec_out = dec.assign(date=dec["date"].dt.strftime("%Y-%m-%d"))
    dec_out.to_csv(run_dir / "decisions_wf.csv", index=False)
    pd.DataFrame({"date": nav_rl.index.strftime("%Y-%m-%d"), "t7rl": nav_rl.to_numpy(), "t7c53": nav_c53.reindex(nav_rl.index).to_numpy(),
                  "t7e1": nav_e1.reindex(nav_rl.index).to_numpy()}).to_csv(run_dir / "daily_nav.csv", index=False)
    e = dec["exposure"].to_numpy()
    yrs = pd.DatetimeIndex(dec["date"]).year
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=io.ROOT, capture_output=True, text=True).stdout.strip()
    (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    out = {"model": t7rl.MODEL_ID, "run_id": t7rl.RUN_ID, "git_commit": commit, "inputs": {"signals": SIG_RUN, "m9": m9.record},
           "checks": checks, "runs": runs,
           "exposure": {"mean": float(e.mean()), "min": float(e.min()), "max": float(e.max()),
                        "changes": int((np.abs(np.diff(e)) > 1e-12).sum()), "decisions": int(len(e)),
                        "by_year": {int(y): float(e[yrs == y].mean()) for y in YEARS}},
           "val2026": man.get("val2026"), "rehearsal_2025": rehearsal, "seconds": round(time.time() - t_all, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("대조: " + ", ".join(f"{k} {v}" for k, v in checks.items()), flush=True)
    print(f"{'모두 같다' if ok else '다른 값이 있다'} → {run_dir / 'metrics.json'} ({(time.time() - t_all) / 60:.1f}분)", flush=True)
    if not ok:
        raise SystemExit(1)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
