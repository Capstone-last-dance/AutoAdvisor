r"""T7 비중 에이전트 v11 (실험 D, docs 33절): v10 + 세계 위험 특성 8개 (11/20 자율 실험, B 실행).

v10과 다른 점 (실행 전 고정 11/20)
- 관측 = 국내 특성 20개 + 세계 위험 특성 8개 + 지금 비중. 세계 위험 특성은 `global_features.parquet`(FRED, 2025-12-31까지)다.
  - g_vix(VIX 로그), g_vix_chg(21일 변화), g_baa(Baa − 10년 국채), g_baa_chg(21일 변화), g_krw_r21·g_krw_r63(원/달러 로그 변화),
    g_nq_r63(나스닥 63일 로그수익), g_nq_ma200(나스닥 200일선 괴리).
  - 결정일 t에는 t보다 앞선 미국 날짜의 값을 쓴다 (미국 종가는 한국 장 마감 전에 나온다).
  - 투자 자산은 그대로 주식 바구니와 현금이다 (13절 사용자 결정). 세계 지표는 관측에만 쓴다.
- 근거 (학습 구간 진단, docs 33절): VIX 수준과 다음 2주 바구니 수익의 순위상관이 2011~2015 +0.12, 2016~2020 +0.15로 같은 방향이다.
- 나머지는 v10과 같다: 보상(v8 보상 − κ 이탈 비용), 행동 12단계, 공식 walk-forward 폴드, PPO, 검증 체크포인트, 앙상블, C 엔진 평가.
- 바구니·가격은 runs/t7i(v9)의 파일을 복사하고, 특성 파일에 세계 위험 특성 열을 붙인다 (prepare_data).

실행 (저장소 루트, PowerShell, 창 1개): .\.venv\Scripts\python.exe -I explore\b09\run.py t7_agent11.py run 8
- 탐색 결과다 (R-15). 결과: t7k_runs.csv, t7k_judgment.json, runs/t7k/, 원장 t7k|*.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import t7_agent8 as base
from cash_yield import CashYield
from common import ALL_YEARS, HERE, REPO, Ctx, Ledger
import exitrule as ex
from src.common import io
from src.portfolio.env_check_report import FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE, kospi200, signal_dates
from t7_agent8 import (BAND, E_C53, EP_LEN, EVAL_EVERY, EVERY, FEATS, LEVELS, MDD_FLOOR, N, SEEDS, STEPS, Book, decide, full_es,
                       norm_stats, normalize, obs_vec, path_from, period_stats, run_cash)
from turnover import ledger_has, smooth_signals

base.LEVELS = np.append(base.LEVELS[:11], E_C53)  # v10: 0~1 11단계 + 0.466107 (12단계). v8 함수들이 이 값을 쓴다
LEVELS = base.LEVELS
GCOLS = ["g_vix", "g_vix_chg", "g_baa", "g_baa_chg", "g_krw_r21", "g_krw_r63", "g_nq_r63", "g_nq_ma200"]
base.FEATS = list(base.FEATS[:20]) + GCOLS  # v11: 국내 20개 + 세계 위험 8개. v8 함수들이 이 값을 쓴다
FEATS = base.FEATS

RUNS = HERE / "runs" / "t7k"
base.RUNS, base.EVAL = RUNS, RUNS / "eval"  # v8의 바구니·특성 함수가 v9 폴더에 쓰고 읽게 한다
base.EXT_PATH, base.PX_PATH, base.FEAT_PATH = RUNS / "basket.parquet", RUNS / "px.parquet", RUNS / "features.parquet"
EVAL = base.EVAL
JOBS = RUNS / "jobs.txt"
LOGS = RUNS / "logs"
LAMBDAS = base.LAMBDAS
TEST_YEARS = tuple(range(2016, 2026))
FINAL_FOLD = 2026  # 학습 2011~2023, 검증 2024~2025. 테스트는 12/11 개봉 뒤 (지금은 학습·검증만)
FOLDS = TEST_YEARS + (FINAL_FOLD,)
PERIODS = {"all": (2016, 2025), "y16_20": (2016, 2020), "y21_25": (2021, 2025), "y24_25": (2024, 2025)}
END = pd.Timestamp("2025-12-30")
TRACK = "T7 비중 에이전트 v11 (v10 + 세계 위험 특성 8개)"
_ns = base._ns


def prepare_data() -> None:
    """runs/t7i(v9)의 바구니·가격을 복사하고, 특성 파일에 세계 위험 특성(global_features.parquet)을 붙인다."""
    src = HERE / "runs" / "t7i"
    RUNS.mkdir(parents=True, exist_ok=True)
    for n in ("basket.parquet", "px.parquet"):
        shutil.copy(src / n, RUNS / n)
    f = pd.read_parquet(src / "features.parquet")
    gf = pd.read_parquet(HERE / "global_features.parquet")
    if len(gf) != len(f) or (pd.to_datetime(gf["date"]).to_numpy() != pd.to_datetime(f["date"]).to_numpy()).any():
        raise RuntimeError("세계 위험 특성의 결정일이 바구니 특성과 다르다")
    if pd.to_datetime(gf["us_date"]).max() > pd.Timestamp("2025-12-31"):
        raise RuntimeError("세계 위험 특성에 2026년 값이 있다")
    for c in GCOLS:
        f[c] = gf[c].to_numpy(dtype=float)
    f["ok"] = np.isfinite(f[FEATS].to_numpy(dtype=float)).all(axis=1)
    f.to_parquet(base.FEAT_PATH, index=False)
    print(f"특성 {len(FEATS)}개 (세계 위험 {len(GCOLS)}개 포함), 결정일 {len(f)}개, 특성 다 있는 날 {int(f['ok'].sum())}개", flush=True)


def lam_id(lam: float) -> str:
    return base.lam_id(lam)


def run_dir(lam: float, y: int, seed: int) -> Path:
    return RUNS / f"y{y}" / lam_id(lam) / f"s{seed}"


def train_rows(data, y: int) -> pd.DataFrame:
    """폴드 Y의 학습 리밸런싱일: 2011 ~ Y−3 (특성이 다 있는 날)."""
    f = data.f
    return f[f["ok"] & (f["year"] <= y - 3)].reset_index(drop=True)


def val_metrics(data, models, mu, sd, y: int) -> dict:
    """폴드 Y의 검증 Y−2 ~ Y−1: Y−2 첫 리밸런싱일 E 0.466107에서 시작, Y−1 마지막 거래일에서 끝 (현금 = CD 91일물)."""
    last = int(np.flatnonzero(data.days.year <= y - 1)[-1])
    rec = []
    es, _ = path_from(data, models, mu, sd, y - 2, y - 1, finish=last, rec=rec)
    nav = pd.Series([v for _, v in rec], index=data.days[[d for d, _ in rec]])
    n = len(nav) - 1
    cagr = float(nav.iloc[-1] ** (252 / n) - 1)
    mdd = float((1 - nav / nav.cummax()).max())
    return {"cagr": cagr, "mdd": mdd, "score": cagr / max(mdd, MDD_FLOOR), "mean_e": float(np.mean(es))}


def static_j(data, g: pd.DataFrame, lam: float) -> dict:
    """학습 리밸런싱일 g에서 고정 E로 1년(EP_LEN) 에피소드를 모든 시작점에서 돌린 평균 목표 (100 × ln 잔고 − λ × 100 × MDD)."""
    dec = g["k"].to_numpy(dtype=np.int64)
    out = {}
    for e in LEVELS:
        vals = []
        for p in range(0, len(dec) - EP_LEN):
            b, tot = Book(data.px, int(dec[p]), float(e), rc=data.rc), 0.0
            for q in range(p, p + EP_LEN):
                lv, dm = b.run(int(dec[q + 1]), float(e))
                tot += 100.0 * lv - lam * 100.0 * dm
            vals.append(tot)
        out[float(e)] = float(np.mean(vals))
    return out


def calibrate_kappa(data, g: pd.DataFrame, lam: float) -> tuple[float, dict]:
    """학습 자료의 고정 비중 중 0.466107이 가장 좋아지는 이탈 비용 (1.25배 여유, 최소 0.005)."""
    J = static_j(data, g, lam)
    j0 = J[float(E_C53)]
    ratios = [(v - j0) / (100.0 * abs(e - E_C53)) for e, v in J.items() if abs(e - E_C53) > 1e-9]
    return 1.25 * max(max(ratios), 0.005), J


def make_devcost(env, kappa: float):
    """학습 환경 보상에서 κ × 100 × |E − 0.466107| × 구간 거래일 ÷ 252를 뺀다 (평가에는 없다)."""
    import gymnasium as gym

    class DevCost(gym.Wrapper):
        def step(self, action):
            u = self.env.unwrapped
            i_, j_ = int(u.dec[u.p]), int(u.dec[u.p + 1])
            e_ = float(LEVELS[int(action)])
            o, r, term, trunc, info = self.env.step(action)
            return o, float(r) - kappa * 100.0 * abs(e_ - E_C53) * (j_ - i_) / 252.0, term, trunc, info

    return DevCost(env)


def train_job(data, lam: float, y: int, seed: int, out: Path, steps: int = STEPS, every: int = EVAL_EVERY) -> dict:
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(1)
    g = train_rows(data, y)
    mu, sd = norm_stats(g[FEATS].to_numpy(dtype=float))
    Z = normalize(g[FEATS].to_numpy(dtype=float), mu, sd)
    kappa, jstat = calibrate_kappa(data, g, lam)
    env = make_devcost(base.make_env_class()(data.px, Z, g["k"].to_numpy(dtype=np.int64), lam, data.rc), kappa)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    class ValCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best = -np.inf

        def _on_step(self) -> bool:
            if self.num_timesteps % every == 0:
                v = val_metrics(data, [self.model], mu, sd, y)
                log.append({"timesteps": int(self.num_timesteps), **v})
                if v["score"] > self.best:
                    self.best = v["score"]
                    self.model.save(out / "best.zip")
            return True

    t0 = time.time()
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0)
    model.learn(total_timesteps=int(steps), callback=ValCallback())
    model.save(out / "last.zip")
    lg = pd.DataFrame(log)
    lg.to_csv(out / "eval_log.csv", index=False)
    best = lg.loc[lg["score"].idxmax()]
    train = [str(g["date"].iloc[0].date()), str(g["date"].iloc[-1].date())]
    (out / "stats.json").write_text(json.dumps({"cols": FEATS, "mu": list(map(float, mu)), "sd": list(map(float, sd)), "lam": lam,
                                                "fold": y, "train": train, "train_rows": int(len(g)), "kappa": kappa,
                                                "static_j": {f"{e:.6f}": v for e, v in jstat.items()}}), encoding="utf-8")
    res = {"lam": lam, "kappa": kappa, "fold": y, "seed": seed, "steps": int(steps), "train_rows": int(len(g)), "train": train, "val": [y - 2, y - 1],
           "best_timesteps": int(best["timesteps"]), "best_val_score": float(best["score"]), "best_val_cagr": float(best["cagr"]),
           "best_val_mdd": float(best["mdd"]), "best_val_mean_e": float(best["mean_e"]), "seconds": round(time.time() - t0, 1),
           "sb3": stable_baselines3.__version__, "torch": torch.__version__}
    (out / "done.json").write_text(json.dumps(res), encoding="utf-8")  # 마지막에 쓴다 (없으면 다시 학습)
    return res


# ---------------------------------------------------------------------------
# 작업 나누기
# ---------------------------------------------------------------------------

def job_list() -> list[tuple[float, int, int]]:
    return [(lam, y, s) for y in FOLDS for lam in LAMBDAS for s in SEEDS]


def write_jobs() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    JOBS.write_text("\n".join(f"{lam:g} {y} {s}" for lam, y, s in job_list()) + "\n", encoding="utf-8")
    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("plan", "t7k|wf"):
        ledger.plan("t7k|wf", TRACK,
                    "v11: v10 + 세계 위험 특성 8개(VIX·Baa 스프레드·원/달러·나스닥, FRED, t보다 앞선 미국 날짜), "
                    "공식 walk-forward 폴드 (폴드 Y: 학습 2011~Y−3, 검증 Y−2~Y−1, 테스트 Y; Y = 2016~2025), λ는 폴드 검증으로 고름, "
                    f"PPO {STEPS:,}스텝, {EVAL_EVERY:,}스텝마다 검증 최고 체크포인트, 5시드 앙상블, 테스트 2016~2025 C 엔진. "
                    "폴드 2026(학습 2011~2023, 검증 2024~2025)은 학습만 하고 테스트는 12/11 홀드아웃 개봉 뒤 PM 승인 때",
                    "사용자 결정(11/20): 학습·검증·테스트를 자유롭게 배분. 2026년은 4.7절(학습·튜닝 2025-12-31까지, 개봉 전 홀드아웃 성과 계산 금지) 때문에 지금 쓰지 않음")
    print(f"작업 {len(job_list())}개 (폴드 {len(FOLDS)} × λ {len(LAMBDAS)} × 시드 {len(SEEDS)}) → {JOBS}", flush=True)


def worker(jobs_path: str, slot: int, nslots: int) -> None:
    data = base.Data()
    jobs = [l.split() for l in Path(jobs_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    mine = [j for k, j in enumerate(jobs) if k % nslots == slot]
    print(f"워커 {slot}/{nslots}: 작업 {len(mine)}개", flush=True)
    for n, (lam, y, seed) in enumerate(mine, 1):
        lam, y, seed = float(lam), int(y), int(seed)  # 작업 파일 값은 문자열이다 (11/20 첫 실행은 요약 출력에서 TypeError, 학습·done.json은 정상)
        out = run_dir(lam, y, seed)
        if (out / "done.json").exists():
            continue
        try:
            r = train_job(data, lam, y, seed, out)
            print(f"[{n}/{len(mine)}] 폴드 {y} λ {lam} 시드 {seed}: {r['seconds']:.0f}초, 학습 {r['train'][0]}~{r['train'][1]} "
                  f"({r['train_rows']}개), 검증 {y - 2}~{y - 1} 최고 {r['best_timesteps']:,}스텝 점수 {r['best_val_score']:+.2f} "
                  f"(평균 E {r['best_val_mean_e']:.2f})", flush=True)
        except Exception as e:  # 다음 작업으로 간다. 다시 돌리면 이 작업을 다시 시도한다
            print(f"[{n}/{len(mine)}] 오류 {lam} {y} {seed}: {e!r}", flush=True)
    print("done", flush=True)


def n_done() -> int:
    return sum((run_dir(l, y, s) / "done.json").exists() for l, y, s in job_list())


def train(nw: int) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    total, t0 = len(job_list()), time.time()
    print(f"2) 학습: 작업 {total}개 중 {n_done()}개 끝, 워커 {nw}개 시작 (로그 {LOGS})", flush=True)
    procs = []
    for i in range(nw):
        fh = open(LOGS / f"worker{i}.log", "a", encoding="utf-8")
        cmd = [sys.executable, "-I", "-X", "utf8", str(HERE / "run.py"), "t7_agent11.py", "worker", str(JOBS), str(i), str(nw)]
        procs.append((subprocess.Popen(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT), fh))
    last = -1
    while any(p.poll() is None for p, _ in procs):
        time.sleep(20)
        d = n_done()
        if d != last:
            print(f"  학습 {d}/{total} ({(time.time() - t0) / 60:.0f}분)", flush=True)
            last = d
    for _, fh in procs:
        fh.close()
    bad = [i for i, (p, _) in enumerate(procs) if p.returncode != 0]
    if bad:
        print(f"  워커 {bad}가 오류로 끝났다. 로그를 보고 resume으로 다시 돌린다", flush=True)
    print(f"  학습 끝: {n_done()}/{total} ({(time.time() - t0) / 60:.0f}분)", flush=True)


def status() -> None:
    print(f"학습: {n_done()}/{len(job_list())} 끝")
    for y in FOLDS:
        print(f"  폴드 {y}: " + ", ".join(f"λ {l:g} {sum((run_dir(l, y, s) / 'done.json').exists() for s in SEEDS)}/5" for l in LAMBDAS))


# ---------------------------------------------------------------------------
# 테스트 경로 (폴드 정책을 해마다 바꾸며 이어 붙임) → C 엔진
# ---------------------------------------------------------------------------

def load_fold(lam: float, y: int):
    from stable_baselines3 import PPO

    models = [PPO.load(run_dir(lam, y, s) / "best.zip", device="cpu") for s in SEEDS]
    st = json.loads((run_dir(lam, y, SEEDS[0]) / "stats.json").read_text(encoding="utf-8"))
    if st["cols"] != FEATS:
        raise RuntimeError(f"폴드 {y} λ {lam:g}: 특성 열이 다르다")
    return models, np.array(st["mu"]), np.array(st["sd"])


def wf_path(data, picks: dict) -> list[float]:
    """2016-01-08부터 2025년 마지막 리밸런싱일까지 E를 정한다. 리밸런싱일이 속한 해 Y의 폴드 정책을 쓰고,
    장부(보유·현금)는 해가 바뀌어도 잇는다 (시작 비중 E 0.466107, 현금 = CD 91일물)."""
    f = data.lv
    X = f[FEATS].to_numpy(dtype=float)
    yr = f["year"].to_numpy()
    dec = f["k"].to_numpy(dtype=np.int64)
    Z = {y: normalize(X, mu, sd) for y, (_, mu, sd) in picks.items()}
    book, es = Book(data.px, int(dec[0]), E_C53, rc=data.rc), []
    for k in range(len(dec)):
        y = int(yr[k])
        e = decide(picks[y][0], obs_vec(Z[y], k, book))
        es.append(e)
        j = int(dec[k + 1]) if k + 1 < len(dec) else len(data.px) - 1
        if j > book.i:
            book.run(j, e)
    return es


def evaluate() -> None:
    t_all = time.time()
    left = len(job_list()) - n_done()
    if left:
        print(f"아직 끝나지 않은 학습이 {left}개 있다 (status로 확인, resume으로 이어서)")
        return
    data = base.Data()
    print("3) 폴드마다 검증 구간 앙상블 점수로 λ 고르기 (엔진 밖 장부, 현금 = CD 91일물)", flush=True)
    sel, pols = {}, {}
    for y in FOLDS:
        sc = {}
        for lam in LAMBDAS:
            pols[(y, lam)] = load_fold(lam, y)
            sc[lam] = val_metrics(data, *pols[(y, lam)], y)
        pick = max(LAMBDAS, key=lambda l: sc[l]["score"])
        sel[y] = {"pick": pick, "scores": {f"{l:g}": v for l, v in sc.items()}}
        print(f"  폴드 {y} (검증 {y - 2}~{y - 1}): " + ", ".join(f"λ {l:g} {sc[l]['cagr']:+.1%}·{sc[l]['mdd']:.1%} (E {sc[l]['mean_e']:.2f})"
                                                       for l in LAMBDAS) + f" → λ {pick:g}", flush=True)
    fin = {"fold": FINAL_FOLD, "train": [2011, FINAL_FOLD - 3], "val": [FINAL_FOLD - 2, FINAL_FOLD - 1], "pick": sel[FINAL_FOLD]["pick"],
           "scores": sel[FINAL_FOLD]["scores"],
           "models": [str(run_dir(sel[FINAL_FOLD]["pick"], FINAL_FOLD, s) / "best.zip") for s in SEEDS],
           "note": "2026 홀드아웃 테스트는 12/11 개봉 뒤 PM 승인 기록이 있을 때만 (명세 4.7절, PM-14)"}
    RUNS.joinpath("final_2026.json").write_text(json.dumps(fin, ensure_ascii=False, indent=1), encoding="utf-8")

    print("4) 테스트 2016~2025 (C 엔진, 현금 = CD 91일물 보정)", flush=True)
    dates = [d for d in signal_dates(io.read_table("universe"), kospi200(io.read_table("index")).index, FIRST_SIGNAL_DATE,
                                     LAST_SIGNAL_DATE) if 2016 <= int(d[:4]) <= 2025]
    ctx = Ctx(ALL_YEARS)
    if list(ctx.dates) != dates:
        raise RuntimeError("Ctx 신호일이 다르다")
    snaps = ctx.snaps(signals=smooth_signals(ctx.signals, 4), key="sm4")
    b = replace(ctx.buffer, enter_rank_max=N, exit_rank_greater_than=N + BAND, max_holdings=N)
    k200 = ctx.inputs.bm_close.copy()
    k200.index = _ns(k200.index)
    cal = k200.index[k200.index <= END]
    fills = pd.DatetimeIndex([cal[cal.searchsorted(d, side="right")] for d in _ns(dates)])
    yr_reb = data.lv["year"].to_numpy()
    if len(yr_reb) != (len(dates) + 1) // 2:
        raise RuntimeError("리밸런싱일 수가 신호일 수와 맞지 않다")
    EVAL.mkdir(parents=True, exist_ok=True)
    results, CY = {}, CashYield()

    def engine(name: str, es_reb: list[float]) -> dict:
        t = time.time()
        m, nav_raw, cfrac = run_cash(ctx, snaps, b, full_es(es_reb, len(dates)))
        nav_raw = nav_raw.copy()
        nav_raw.index = _ns(nav_raw.index)
        nav = CY.adjust(nav_raw, cfrac)
        pd.DataFrame({"date": nav.index, "nav": nav.to_numpy(), "nav_raw": nav_raw.to_numpy(),
                      "cash_frac": cfrac.reindex(nav.index).ffill().to_numpy()}).to_parquet(EVAL / f"nav_{name}.parquet", index=False)
        es_a = np.array(es_reb)
        r = {"name": name, "cagr10_raw": float(m["cagr"]), "mdd10_raw": float(m["mdd"]), "mean_cash": float(cfrac.mean()),
             "turnover10": float(m["annual_turnover"]), "mean_e": float(es_a.mean()),
             "switches": int((np.abs(np.diff(es_a)) > 1e-9).sum())}
        for p, (y0, y1) in PERIODS.items():
            r[f"cagr_{p}"], r[f"mdd_{p}"] = period_stats(nav, y0, y1, fills)
        r["yearly"] = {int(y): float(v) for y, v in (lambda l: l / l.shift(1) - 1)(nav.groupby(nav.index.year).last()).items()
                       if y >= 2017}
        r["yearly"][2016] = float(nav[nav.index.year == 2016].iloc[-1] / nav.iloc[0] - 1)
        r["mean_e_by_year"] = {int(y): float(es_a[yr_reb == y].mean()) for y in TEST_YEARS}
        r["window_mean_e"] = {}
        for wname, a, z in [w_ for w_ in ex.WINDOWS if int(w_[1][:4]) >= 2016]:
            msk = ((data.lv["date"] >= pd.Timestamp(a)) & (data.lv["date"] <= pd.Timestamp(z))).to_numpy()
            r["window_mean_e"][wname] = float(es_a[msk].mean()) if msk.any() else None
        results[name] = r
        (EVAL / f"{name}.json").write_text(json.dumps({**r, "es": es_reb}, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"  {name}: 2016~2025 {r['cagr_all']:+.2%}·{r['mdd_all']:.1%}, 2021~2025 {r['cagr_y21_25']:+.2%}·{r['mdd_y21_25']:.1%}, "
              f"평균 E {r['mean_e']:.2f} ({time.time() - t:.0f}초)", flush=True)
        return r

    c53 = engine("c53", [E_C53] * len(yr_reb))
    ref = json.loads((base.SAMECASH / "cash.json").read_text(encoding="utf-8"))
    d = abs(c53["cagr10_raw"] - ref["summary_cagr"])
    print(f"  대조 (현금 보정 전): E 고정 CAGR {c53['cagr10_raw']:+.6%} vs 25절 T7-80-C53 {ref['summary_cagr']:+.6%} (차이 {d:.1e})", flush=True)
    if d > 1e-12:
        raise RuntimeError("E 고정 판이 25절 T7-80-C53과 다르다")

    pick_of = {y: sel[y]["pick"] for y in TEST_YEARS}
    paths = {"wf_ens": {y: pols[(y, pick_of[y])] for y in TEST_YEARS}}
    for i, s in enumerate(SEEDS):
        paths[f"wf_s{s}"] = {y: ([pols[(y, pick_of[y])][0][i]], *pols[(y, pick_of[y])][1:]) for y in TEST_YEARS}
    for lam in LAMBDAS:
        paths[f"wf_{lam_id(lam)}"] = {y: pols[(y, lam)] for y in TEST_YEARS}
    for name, picks in paths.items():
        es = wf_path(data, picks)
        engine(name, es)
        if name == "wf_ens" or name.startswith("wf_s"):
            engine(f"{name}_fixed", [float(np.mean(es))] * len(es))

    k = pd.read_parquet(base.KOSPI_PATH)
    kospi = pd.Series(k["close"].to_numpy(dtype=float), index=_ns(k["date"]))
    ref_rows = {}
    for nm, s in (("kospi", kospi), ("k200", k200), ("e1", base.navp(base.SAMECASH / "nav_e1.parquet"))):
        s = s[(s.index >= fills[0]) & (s.index <= END)]
        ref_rows[nm] = {f"{x}_{p}": v for p, (y0, y1) in PERIODS.items() for x, v in zip(("cagr", "mdd"), period_stats(s, y0, y1, fills))}
        if nm in ("kospi", "k200"):  # 지수 46.6% + 현금(CD 91일물), 매일 같은 비중, 비용 없음
            mx = CY.mix(s, E_C53)
            ref_rows[f"{nm}_mix"] = {f"{x}_{p}": v for p, (y0, y1) in PERIODS.items()
                                     for x, v in zip(("cagr", "mdd"), period_stats(mx, y0, y1, fills))}

    ens, fx = results["wf_ens"], results["wf_ens_fixed"]
    seeds_beat = int(sum(results[f"wf_s{s}"]["cagr_all"] > results[f"wf_s{s}_fixed"]["cagr_all"] for s in SEEDS))
    q1 = "(1) 타이밍 실력 (테스트 2016~2025: 앙상블 CAGR·MDD 모두 같은 평균 고정보다 낫고 시드 4/5 이상)"
    q2 = "(2) 고정 모델 대비 (테스트 2016~2025: CAGR > T7-80-C53, MDD < T7-80-C53)"
    j = {"폴드별 고른 λ (그 폴드 검증 점수 최고)": {str(y): sel[y]["pick"] for y in FOLDS},
         q1: bool(ens["cagr_all"] > fx["cagr_all"] and ens["mdd_all"] < fx["mdd_all"] and seeds_beat >= 4),
         q2: bool(ens["cagr_all"] > c53["cagr_all"] and ens["mdd_all"] < c53["mdd_all"]),
         "자기 고정보다 CAGR 높은 시드 수 (테스트 2016~2025)": seeds_beat}
    j["교체 후보"] = bool(j[q1] and j[q2])

    flat = []
    for n_, r in list(results.items()) + [(k_, {"name": k_, **v}) for k_, v in ref_rows.items()]:
        f_ = {k_: v for k_, v in r.items() if not isinstance(v, dict)}
        f_.update({f"y{y}": v for y, v in sorted(r.get("yearly", {}).items())})
        f_.update({f"e{y}": v for y, v in r.get("mean_e_by_year", {}).items()})
        flat.append(f_)
    rd = pd.DataFrame(flat)
    rd.to_csv(HERE / "t7k_runs.csv", index=False)
    train_info = {f"y{y}_{lam_id(l)}_s{s}": json.loads((run_dir(l, y, s) / "done.json").read_text(encoding="utf-8")) for l, y, s in job_list()}
    (HERE / "t7k_judgment.json").write_text(json.dumps({"judgment": j, "selection": {str(y): v for y, v in sel.items()}, "final_2026": fin,
                                                        "results": results, "refs": ref_rows, "train": train_info},
                                                       ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("result", "t7k|판정"):
        ledger.result("t7k|wf_ens", **{k_: ens[k_] for k_ in ("cagr_all", "mdd_all", "cagr_y21_25", "mdd_y21_25", "cagr_y24_25", "mdd_y24_25",
                                                             "mean_e", "turnover10")})
        ledger.result("t7k|판정", **{k_: v for k_, v in j.items() if not isinstance(v, dict)})

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    show = rd.copy()
    for c in [c for c in show.columns if c.startswith(("cagr", "mdd")) or (c[0] == "y" and c[1:].isdigit())]:
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(1)
    for c in ["mean_e", "turnover10"] + [c for c in show.columns if c[0] == "e" and c[1:].isdigit()]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    print("\n[기간별 CAGR·MDD %: all = 테스트 2016~2025 (판정), y16_20, y21_25, y24_25]")
    print(show[[c for c in ["name", "cagr_all", "mdd_all", "cagr_y16_20", "mdd_y16_20", "cagr_y21_25", "mdd_y21_25", "cagr_y24_25",
                            "mdd_y24_25", "mean_e", "switches", "turnover10"] if c in show.columns]].to_string(index=False))
    print("\n[연도별 수익률 %]")
    print(show[["name"] + [c for c in show.columns if c[0] == "y" and c[1:].isdigit()]].to_string(index=False))
    print("\n[연도별 평균 E]")
    print(show[["name"] + [c for c in show.columns if c[0] == "e" and c[1:].isdigit()]].dropna(how="all", axis=0).to_string(index=False))
    print("\n[판정] (실행 전 고정)")
    for q, v in j.items():
        print(f"  {q}: {v if not isinstance(v, bool) else ('만족' if v else '미달')}")
    print(f"\n폴드 2026 (학습 2011~2023, 검증 2024~2025): λ {fin['pick']:g}, 모델은 {RUNS / 'final_2026.json'}. "
          "2026 테스트는 12/11 개봉 뒤 PM 승인 때.")
    print(f"done ({(time.time() - t_all) / 60:.0f}분)")


def smoke() -> None:
    if not base.PX_PATH.exists():
        prepare_data()
    data = base.Data()
    sm = RUNS / "_smoke"
    if sm.exists():
        shutil.rmtree(sm)
    r = train_job(data, 0.25, 2021, 0, sm, steps=2_048, every=1_024)
    print(json.dumps(r, ensure_ascii=False))
    from stable_baselines3 import PPO

    m = PPO.load(sm / "best.zip", device="cpu")
    st = json.loads((sm / "stats.json").read_text(encoding="utf-8"))
    es = wf_path(data, {y: ([m], np.array(st["mu"]), np.array(st["sd"])) for y in TEST_YEARS})
    print(f"테스트 경로 점검: 리밸런싱일 {len(es)}개, E 범위 {min(es):.1f}~{max(es):.1f}")
    print("smoke ok")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    nw = int(sys.argv[2]) if len(sys.argv) > 2 and cmd in ("run", "resume") else 5
    if cmd == "run":
        if RUNS.exists():
            shutil.rmtree(RUNS)  # 이전 v11 결과를 지우고 처음부터
        print("1) 바구니·특성을 처음부터 만든다", flush=True)
        prepare_data()
        write_jobs()
        train(nw)
        evaluate()
    elif cmd == "resume":
        if not base.PX_PATH.exists():
            print("1) 바구니·특성을 만든다", flush=True)
            prepare_data()
        if not JOBS.exists():
            write_jobs()
        train(nw)
        evaluate()
    elif cmd == "worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "status":
        status()
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "smoke":
        smoke()
    else:
        raise SystemExit("명령: run N | resume N | status | evaluate | smoke")


if __name__ == "__main__":
    main()
