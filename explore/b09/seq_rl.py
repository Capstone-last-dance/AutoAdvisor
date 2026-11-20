r"""순차 적합 신호로 RL 다시 학습 (11/20 사용자 결정: 23절 A 답장 뒤 "이걸 바탕으로 학습 진행").

질문: B-07 PPO는 학습·검증 구간에서 교차 적합 신호(rlsig)로 배운다. 그 신호는 맨 위 종목이 시험 신호보다 훨씬 좋다
(21절: 2016~2022 같은 해 상위 20 초과 연 +15.7% vs 시험 +3.8%). 학습·검증 신호를 시험 신호와 같은 방식으로 만든 것
(순차 적합)으로 바꾸면 시험 성과가 나아지나.

순차 적합 신호 (새로 만들지 않는다)
- A 시험 신호 20261117_A_wfnews(2016~2025)가 곧 순차 적합 신호다. y년 행은 폴드 y 모델(2011 ~ (y−3)년 학습,
  (y−2)·(y−1)년 검증)이 낸 예측이라 y년 이전 데이터만 쓴다. A 답장(23절)의 "y 이전 해만으로 학습 → y 예측"과 같은 절차다.
- 2014·2015년 순차 적합 신호(A sequential_fit_check.py)는 아직 없어 쓰지 않는다. 그래서 학습 구간은 2016년부터다.
  A 신호가 오면 FIRST = 2014로 폴드 2018~2025를 따로 돌릴 수 있다 (이번 판정 밖).

실행 (저장소 루트, PowerShell). 모두 사용자가 돌린다.
  1) 점검 (5분 안팎):   .\.venv\Scripts\python.exe -I explore\b09\run.py seq_rl.py smoke
  2) 작업 파일:         .\.venv\Scripts\python.exe -I explore\b09\run.py seq_rl.py jobs
  3) 학습 (작업 36개, 작업당 약 6~7분, 창 2개면 2시간 안팎):
       창 1: .\.venv\Scripts\python.exe -I explore\b09\run.py seq_rl.py worker explore\b09\jobs_seq.txt 0 2
       창 2: .\.venv\Scripts\python.exe -I explore\b09\run.py seq_rl.py worker explore\b09\jobs_seq.txt 1 2
     진행: .\.venv\Scripts\python.exe -I explore\b09\run.py seq_rl.py status
  4) 평가 (C 엔진, 15분 안팎): .\.venv\Scripts\python.exe -I explore\b09\run.py seq_rl.py evaluate
  끊기면 같은 명령을 다시 돌린다. 끝난 학습은 건너뛴다.

구성 (실행 전 고정 11/20)
- 사전등록 B-07 설정 그대로: PPO(SB3 기본값) 300,000스텝, 10,000스텝마다 검증 IR, 검증 IR 최고 체크포인트,
  52주 에피소드, 후보 rank ≤ 40, 최대 20종목, M8 + M9, C 엔진. λ는 v7 공식 폴드 값(20261117_B_rlwf_y{Y}) 고정.
- 바꾸는 것은 학습 구간 시작(2016)과 신호 출처뿐이다.
- 폴드 Y = 2020~2025. 학습 = 2016 ~ (Y−3)년, 검증 = (Y−2)·(Y−1)년 (A make_folds, first_train_year = 2016,
  C 실제 라벨 종료일 purge). 2019 이전 폴드는 학습 구간이 1년 이하라 52주 에피소드를 만들 수 없다.
- Q-seq: 학습·검증 신호 = 순차 적합 신호 (wfnews의 같은 날짜 행).
- Q-xfit (대조): 같은 날짜, 신호 = 20261117_A_rlsig_y{Y} (공식 학습 신호: 학습 연도는 교차 적합, 검증 연도는 폴드 Y 모델).
  Q-seq와 학습 날짜·설정·시드가 모두 같고 신호 값만 다르다.
- 공식 (참고, 학습 없음): v7 공식 체크포인트 (2011 ~ (Y−3)년 교차 적합 신호로 학습).
- 시드 0·1·2. 학습 작업 = 2판 × 6폴드 × 3시드 = 36개.

평가: 2020~2025 연속 (2020년 첫 신호일에 현금에서 시작). 시험 신호 wfnews, M8 + M9, C 엔진 (공식 RL 시험 운용과 같다).

판정 (실행 전 고정 11/20)
(1) 신호 효과 (주 질문): Q-seq 3시드 평균 CAGR > Q-xfit 3시드 평균 CAGR, 그리고 같은 시드끼리 비교해 Q-seq가 높은 시드가 2개 이상.
(2) 공식 대비: Q-seq 3시드 평균 CAGR > 공식 정책(시드 0·1·2) 평균 CAGR.
- 참고: 사전등록 버퍼(G5 비교 대상, 같은 신호·M8·M9), T7 보유 80(22절 10년 연속 실행의 2020~2025 부분, 2020년 첫 체결일
  잔고로 나눔), 코스피 종합·코스피 200. MDD, 코스피 종합 대비 연 초과·IR, 연 회전율, 평균 현금, 연도별, 2020과
  2021~2025 나눠 보기, 검증 IR과 그 폴드 시험 연도 초과수익(코스피 200 대비)의 순위 상관(판마다 18점).
- 해석: (1) 만족 → 학습 신호의 낙관이 RL 성과를 깎는다. 순차 적합 학습을 PM·A에 제안한다 (DEC-08 사안).
  (1) 미달 → 학습 신호의 낙관은 RL 약점의 주원인이 아니다.
- 주의: 2020~2025는 여러 번 본 구간이다 (16·21·22절). 넘어도 forward test(PM이 허락하면 2026 홀드아웃)로 확인해야 한다.
  탐색 결과다 (R-15). 결과: seq_runs.csv, seq_judgment.json, runs/seq/ (git 제외), 원장 seq|*.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from bench_kospi import END, KOSPI_PATH, stats as kospi_stats
from common import HERE, REPO, SIG_RUN, Ctx, Ledger, brief, buffer_run, iso
import exitrule as ex
from rl_screen import RLSIG, Base, XEnv, official_lambda
from turnover import ledger_has, split_cagr, yearly

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.backtest.metrics import validation_ir
from src.common import io
from src.models.splits import make_folds
from src.portfolio.engine_adapter import build_engine
from src.portfolio.env_check_report import feature_table, members_by_date
from src.portfolio.params import RLParams
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_fold_report import policy_sha256
from src.portfolio.rl_train import DEVICE, ValidationIRCallback, rollout

FIRST = 2016
FOLDS = (2020, 2021, 2022, 2023, 2024, 2025)
SEEDS = (0, 1, 2)
VIDS = ("Q-seq", "Q-xfit")
OFFICIAL = "official"
WHY = {"Q-seq": "학습·검증 신호 = 순차 적합 (A 시험 신호 wfnews의 같은 날짜 행)",
       "Q-xfit": "대조: 같은 날짜·설정, 신호 = 공식 학습 신호 rlsig_y{Y} (교차 적합 + 폴드 모델)"}
LABEL = {"Q-seq": "Q-seq 순차 적합 학습", "Q-xfit": "Q-xfit 교차 적합 학습 (대조)", OFFICIAL: "공식 정책 (2011~ 교차 적합)"}
OUT = HERE / "runs" / "seq"
EVAL = OUT / "eval"
JOBS = HERE / "jobs_seq.txt"
HOLD80 = HERE / "runs" / "hold" / "nav_N80.parquet"


class SeqBase(Base):
    """rl_screen.Base와 같은 표·설정. 폴드 분할(학습 시작 2016)과 신호 출처만 바꾼다."""

    def __init__(self):
        super().__init__()
        self._seq = None

    def split(self, fold: int):
        h = int(self.cfg["labels"]["main_horizon_days"])
        (f,) = make_folds(pd.DatetimeIndex(pd.to_datetime(list(self.dates))),
                          pd.DatetimeIndex(pd.to_datetime(list(self.bm.index))), h=h, test_years=(fold,),
                          first_train_year=FIRST, actual_end_dates=self.ends)
        return f

    def source(self, vid: str, fold: int) -> pd.DataFrame:
        if vid == "Q-seq":
            if self._seq is None:
                self._seq = io.read_table("signals", run_id=SIG_RUN)
            return self._seq
        if vid == "Q-xfit":
            return io.read_table("signals", run_id=f"{RLSIG}_y{fold}")
        raise ValueError(f"모르는 판: {vid}")

    def signals(self, vid: str, fold: int, f) -> pd.DataFrame:
        """학습·검증 신호일의 행만 남긴다. 하루라도 없으면 멈춘다."""
        src = self.source(vid, fold)
        want = {iso(d) for d in list(f.train) + list(f.val)}
        day = pd.to_datetime(src["date"]).dt.strftime("%Y-%m-%d")
        keep = day.isin(want)
        missing = sorted(want - set(day[keep]))
        if missing:
            raise ValueError(f"{vid} 폴드 {fold}: 신호가 없는 학습·검증 신호일 {len(missing)}개 (첫 번째 {missing[0]})")
        return src[keep].reset_index(drop=True)

    def seq_fold_data(self, vid: str, fold: int):
        f = self.split(fold)
        needed = sorted({iso(d) for d in list(f.train) + list(f.val)})
        members = members_by_date(self.tables["universe"], needed)
        signals = self.signals(vid, fold, f)
        features = feature_table(self.tables["prices"], self.bm, members, self.risk)
        snaps = build_snapshots(needed, signals, features, self.tables["universe"], self.tables["sector"],
                                self.inputs.bm_close, self.m9.cooldown_at, self.risk)
        engine = build_engine(self.inputs, self.cfg, cooldowns=self.m9.cooldowns, event_liquidations=self.m9.liquidations)
        return f, snaps, engine


def train(base: SeqBase, vid: str, fold: int, seed: int, out: Path | None = None, steps: int | None = None,
          every: int | None = None) -> dict:
    """rl_screen.train과 같은 학습 호출 (R0 = 사전등록). 폴드 데이터만 seq_fold_data에서 온다."""
    out = out or OUT / vid / f"y{fold}_s{seed}"
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    rl = base.rl
    lam = official_lambda(fold)
    steps = steps or rl.total_timesteps_per_seed
    every = every or rl.eval_every_steps
    t0 = time.perf_counter()
    f, snaps, engine = base.seq_fold_data(vid, fold)
    by = {s.date: s for s in snaps}
    train_snaps, val_snaps = [by[iso(d)] for d in f.train], [by[iso(d)] for d in f.val]
    val_env = XEnv(val_snaps, engine, rl, base.risk, 0.0, "full", calendar=base.dates)
    train_env = XEnv(train_snaps, engine, rl, base.risk, lam, "random", calendar=base.dates)
    prep = time.perf_counter() - t0
    model = PPO("MlpPolicy", train_env, seed=seed, device=DEVICE)
    cb = ValidationIRCallback(val_env, every, validation_ir, out / "best.zip")
    model.learn(total_timesteps=steps, callback=cb)
    model.save(out / "last.zip")
    pd.DataFrame(cb.records, columns=["timesteps", "val_ir"]).to_csv(out / "eval_log.csv", index=False)
    res = {"vid": vid, "fold": fold, "seed": seed, "lambda": lam, "steps": steps,
           "train": [iso(f.train[0]), iso(f.train[-1]), len(f.train)], "val": [iso(f.val[0]), iso(f.val[-1]), len(f.val)],
           "best_val_ir": cb.best_ir, "best_timesteps": cb.best_timesteps,
           "policy_sha256": policy_sha256(out / "best.zip") if (out / "best.zip").exists() else None,
           "prep_sec": round(prep, 1), "train_sec": round(time.perf_counter() - t0 - prep, 1)}
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    return res


def job_lines() -> list[list[str]]:
    return [l.split() for l in JOBS.read_text(encoding="utf-8").splitlines() if l.strip()]


def done(vid: str, fold, seed) -> bool:
    return (OUT / vid / f"y{fold}_s{seed}" / "result.json").exists()


# ---- 명령 ---------------------------------------------------------------------


def smoke() -> None:
    """폴드마다 분할·신호 범위를 보고, 가장 짧은 폴드(2020)를 두 판으로 4,096스텝만 학습해 본다 (runs/seq/_smoke)."""
    t0 = time.time()
    base = SeqBase()
    print(f"입력 읽음 ({time.time() - t0:.0f}초)", flush=True)
    seq_days = set(pd.to_datetime(base.source("Q-seq", FOLDS[0])["date"]).dt.strftime("%Y-%m-%d"))
    for fold in FOLDS:
        f = base.split(fold)
        want = {iso(d) for d in list(f.train) + list(f.val)}
        x = base.source("Q-xfit", fold)
        x_days = set(pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d"))
        print(f"폴드 {fold}: 학습 {iso(f.train[0])} ~ {iso(f.train[-1])} ({len(f.train)}일), 검증 {iso(f.val[0])} ~ "
              f"{iso(f.val[-1])} ({len(f.val)}일), 시험 첫 신호일 {iso(f.test[0])}, λ {official_lambda(fold)} | "
              f"순차 신호 없는 날 {len(want - seq_days)}, rlsig 없는 날 {len(want - x_days)}", flush=True)
        if want - seq_days or want - x_days:
            raise RuntimeError(f"폴드 {fold}: 학습·검증 신호일 중 신호가 없는 날이 있다")
        if len(f.train) < base.rl.episode_steps + 1:
            raise RuntimeError(f"폴드 {fold}: 학습 신호일 {len(f.train)}개 < 에피소드 {base.rl.episode_steps + 1}")

    # 같은 날짜에서 두 신호가 얼마나 다른가 (마지막 폴드의 학습·검증 신호일)
    f = base.split(FOLDS[-1])
    a, b = (base.signals(v, FOLDS[-1], f)[["date", "ticker", "rank"]] for v in VIDS)
    a, b = (t.assign(date=pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")) for t in (a, b))
    m = a.merge(b, on=["date", "ticker"], suffixes=("_seq", "_x"))
    if len(m) != len(a) or len(m) != len(b):
        raise RuntimeError(f"두 신호의 (날짜, 종목)이 다르다 ({len(a)}, {len(b)}, 겹침 {len(m)})")
    rho, top = [], []
    for _, g in m.groupby("date"):  # rank는 날짜 안 순위라 rank끼리의 Pearson = 순위 상관
        rho.append(float(np.corrcoef(g["rank_seq"], g["rank_x"])[0, 1]))
        top.append(len(set(g.loc[g["rank_seq"] <= 20, "ticker"]) & set(g.loc[g["rank_x"] <= 20, "ticker"])) / 20)
    print(f"폴드 {FOLDS[-1]} 학습·검증 신호일 {len(rho)}일: 두 신호 rank 순위 상관 평균 {np.mean(rho):.3f}, "
          f"상위 20 겹침 평균 {np.mean(top):.0%}", flush=True)

    sm = OUT / "_smoke"
    shutil.rmtree(sm, ignore_errors=True)
    for vid in VIDS:
        r = train(base, vid, FOLDS[0], 0, out=sm / vid, steps=4096, every=2048)
        print(f"{vid} 폴드 {FOLDS[0]} 4,096스텝: 검증 IR {r['best_val_ir']:+.3f}, 준비 {r['prep_sec']}초, "
              f"학습 {r['train_sec']}초", flush=True)
    print("smoke ok")


def jobs() -> None:
    lines = [f"{vid} {fold} {seed}" for seed in SEEDS for fold in FOLDS for vid in VIDS]
    JOBS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger = Ledger(HERE / "ledger.jsonl")
    for vid in VIDS:
        if not ledger_has("plan", f"seq|{vid}"):
            ledger.plan(f"seq|{vid}", "순차 적합 학습 (B-07 PPO)",
                        f"{WHY[vid]}. 폴드 {FOLDS[0]}~{FOLDS[-1]}, 학습 {FIRST}~(Y−3)·검증 (Y−2)·(Y−1), 시드 0·1·2, "
                        "사전등록 PPO 300,000스텝·검증 IR 최고, λ v7 공식 폴드 값",
                        "사용자 결정(11/20): 23절 A 답장 뒤 순차 적합 신호로 학습. 교차 적합 학습 신호는 맨 위 종목이 낙관적(21절)")
    print(f"{JOBS.name}: 작업 {len(lines)}개 (끝난 것 {sum(done(*j) for j in job_lines())}개)")


def worker(jobfile: str, slot: int, nslots: int) -> None:
    lines = [l.split() for l in Path(jobfile).read_text(encoding="utf-8").splitlines() if l.strip()]
    base = SeqBase()
    for k, (vid, fold, seed) in enumerate(lines):
        if k % nslots != slot or done(vid, fold, seed):
            continue
        try:
            r = train(base, vid, int(fold), int(seed))
            print(json.dumps(r, ensure_ascii=False, default=float), flush=True)
        except Exception as e:
            print(json.dumps({"vid": vid, "fold": fold, "seed": seed, "error": repr(e)}, ensure_ascii=False), flush=True)


def status() -> None:
    lines = job_lines()
    print(f"끝난 학습 {sum(done(*j) for j in lines)}/{len(lines)}")
    for vid in VIDS:
        row = []
        for fold in FOLDS:
            row.append(f"{fold}:" + "".join("o" if done(vid, fold, s) else "." for s in SEEDS))
        print(f"  {vid:7s} " + "  ".join(row))


def policy_path(vid: str, year: int, seed: int) -> Path:
    if vid == OFFICIAL:
        return REPO / f"reports/20261117_B_rlwf_y{year}/seed_{seed}/best.zip"
    return OUT / vid / f"y{year}_s{seed}" / "best.zip"


def best_val_ir(vid: str, year: int, seed: int) -> float:
    if vid == OFFICIAL:
        m = json.loads((REPO / f"reports/20261117_B_rlwf_y{year}/metrics.json").read_text(encoding="utf-8"))
        return next(float(p["best_val_ir"]) for p in m["main_training"]["per_seed"] if int(p["seed"]) == seed)
    return float(json.loads((OUT / vid / f"y{year}_s{seed}" / "result.json").read_text(encoding="utf-8"))["best_val_ir"])


def as_ns(s: pd.Series) -> pd.Series:
    s = s.astype("float64").copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).as_unit("ns")
    return s


def nav_stats(nav: pd.Series, bms: dict, fills: pd.DatetimeIndex) -> dict:
    s = kospi_stats(nav, bms, fills)
    c1, c2 = split_cagr(nav)
    d1, d2 = ex.half_mdd(nav)
    return {**s, "cagr_2020": c1, "cagr_2021_2025": c2, "mdd_2020": d1, "mdd_2021_2025": d2, "yearly": yearly(nav)}


def evaluate() -> None:
    missing = [j for j in job_lines() if not done(*j)]
    if missing:
        print(f"아직 끝나지 않은 학습 {len(missing)}개: {[' '.join(j) for j in missing[:6]]}")
        sys.exit(1)
    t0 = time.time()
    ctx = Ctx(FOLDS)
    rl = RLParams.from_prereg(ctx.cfg)
    snaps = ctx.snaps()
    k = pd.read_parquet(KOSPI_PATH)
    kospi = pd.Series(k["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(k["date"])).as_unit("ns"))
    k200 = as_ns(ctx.inputs.bm_close)
    cal = k200.index[k200.index <= END]
    fills = pd.DatetimeIndex([cal[cal.searchsorted(pd.Timestamp(d), side="right")] for d in ctx.dates])
    bms = {"kospi": kospi, "k200": k200}
    span = lambda s: s[(s.index >= fills[0]) & (s.index <= END)] / s[fills[0]]
    print(f"입력 읽음 ({time.time() - t0:.0f}초). 기간 {iso(fills[0])} ~ {iso(END)}", flush=True)

    rows, navs = [], {}

    def add(config: str, seed, nav: pd.Series, s: dict | None) -> dict:
        nav = as_ns(nav)
        st = nav_stats(nav, bms, fills)
        r = {"config": config, "seed": seed, **{k_: v for k_, v in st.items() if k_ != "yearly"},
             **({f"m5_{k_}": v for k_, v in brief(s).items()} if s is not None else {}),
             **{f"y{y}": v for y, v in st["yearly"].items()}}
        rows.append(r)
        navs[(config, seed)] = nav
        return r

    for name, ser in (("코스피 종합", kospi), ("코스피 200", k200)):
        add(name, None, span(ser), None)
    h = pd.read_parquet(HOLD80)
    nav80 = pd.Series(h["nav"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(h["date"])).as_unit("ns"))
    add("T7 보유 80 (22절, 10년 실행의 부분)", None, span(nav80), None)
    buf = ctx.evaluate(buffer_run(ctx, snaps, mode="full"), with_m9=True)
    add("사전등록 버퍼 + M8 + M9", None, buf["_rep"].nav, buf)
    print(f"  기준 4개 ({time.time() - t0:.0f}초)", flush=True)

    EVAL.mkdir(parents=True, exist_ok=True)
    for vid in (OFFICIAL,) + VIDS:
        for seed in SEEDS:
            t = time.time()
            pols = {y: PPO.load(policy_path(vid, y, seed), device=DEVICE) for y in FOLDS}
            env = XEnv(snaps, ctx.engine(True, ledger=False), rl, ctx.risk, 0.0, "full", calendar=ctx.all_dates)
            pos = rollout(env, lambda d, p=pols: p[int(d[:4])]).positions
            pos.to_parquet(EVAL / f"{vid}_s{seed}.parquet", index=False)
            s = ctx.evaluate(pos, with_m9=True)
            r = add(vid, seed, s["_rep"].nav, s)
            print(f"  {LABEL[vid]} 시드 {seed}: CAGR {r['cagr']:+.2%}, MDD {r['mdd']:.1%}, 회전율 {r['m5_annual_turnover']:.1f}, "
                  f"현금 {r['m5_mean_cash']:.0%} ({time.time() - t:.0f}초)", flush=True)

    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("config", "seed")]
    means = df[df["config"].isin((OFFICIAL,) + VIDS)].groupby("config", sort=False)[num].mean().reset_index()
    means["seed"] = "평균"
    out = pd.concat([df, means], ignore_index=True)
    out.to_csv(HERE / "seq_runs.csv", index=False)

    cagr = {vid: {s: float(df[(df.config == vid) & (df.seed == s)]["cagr"].iloc[0]) for s in SEEDS} for vid in (OFFICIAL,) + VIDS}
    mean = {vid: float(np.mean(list(v.values()))) for vid, v in cagr.items()}
    pairs = sum(cagr["Q-seq"][s] > cagr["Q-xfit"][s] for s in SEEDS)
    j = {"(1) 신호 효과: Q-seq 평균 CAGR > Q-xfit, 같은 시드 2/3 이상": bool(mean["Q-seq"] > mean["Q-xfit"] and pairs >= 2),
         "(2) 공식 대비: Q-seq 평균 CAGR > 공식 (시드 0·1·2)": bool(mean["Q-seq"] > mean[OFFICIAL]),
         "Q-seq가 높은 시드 수": int(pairs), "평균 CAGR": mean}

    # 검증 IR ↔ 그 폴드 시험 연도 초과수익 (코스피 200 대비, 연도별 수익률 차)
    k200y = yearly(span(k200))
    corr = {}
    for vid in (OFFICIAL,) + VIDS:
        pts = [(best_val_ir(vid, y, s), yearly(navs[(vid, s)])[y] - k200y[y]) for y in FOLDS for s in SEEDS]
        p = pd.DataFrame(pts, columns=["val_ir", "test_excess"])
        corr[vid] = float(np.corrcoef(p["val_ir"].rank(), p["test_excess"].rank())[0, 1])
    j["참고: 검증 IR과 시험 연도 초과수익의 순위 상관"] = corr

    (HERE / "seq_judgment.json").write_text(json.dumps({"folds": FOLDS, "seeds": SEEDS, "first_train_year": FIRST,
                                                        "period": [iso(fills[0]), iso(END)], "judgment": j},
                                                       ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("result", "seq|판정"):
        for vid in (OFFICIAL,) + VIDS:
            r = means[means["config"] == vid].iloc[0]
            ledger.result(f"seq|{vid}", seeds=list(SEEDS), period=f"{FOLDS[0]}-{FOLDS[-1]}",
                          **{c: float(r[c]) for c in ("cagr", "mdd", "cagr_2020", "cagr_2021_2025", "excess_kospi", "ir_kospi",
                                                      "m5_annual_turnover", "m5_mean_cash")})
        ledger.result("seq|판정", **j)

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    show = out.copy()
    show["config"] = show["config"].map(lambda c: LABEL.get(c, c))
    pct = [c for c in show.columns if c.startswith(("cagr", "mdd", "excess", "m5_mean_cash", "m5_cagr", "m5_mdd"))
           or (c.startswith("y") and c[1:].isdigit())]
    for c in pct:
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(1)
    for c in ("ir_kospi", "ir_k200", "m5_ir", "m5_annual_turnover", "final_nav"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    print(f"\n[{iso(fills[0])} ~ {iso(END)}, 시험 신호 wfnews, C 엔진. RL·버퍼는 M8 + M9]")
    print(show[["config", "seed", "cagr", "cagr_2020", "cagr_2021_2025", "mdd", "mdd_2021_2025", "excess_kospi", "ir_kospi",
                "m5_annual_turnover", "m5_mean_cash", "final_nav"]].to_string(index=False))
    print("\n[연도별 수익률 %]")
    print(show[["config", "seed"] + [c for c in show.columns if c.startswith("y") and c[1:].isdigit()]].to_string(index=False))
    print("\n[판정] (실행 전 고정)")
    for q, v in j.items():
        if q.startswith("("):
            print(f"  {q}: {'만족' if v else '미달'}")
    print(f"  평균 CAGR: " + ", ".join(f"{LABEL[k_]} {v:+.2%}" for k_, v in mean.items()) + f" | Q-seq가 높은 시드 {pairs}/3")
    print("  참고 검증 IR ↔ 시험 연도 초과수익 순위 상관: " + ", ".join(f"{LABEL[k_]} {v:+.2f}" for k_, v in corr.items()))
    print("done")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "smoke":
        smoke()
    elif cmd == "jobs":
        jobs()
    elif cmd == "worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "status":
        status()
    elif cmd == "evaluate":
        evaluate()
    else:
        raise SystemExit(f"모르는 명령: {cmd} (smoke | jobs | worker <작업 파일> <슬롯> <슬롯 수> | status | evaluate)")
