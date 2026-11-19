"""G5 트랙 RL 선별 (스크래치). 공식 rl_fold_report·rl_wf_report와 같은 입력·환경·학습 호출을 쓰고
변형은 PortfolioEnv 하위 클래스(XEnv)의 행동 해석·보상과 PPO 인자만 바꾼다.

  train <vid> <fold> <seed> : 폴드 하나·시드 하나 학습 → runs/<vid>/y<fold>_s<seed>/
  eval  <vid> <seed>        : 선별 연도(2016~2020) 시험 운용 → G5 참고 판정 (버퍼 = 사전등록 규칙 + 같은 신호·M8·M9)
  worker <jobfile> <slot>   : jobfile의 train 작업 중 slot 번호(0/1)에 해당하는 것을 차례로 실행
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from common import ALL_YEARS, HERE, M9_RUN, REPO, SEARCH_YEARS, Ctx, Ledger, brief, buffer_run, g5_judgment

import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO

from src.backtest.metrics import validation_ir
from src.common import io
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.env_check_report import FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE, TABLES, feature_table, kospi200, members_by_date, signal_dates
from src.portfolio.params import PREREG_PATH, RiskLayerParams, RLParams, load_prereg
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_env import ACTION_BOUND, PortfolioEnv
from src.portfolio.rl_fold_report import (ENGINE_TABLES, fold_split, m9_schedule, ml_signal_table, policy_sha256,
                                          read_label_end_dates)
from src.portfolio.rl_train import DEVICE, ValidationIRCallback, rollout

END_DATES = REPO / "files/C02_labels_20261109_preholdout/end_dates.parquet"
RLSIG = "20261117_A_rlsig"
RUNS = HERE / "runs"

# ---- 변형 정의 ------------------------------------------------------------------
# x: 환경 변형, ppo: PPO 인자, steps, lam: λ 고정값(None이면 v7 공식 폴드 λ), rl: RLParams 바꿀 값, ckpt: best|last
VARIANTS: dict[str, dict] = {
    "R0": {"why": "대조 (사전등록 그대로)"},
    "R-nocash": {"x": {"nocash": True}, "why": "현금 로짓 없음, 현금은 M8만"},
    "R-band-0.01": {"x": {"band": 0.01}, "why": "비중 변화 < 0.01이면 유지 (회전율)"},
    "R-band-0.02": {"x": {"band": 0.02}, "why": "비중 변화 < 0.02이면 유지 (회전율)"},
    "R-smooth-0.5": {"x": {"smooth": 0.5}, "why": "목표 = 0.5 직전 + 0.5 새 비중 (회전율)"},
    "R-smoothk-0.5": {"x": {"smooth": 0.5, "keep_invested": True}, "why": "평활 0.5, 투자 비중은 새 행동 값 유지 (학습에도 적용)"},
    "R-tpen-0.002": {"x": {"tpen": 0.002}, "why": "보상에서 0.002 × 회전율 추가 차감"},
    "R-tpen-0.005": {"x": {"tpen": 0.005}, "why": "보상에서 0.005 × 회전율 추가 차감"},
    "R-pre20": {"x": {"pre20": True}, "why": "rank 상위 20개를 먼저 고르고 RL은 그 20개 + 현금 비중 (PM 3번 방식 추정)"},
    "R-lam-0": {"lam": 0.0, "why": "λ 0 고정"},
    "R-lam-2": {"lam": 2.0, "why": "λ 2 고정"},
    "R-steps-1m": {"steps": 1_000_000, "why": "학습량 1,000,000스텝"},
    "R-ppo-ent": {"ppo": {"ent_coef": 0.01}, "why": "탐색 보너스"},
    "R-ppo-lr": {"ppo": {"learning_rate": 1e-4}, "why": "작은 학습률"},
    "R-ppo-nsteps": {"ppo": {"n_steps": 4096}, "why": "긴 rollout"},
    "R-ppo-gamma": {"ppo": {"gamma": 0.9}, "why": "짧은 할인 (주간 보상)"},
    "R-slots-20": {"rl": {"candidate_top_k": 20, "slots": 40}, "why": "후보 rank ≤ 20"},
    "R-last": {"ckpt": "last", "src": "R0", "why": "마지막 스텝 체크포인트 (R0 학습 결과의 last.zip)"},
    # 11/19 추가 (시드 0 선별 뒤): 보상 회전율 차감을 늘린 판, 학습 변형 정책 + 운용 때 평활 조합 (학습 없음, src 체크포인트)
    "R-tpen-0.01": {"x": {"tpen": 0.01}, "why": "보상에서 0.01 × 회전율 추가 차감 (0.002 → 0.005에서 IR이 오른 방향 연장)"},
    "R-tpen-0.005+Ik-0.5": {"x": {"smooth": 0.5, "keep_invested": True}, "src": "R-tpen-0.005",
                            "why": "R-tpen-0.005 정책 + 운용 때 평활 0.5 (투자 비중 유지)"},
    "R-tpen-0.005+Ik-0.7": {"x": {"smooth": 0.7, "keep_invested": True}, "src": "R-tpen-0.005",
                            "why": "R-tpen-0.005 정책 + 운용 때 평활 0.7 (투자 비중 유지)"},
}


class XEnv(PortfolioEnv):
    """x가 비어 있으면 PortfolioEnv와 수치가 같다 (대조 R0로 확인)."""

    def __init__(self, *args, x: dict | None = None, **kw):
        self.x = dict(x or {})
        super().__init__(*args, **kw)
        if self.x.get("nocash"):
            self.action_space = spaces.Box(-ACTION_BOUND, ACTION_BOUND, shape=(self.rl.slots,), dtype=np.float32)

    def _start_run(self) -> None:
        super()._start_run()
        self._prev_raw: dict[str, float] = {}

    def _topk(self, w: np.ndarray) -> np.ndarray:
        n = len(w)
        order = np.lexsort((np.arange(n), -w))
        keep = np.zeros(n, dtype=bool)
        keep[order[: self.rl.max_holdings]] = True
        keep &= w >= self.rl.min_weight
        return np.where(keep, w, 0.0)

    def _interpret(self, action: np.ndarray) -> np.ndarray:
        x, n = self.x, self.rl.slots
        if x.get("nocash") or x.get("pre20"):
            tradable = self._tradable.copy()
            if x.get("pre20"):  # 슬롯은 rank 오름차순이다
                idx = np.flatnonzero(tradable)[: self.rl.max_holdings]
                tradable = np.zeros(n, dtype=bool)
                tradable[idx] = True
            if x.get("nocash"):
                if not tradable.any():
                    w = np.zeros(n)
                else:
                    z = np.where(tradable, action[:n], -np.inf)
                    e = np.exp(z - z.max())
                    w = self._topk(e / e.sum())
                    w = w / w.sum()
            else:
                z = np.append(np.where(tradable, action[:n], -np.inf), action[n])
                e = np.exp(z - z.max())
                w = self._topk(e[:n] / e.sum())
        else:
            w = super()._interpret(action)
        if x.get("band") or x.get("smooth"):
            tick = self._slot_tickers
            prev = np.zeros(n)
            prev[: len(tick)] = [self._prev_raw.get(t, 0.0) for t in tick]
            invested = float(w.sum())  # 새 행동이 고른 투자 비중
            if x.get("smooth"):
                a = float(x["smooth"])
                w = (1.0 - a) * w + a * prev
            if x.get("band"):
                w = np.where(np.abs(w - prev) < float(x["band"]), prev, w)
            w = self._topk(w)
            if x.get("keep_invested") and w.sum() > 0.0:  # 섞은 뒤에도 투자 비중은 새 행동의 값으로 (현금 누적 방지)
                w = w * (invested / w.sum())
            if w.sum() > 1.0:
                w = w / w.sum()
        tick = self._slot_tickers
        self._prev_raw = {tick[j]: float(w[j]) for j in np.flatnonzero(w[: len(tick)] > 0.0)}
        return w

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        if self.x.get("tpen") and info.get("executed"):
            r -= float(self.x["tpen"]) * float(info["turnover"])
        return obs, r, term, trunc, info


def spec(vid: str) -> dict:
    v = dict(VARIANTS[vid])
    v.setdefault("x", {})
    v.setdefault("ppo", {})
    v.setdefault("steps", None)
    v.setdefault("lam", None)
    v.setdefault("rl", {})
    v.setdefault("ckpt", "best")
    return v


def official_lambda(fold: int) -> float:
    m = json.loads((REPO / f"reports/20261117_B_rlwf_y{fold}/metrics.json").read_text(encoding="utf-8"))
    return float(m["main_training"]["lambda"])


class Base:
    """표·설정을 한 번 읽는다 (작업자 프로세스마다 하나)."""

    def __init__(self):
        self.cfg = load_prereg(PREREG_PATH)
        self.rl, self.risk = RLParams.from_prereg(self.cfg), RiskLayerParams.from_prereg(self.cfg)
        self.tables = {n: io.read_table(n) for n in TABLES}
        self.bm = kospi200(self.tables["index"])
        self.dates = signal_dates(self.tables["universe"], self.bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)
        self.ends = read_label_end_dates(END_DATES)
        self.inputs = engine_inputs(*(self.tables[n] for n in ENGINE_TABLES))
        self.m9 = m9_schedule(M9_RUN, self.dates[0], self.dates[-1])

    def fold_data(self, fold: int):
        fs = fold_split(self.dates, self.bm.index, fold, self.cfg, actual_end_dates=self.ends)
        needed = sorted({pd.Timestamp(d).strftime("%Y-%m-%d") for d in list(fs.train) + list(fs.val)})
        members = members_by_date(self.tables["universe"], needed)
        signals = ml_signal_table(f"{RLSIG}_y{fold}", fs)
        features = feature_table(self.tables["prices"], self.bm, members, self.risk)
        snaps = build_snapshots(needed, signals, features, self.tables["universe"], self.tables["sector"],
                                self.inputs.bm_close, self.m9.cooldown_at, self.risk)
        engine = build_engine(self.inputs, self.cfg, cooldowns=self.m9.cooldowns, event_liquidations=self.m9.liquidations)
        return fs, snaps, engine


def train(base: Base, vid: str, fold: int, seed: int) -> dict:
    v = spec(vid)
    out = RUNS / vid / f"y{fold}_s{seed}"
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    rl = replace(base.rl, **v["rl"]) if v["rl"] else base.rl
    lam = official_lambda(fold) if v["lam"] is None else float(v["lam"])
    steps = v["steps"] or rl.total_timesteps_per_seed
    t0 = time.perf_counter()
    fs, snaps, engine = base.fold_data(fold)
    by = {s.date: s for s in snaps}
    iso = lambda ds: [pd.Timestamp(d).strftime("%Y-%m-%d") for d in ds]
    train_snaps, val_snaps = [by[d] for d in iso(fs.train)], [by[d] for d in iso(fs.val)]
    val_env = XEnv(val_snaps, engine, rl, base.risk, 0.0, "full", calendar=base.dates, x=v["x"])
    make_env = lambda lam_: XEnv(train_snaps, engine, rl, base.risk, lam_, "random", calendar=base.dates, x=v["x"])
    prep = time.perf_counter() - t0
    model = PPO("MlpPolicy", make_env(lam), seed=seed, device=DEVICE, **v["ppo"])
    cb = ValidationIRCallback(val_env, rl.eval_every_steps, validation_ir, out / "best.zip")
    model.learn(total_timesteps=steps, callback=cb)
    model.save(out / "last.zip")
    log = pd.DataFrame(cb.records, columns=["timesteps", "val_ir"])
    log.to_csv(out / "eval_log.csv", index=False)
    res = {"vid": vid, "fold": fold, "seed": seed, "lambda": lam, "steps": steps, "best_val_ir": cb.best_ir,
           "best_timesteps": cb.best_timesteps, "policy_sha256": policy_sha256(out / "best.zip"),
           "prep_sec": round(prep, 1), "train_sec": round(time.perf_counter() - t0 - prep, 1)}
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    return res


def evaluate(vid: str, seed: int, ctx: Ctx | None = None, buffer_full: dict | None = None) -> dict:
    v = spec(vid)
    ctx = ctx or Ctx(SEARCH_YEARS)
    base_rl = RLParams.from_prereg(ctx.cfg)
    rl = replace(base_rl, **v["rl"]) if v["rl"] else base_rl
    pols = {}
    for y in ctx.years:
        d = RUNS / v.get("src", vid) / f"y{y}_s{seed}"
        pols[y] = PPO.load(d / ("last.zip" if v["ckpt"] == "last" else "best.zip"), device=DEVICE)
    engine = ctx.engine(True, ledger=False)
    env = XEnv(ctx.snaps(), engine, rl, ctx.risk, 0.0, "full", calendar=ctx.all_dates, x=v["x"])
    run = rollout(env, lambda d: pols[int(d[:4])])
    pos = run.positions
    s = ctx.evaluate(pos, with_m9=True)
    if buffer_full is None:
        buffer_full = ctx.evaluate(buffer_run(ctx, ctx.snaps(), mode="full"), with_m9=True)
    g5 = g5_judgment({seed: s}, buffer_full, ctx.kpi)
    out = {"vid": vid, "seed": seed, "years": f"{ctx.years[0]}-{ctx.years[-1]}", **brief(s),
           "buffer": brief(buffer_full), "ir_gain": g5["ir_gain"], "turnover_ratio": g5["turnover_ratio"],
           "cagr_not_lower": g5["cagr_not_lower"], "pass_by_ir": g5["pass_by_ir"],
           "pass_by_alternative": g5["pass_by_alternative"], "pass": g5["pass"]}
    (RUNS / vid).mkdir(parents=True, exist_ok=True)  # src 변형(R-last 등)은 학습 폴더가 없다
    pos.to_parquet(RUNS / vid / f"positions_s{seed}_{out['years']}.parquet", index=False)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        print(json.dumps(train(Base(), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])), ensure_ascii=False, default=float))
    elif cmd == "worker":
        jobs = [l.split() for l in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if l.strip()]
        slot, nslots = int(sys.argv[3]), int(sys.argv[4])
        base = Base()
        for k, (vid, fold, seed) in enumerate(jobs):
            if k % nslots != slot:
                continue
            try:
                r = train(base, vid, int(fold), int(seed))
                print(json.dumps(r, ensure_ascii=False, default=float), flush=True)
            except Exception as e:
                print(json.dumps({"vid": vid, "fold": fold, "seed": seed, "error": repr(e)}, ensure_ascii=False), flush=True)
    elif cmd == "eval":
        print(json.dumps(evaluate(sys.argv[2], int(sys.argv[3])), ensure_ascii=False, default=float))
