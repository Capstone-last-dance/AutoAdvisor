r"""학습형 M8: 장을 보고 종목 포트폴리오의 투자 비중을 정하는 RL 에이전트 (11/19 사용자 결정).

질문: 에이전트가 시장 상태를 보고 투자 비중 E_t를 정하면 (장이 좋으면 늘리고 나쁘면 줄이면),
프로젝트 포트폴리오(개별 종목 + 현금)의 최대 낙폭을 줄이면서 수익도 지킬 수 있나 (두 마리 토끼).

실행 (저장소 루트, PowerShell). 모두 사용자가 돌린다.
  1) 특성 (한 번, 1~3분):      .\.venv\Scripts\python.exe -I explore\b09\run.py mt_agent.py features
  2) 점검 (5분 안팎):          .\.venv\Scripts\python.exe -I explore\b09\run.py mt_agent.py smoke
  3) 학습 (작업 200개):        .\.venv\Scripts\python.exe -I explore\b09\run.py mt_agent.py jobs
                               .\.venv\Scripts\python.exe -I explore\b09\run.py mt_agent.py worker explore\b09\jobs_mt.txt 0 4
                               (창 4개에서 끝 숫자 0·1·2·3). 진행: ... mt_agent.py status
  4) 평가 (C 엔진):            .\.venv\Scripts\python.exe -I explore\b09\run.py mt_agent.py evaluate
  끊기면 같은 명령을 다시 돌린다. 끝난 학습·평가는 건너뛴다.

역할 (프로젝트 구조 그대로)
- 종목 선택: A v7 시험 신호(SIG_RUN) → 버퍼 규칙(사전등록). 체결: C 엔진 (다음 거래일 종가, 종목 10%·업종 30% 한도,
  수수료·슬리피지·매도세). M8 Step 1~3(급락 차단·한도)와 M9 경보는 그대로다.
- 에이전트는 M8 Step 4~5 자리의 투자 비중 E_t만 정한다 (사전등록 규칙 E_t = clip(0.157/σ, 0.3, 1), 국면 ≤ 0.7을 대신한다).
  E_t는 0·25·50·75·100% 중 하나다 (사전등록 M8의 하한 0.3과 다르다. 운용에 쓰려면 PM 결정이 필요하다).
- 코스피 200은 사지 않는다. 시장 상태를 읽는 입력과 비교 기준으로만 쓴다.

학습 (시장으로 학습, 실행 전 고정 11/19)
- 관측 (신호일 t 종가까지): 지수 9개(5·21·63·126거래일 로그수익률, 20·60일 변동성, 50·200일선 대비 거리, 126일 고점 대비 낙폭),
  시장 폭 5개(그날 시가총액 상위 200 코스피 종목의 50일선 위 비율, 200일선 위 비율, 상승 종목 비율 20일 평균,
  20일 수익률의 종목 간 표준편차, 거래대금 추세), 지금 투자 비중 1개. 시장 특성은 폴드 학습 구간의 평균·표준편차로 표준화(±5).
- 학습 환경: 지수(2010-10 ~ 2025)를 '시장 전체'로 보고, 비중 E_t로 시장에 노출됐을 때의 잔고로 보상한다.
  보상 = 100 × [ln(끝 잔고 ÷ 시작 잔고) − λ × Σ max(0, 하루 낙폭 증가)], 낙폭은 자기 잔고 고점 대비.
  비용은 종목 매매와 같게: 수수료 0.015% + 슬리피지 0.10% (한쪽), 줄일 때 매도세 0.20%. 현금 수익 0.
- λ 격자 {0, 0.025, 0.05, 0.1} (위험 성향). 고르지 않고 네 개 모두 평가한다. 판단은 매주(프로젝트 신호일).
- PPO(MlpPolicy, SB3 기본값, M4와 같다), cpu, 스레드 1, 시드 0~4, 50,000스텝, 에피소드 52주(학습 구간 안 무작위 시작,
  시작 비중 무작위). 2,500스텝마다 검증 연도를 결정적 행동으로 운용해 보상 합이 가장 높은 체크포인트를 쓴다.
- Walk-forward: 시험 연도 Y = 2016~2025. 학습 = 2010-10 ~ Y−2년 말, 검증 = Y−1년. 신호일이 속한 해의 폴드 정책을 쓴다.
- 포트폴리오에 적용할 때 '지금 투자 비중' 입력은 직전 신호일에 정한 E다 (첫 신호일은 현금 100%라 0).

평가 (포트폴리오, 2016~2025 연속, C 엔진, M9 포함)
- 비교: 버퍼 + 사전등록 M8 + M9 (지금 구조), 버퍼 + 노출 조절 끔(E1, Step 4~5만 끔, M9 켬), M8·M9 끔, 코스피 200,
  같은 평균 비중의 고정 E (에이전트 5시드의 평균 E와 같은 상수 E, 판단 없이 비중만 낮춘 것).
- 판정 (실행 전 고정, 11/19). λ마다 5시드로 본다.
  (1) 두 마리 토끼: 5시드 평균 최대 낙폭 < 노출 조절 끔, 평균 CAGR ≥ 노출 조절 끔, 둘 다 만족한 시드 4개 이상.
  (2) 지금 M8보다 나음: 평균 CAGR > 사전등록 M8, 평균 최대 낙폭 < 사전등록 M8, CAGR이 앞선 시드 4개 이상.
  (3) 장을 보는 능력: 평균 CAGR > 같은 평균 비중 고정 E, 평균 최대 낙폭 < 그것, CAGR이 앞선 시드 4개 이상.
      λ 4개 중 3개 이상이 (3)을 만족하면 '능력 있음'.
  참고: G7 확인값(M8·M9 끔 대비 낙폭 25% 이상 감소, CAGR 차 −2%p 이내), 국면 성적표(지수 고점·저점 10% 구간의 상승·하락 따라감),
  지수로 잰 판단력(같은 정책을 지수에 적용, 프로젝트 밖 참고).
- 4개 λ를 봤으므로 하나만 넘으면 우연일 수 있다고 같이 적는다. 2016~2025는 이미 본 구간이다. 확인 대상은 (3)을 만족한 λ 중
  같은 평균 비중 고정 대비 초과 CAGR이 가장 큰 것으로 정하고, 처음 보는 데이터(forward test, PM이 허락하면 2026 홀드아웃)로 확인한다.
- 탐색 결과다 (R-15). 결과: mt_summary.csv, mt_runs.csv, mt_judgment.json, runs/mt/eval/ (포트폴리오 NAV·E·positions), 원장.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

from common import HERE, Ledger
import numpy as np
import pandas as pd
from src.backtest import metrics as m5
from src.common import io
from src.portfolio.env_check_report import FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE, kospi200, signal_dates
from src.portfolio.params import PREREG_PATH, load_prereg
from src.portfolio.weekly import signal_dates_between

FEAT_PATH = HERE / "mt_features.parquet"
META_PATH = HERE / "mt_features.json"
RUNS = HERE / "runs" / "mt"
EVAL = RUNS / "eval"
JOBS = HERE / "jobs_mt.txt"
END = "2025-12-31"  # 특성 끝 (2026 홀드아웃은 쓰지 않는다)
LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
LAMS = (0.0, 0.025, 0.05, 0.1)
SEEDS = (0, 1, 2, 3, 4)
YEARS = tuple(range(2016, 2026))
FREQ = "weekly"
STEPS, EVAL_EVERY, EP_LEN = 50_000, 2_500, 52
MARKET = ["r5", "r21", "r63", "r126", "vol20", "vol60", "ma50gap", "ma200gap", "dd126",
          "breadth50", "breadth200", "adv20", "disp20", "valtrend"]
TOP_N = 200
CLIP = 5.0
THETA = 0.10
BENCH = {"m8": ("버퍼 + 사전등록 M8 + M9", "full"), "e1": ("버퍼 + 노출 조절 끔 (M9 켬)", "no_exposure"),
         "off": ("버퍼, M8·M9 끔", "off")}
_cfg = load_prereg(PREREG_PATH)
assert float(_cfg["costs"]["cash_return"]) == 0.0, "부록 1 cash_return이 0이 아니다"
RATE = float(_cfg["costs"]["commission_per_side"]) + float(_cfg["costs"]["slippage_per_side"])
TAX = float(_cfg["costs"]["sell_tax"])
KPI = _cfg["kpi"]
SIG_TAG = ""  # 평가에 쓸 시험 신호: ""이면 A v7(common.SIG_RUN), 아니면 use_signals로 바꾼 run_id


def use_signals(run_id: str | None) -> None:
    """포트폴리오 평가의 시험 신호를 바꾼다 (예: PM 급락 필터 교체본). 학습은 신호를 쓰지 않아 그대로다.
    결과는 runs/mt/eval_{run_id}/와 mt_*_{run_id}.* 로 따로 둔다."""
    global EVAL, SIG_TAG
    if not run_id:
        return
    import common

    if not (io.ROOT / "signals" / f"wf_{run_id}.parquet").is_file():
        raise SystemExit(f"signals/wf_{run_id}.parquet가 없다")
    common.SIG_RUN = run_id
    EVAL = RUNS / f"eval_{run_id}"
    SIG_TAG = f"_{run_id}"


# ---------------------------------------------------------------------------
# 특성
# ---------------------------------------------------------------------------

def _ns(index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(index)).as_unit("ns")


def build_features() -> None:
    import pyarrow.parquet as pq

    full = kospi200(io.read_table("index"))
    full.index = _ns(full.index)
    idx = full[full.index <= pd.Timestamp(END)]
    days = idx.index
    lp = np.log(idx)
    r1 = lp.diff()
    f = pd.DataFrame(index=days)
    for k in (5, 21, 63, 126):
        f[f"r{k}"] = lp - lp.shift(k)
    f["vol20"] = r1.rolling(20, min_periods=20).std() * math.sqrt(252)
    f["vol60"] = r1.rolling(60, min_periods=60).std() * math.sqrt(252)
    f["ma50gap"] = lp - np.log(idx.rolling(50, min_periods=50).mean())
    f["ma200gap"] = lp - np.log(idx.rolling(200, min_periods=200).mean())
    f["dd126"] = lp - np.log(idx.rolling(126, min_periods=126).max())

    t0 = time.time()  # 시장 폭: 그날 시가총액 상위 TOP_N 코스피 종목 (필요한 열만 읽는다)
    tbl = pq.read_table(io.ROOT / "data" / "processed" / "prices.parquet",
                        columns=["date", "ticker", "market", "close", "value", "mcap"], filters=[("market", "==", "KOSPI")])
    px = tbl.to_pandas()
    del tbl
    px["date"] = _ns(px["date"])
    px = px[px["date"] <= pd.Timestamp(END)]
    px["ticker"] = px["ticker"].astype(str)
    close = px.pivot(index="date", columns="ticker", values="close").reindex(days).astype(float)
    close = close.where(close > 0)
    mcap = px.pivot(index="date", columns="ticker", values="mcap").reindex(days).astype(float)
    value = px.pivot(index="date", columns="ticker", values="value").reindex(days).astype(float)
    n_tickers = close.shape[1]
    del px
    top = mcap.rank(axis=1, ascending=False, method="first") <= TOP_N
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()

    def frac(cond: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
        den = (valid & top).sum(axis=1)
        return (cond & valid & top).sum(axis=1) / den.where(den > 0)

    f["breadth50"] = frac(close > ma50, ma50.notna())
    f["breadth200"] = frac(close > ma200, ma200.notna())
    ret = np.log(close).diff()
    f["adv20"] = frac(ret > 0, ret.notna()).rolling(20, min_periods=20).mean()
    r20 = np.log(close / close.shift(20))
    f["disp20"] = r20.where(top).std(axis=1)
    vt = value.where(top).sum(axis=1, min_count=1)
    f["valtrend"] = np.log(vt.rolling(20, min_periods=20).mean() / vt.rolling(120, min_periods=120).mean())
    top_count = top.sum(axis=1)

    valid = f[MARKET].notna().all(axis=1)
    start = valid[valid].index[0]
    f = f.loc[start:].copy()
    nan_filled = int(f[MARKET].isna().sum().sum())
    f[MARKET] = f[MARKET].ffill()

    proj = [pd.Timestamp(d) for d in signal_dates(io.read_table("universe"), full.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)]
    pre = [pd.Timestamp(d) for d in signal_dates_between(full.index, "2010-01-01", FIRST_SIGNAL_DATE)]
    f["weekly"] = f.index.isin(list(set(pre) | set(proj)))
    f["px"] = idx.reindex(f.index)
    f["top_count"] = top_count.reindex(f.index)
    f.reset_index().rename(columns={"index": "date"}).to_parquet(FEAT_PATH, index=False)

    per_year = f.groupby(f.index.year)["weekly"].sum()
    meta = {"feature_start": str(start.date()), "end": str(f.index[-1].date()), "n_days": len(f),
            "n_weekly": int(f["weekly"].sum()), "weekly_per_year_min": int(per_year.min()), "weekly_per_year_max": int(per_year.max()),
            "nan_filled_cells": nan_filled, "top_n": TOP_N, "kospi_tickers_seen": int(n_tickers),
            "top_count_min": int(f["top_count"].min()), "columns": MARKET, "seconds_prices": round(time.time() - t0, 1)}
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    pd.set_option("display.width", 250)
    print(f[MARKET].describe().T[["mean", "std", "min", "max"]].round(4).to_string())
    print("주별 판단일 수 (해마다):", {int(k): int(v) for k, v in per_year.items()})
    print("done")


class Data:
    """특성 파일 (판단일 배열, 지수 종가, 시장 특성)."""

    def __init__(self) -> None:
        f = pd.read_parquet(FEAT_PATH)
        self.days = _ns(f["date"])
        self.px = f["px"].to_numpy(dtype=float)
        self.F = f[MARKET].to_numpy(dtype=float)
        self.dec = np.flatnonzero(f["weekly"].to_numpy(dtype=bool))
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def index_of(self, day) -> int:
        return int(self.days.get_loc(pd.Timestamp(day)))

    def fold(self, y: int):
        yr = self.days.year
        tr, va, te = np.flatnonzero(yr <= y - 2), np.flatnonzero(yr == y - 1), np.flatnonzero(yr == y)
        return (int(tr[0]), int(tr[-1])), (int(va[0]), int(va[-1])), (int(te[0]), int(te[-1]))


def norm_stats(F: np.ndarray, lo: int, hi: int):
    x = F[lo:hi + 1]
    mu, sd = x.mean(axis=0), x.std(axis=0)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def normalize(F: np.ndarray, mu, sd) -> np.ndarray:
    return np.clip((F - np.asarray(mu)) / np.asarray(sd), -CLIP, CLIP)


# ---------------------------------------------------------------------------
# 학습용 장부와 환경 (시장 = 지수)
# ---------------------------------------------------------------------------

class Book:
    """일별 장부: 판단일 종가에 정한 목표 비중을 다음 거래일 종가에 맞춘다 (현금 수익 0). tax: 줄일 때 매도세."""

    __slots__ = ("px", "i", "units", "cash", "nav", "peak", "dd", "turn", "tax")

    def __init__(self, px: np.ndarray, i0: int, w0: float = 0.0, tax: float = 0.0) -> None:
        self.px, self.i, self.tax = px, int(i0), float(tax)
        self.nav, self.peak, self.dd, self.turn = 1.0, 1.0, 0.0, 0.0
        self.units = w0 / px[self.i]
        self.cash = 1.0 - w0

    def weight(self) -> float:
        return self.units * self.px[self.i] / self.nav

    def run(self, j: int, target: float | None, rec: list | None = None):
        """i 종가에서 정한 target을 i+1 종가에 체결하고 j 종가까지 간다. 반환: (ln 잔고 변화, 낙폭 증가 합)."""
        nav0, pen, first = self.nav, 0.0, self.i + 1
        for d in range(first, j + 1):
            p = self.px[d]
            nav = self.units * p + self.cash
            if d == first and target is not None:
                w = self.units * p / nav
                trade = (target - w) * nav
                nav -= abs(trade) * RATE + max(0.0, -trade) * self.tax
                self.units = target * nav / p
                self.cash = nav - self.units * p
                self.turn += abs(target - w)
            self.peak = max(self.peak, nav)
            dd = 1.0 - nav / self.peak
            pen += max(0.0, dd - self.dd)
            self.dd, self.nav = dd, nav
            if rec is not None:
                rec.append((d, nav, self.units * p / nav))
        self.i = int(j)
        return math.log(self.nav / nav0), pen


def obs_vec(Z: np.ndarray, i: int, weight: float) -> np.ndarray:
    return np.append(Z[i], weight).astype(np.float32)


def play(px, dec, start_k: int, stop: int, act, lam: float = 0.0, w0: float = 0.0, finish: bool = False, tax: float = 0.0):
    """dec[start_k]에서 시작해 다음 판단일이 stop 이하인 동안 act(i, book)으로 운용한다.
    finish=True면 마지막 판단일의 결정도 체결하고 stop까지 간다. 반환: (보상 합, book, 기록)."""
    book = Book(px, int(dec[start_k]), w0, tax)
    rec = [(book.i, book.nav, book.weight())]
    total, k = 0.0, start_k
    while k + 1 < len(dec) and dec[k + 1] <= stop:
        lr, pen = book.run(int(dec[k + 1]), act(int(dec[k]), book), rec)
        total += 100.0 * (lr - lam * pen)
        k += 1
    if finish and book.i < stop:
        lr, pen = book.run(stop, act(book.i, book), rec)
        total += 100.0 * (lr - lam * pen)
    return total, book, rec


def make_env_class():
    import gymnasium as gym
    from gymnasium import spaces

    class MarketEnv(gym.Env):
        """학습 환경: 학습 구간 안에서 무작위 시작, EP_LEN 판단 스텝 뒤 끝. 종목 매매와 같은 비용."""

        def __init__(self, px, Z, dec, lam, lo, hi):
            super().__init__()
            self.px, self.Z, self.lam = px, Z, float(lam)
            self.dec = dec[(dec >= lo) & (dec <= hi)]
            if len(self.dec) < EP_LEN + 1:
                raise ValueError(f"학습 구간 판단일 {len(self.dec)}개 < 에피소드 {EP_LEN} + 1")
            self.observation_space = spaces.Box(-np.inf, np.inf, (Z.shape[1] + 1,), np.float32)
            self.action_space = spaces.Discrete(len(LEVELS))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            k0 = int(self.np_random.integers(0, len(self.dec) - EP_LEN))
            w0 = float(LEVELS[int(self.np_random.integers(0, len(LEVELS)))])
            self.k, self.t = k0, 0
            self.book = Book(self.px, int(self.dec[k0]), w0, TAX)
            return obs_vec(self.Z, self.book.i, self.book.weight()), {}

        def step(self, action):
            lr, pen = self.book.run(int(self.dec[self.k + 1]), float(LEVELS[int(action)]))
            self.k += 1
            self.t += 1
            reward = 100.0 * (lr - self.lam * pen)
            truncated = self.t >= EP_LEN or self.k + 1 >= len(self.dec)
            return obs_vec(self.Z, self.book.i, self.book.weight()), float(reward), False, truncated, {}

    return MarketEnv


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def run_dir(y: int, lam: float, seed: int, root: Path = RUNS) -> Path:
    return root / FREQ / f"y{y}" / f"lam{lam:g}" / f"s{seed}"


def train_job(data: Data, y: int, lam: float, seed: int, out: Path, steps: int = STEPS, eval_every: int = EVAL_EVERY) -> dict:
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(1)
    (tr0, tr1), (va0, va1), _ = data.fold(y)
    mu, sd = norm_stats(data.F, tr0, tr1)
    Z = normalize(data.F, mu, sd)
    env = make_env_class()(data.px, Z, data.dec, lam, tr0, tr1)
    vk = int(np.searchsorted(data.dec, va0))
    out.mkdir(parents=True, exist_ok=True)

    def val_score(model):
        def act(i, b):
            return float(LEVELS[int(model.predict(obs_vec(Z, i, b.weight()), deterministic=True)[0])])
        s, book, rec = play(data.px, data.dec, vk, va1, act, lam=lam, tax=TAX)
        return s, book.nav, float(np.mean([r[2] for r in rec[1:]]))

    class ValCB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.records, self.best, self.best_step, self.next = [], -np.inf, None, eval_every

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next:
                self.next += eval_every
                s, nav, w = val_score(self.model)
                self.records.append({"timesteps": self.num_timesteps, "val_score": s, "val_nav": nav, "val_mean_w": w})
                if np.isfinite(s) and s > self.best:
                    self.best, self.best_step = s, self.num_timesteps
                    self.model.save(out / "best.zip")
            return True

    t0 = time.time()
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0)
    cb = ValCB()
    model.learn(total_timesteps=int(steps), callback=cb)
    log = pd.DataFrame(cb.records)
    log.to_csv(out / "eval_log.csv", index=False)
    if cb.best_step is None:
        raise RuntimeError(f"y{y} λ{lam:g} s{seed}: 유한한 검증 점수가 없다")
    (out / "stats.json").write_text(json.dumps({"mu": list(map(float, mu)), "sd": list(map(float, sd)),
                                                "train": [str(data.days[tr0].date()), str(data.days[tr1].date())],
                                                "val": [str(data.days[va0].date()), str(data.days[va1].date())]}),
                                    encoding="utf-8")
    best = log[log.timesteps == cb.best_step].iloc[0]
    res = {"year": y, "lam": lam, "seed": seed, "best_step": int(cb.best_step), "best_val_score": float(cb.best),
           "best_val_nav": float(best.val_nav), "best_val_mean_w": float(best.val_mean_w), "steps": int(steps),
           "seconds": round(time.time() - t0, 1), "sb3": stable_baselines3.__version__, "torch": torch.__version__}
    (out / "done.json").write_text(json.dumps(res), encoding="utf-8")  # 마지막에 쓴다 (없으면 다시 학습)
    return res


def write_jobs() -> None:
    lines = [f"{y} {lam:g} {seed}" for y in YEARS for lam in LAMS for seed in SEEDS]
    JOBS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger = Ledger(HERE / "ledger.jsonl")
    for lam in LAMS:
        ledger.plan(f"mt|λ{lam:g}", "학습형 M8 (장을 보는 비중 에이전트)",
                    f"시장(지수·시장 폭)으로 학습한 주간 E_t(0~100% 5단계), 보상 ln 잔고 − {lam:g} × 낙폭 증가, 종목 매매 비용. "
                    f"버퍼 + M8 Step 1~3 + M9 + C 엔진 포트폴리오에 적용, walk-forward 2016~2025 × 시드 5",
                    "사용자 결정(11/19): 장이 좋으면 비중을 늘리고 나쁘면 줄이는 에이전트로 낙폭과 수익을 함께 잡을 수 있는지. "
                    "학습은 시장, 평가는 프로젝트 포트폴리오")
    print(f"{JOBS}: 작업 {len(lines)}개 (시험 연도 {len(YEARS)} × λ {len(LAMS)} × 시드 {len(SEEDS)})")


def worker(jobs_path: str, slot: int, nslots: int) -> None:
    data = Data()
    jobs = [l.split() for l in Path(jobs_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    mine = [j for k, j in enumerate(jobs) if k % nslots == slot]
    print(f"워커 {slot}/{nslots}: 작업 {len(mine)}개", flush=True)
    for n, (y, lam, seed) in enumerate(mine, 1):
        out = run_dir(int(y), float(lam), int(seed))
        if (out / "done.json").exists():
            continue
        try:
            r = train_job(data, int(y), float(lam), int(seed), out)
            print(f"[{n}/{len(mine)}] " + json.dumps(r, ensure_ascii=False, default=float), flush=True)
        except Exception as e:  # 다음 작업으로 간다. 다시 돌리면 이 작업을 다시 시도한다
            print(f"[{n}/{len(mine)}] " + json.dumps({"job": [y, lam, seed], "error": repr(e)}, ensure_ascii=False), flush=True)
    print("done", flush=True)


def status() -> None:
    jobs = [l.split() for l in JOBS.read_text(encoding="utf-8").splitlines() if l.strip()]
    secs = [json.loads(p.read_text(encoding="utf-8"))["seconds"]
            for p in (run_dir(int(y), float(l), int(s)) / "done.json" for y, l, s in jobs) if p.exists()]
    print(f"학습: {len(secs)}/{len(jobs)} 끝" + (f" (평균 {np.mean(secs):.0f}초/작업)" if secs else ""))
    n_eval = sum((EVAL / f"lam{l:g}_s{s}.json").exists() for l in LAMS for s in SEEDS)
    print(f"포트폴리오 평가: 에이전트 {n_eval}/{len(LAMS) * len(SEEDS)}, 기준 "
          f"{sum((EVAL / f'bench_{k}.json').exists() for k in BENCH)}/{len(BENCH)}")


# ---------------------------------------------------------------------------
# 정책 → 포트폴리오 E_t
# ---------------------------------------------------------------------------

def trained(lam: float, seed: int) -> bool:
    return all((run_dir(y, lam, seed) / "done.json").exists() for y in YEARS)


def load_policy(data: Data, lam: float, seed: int, root: Path = RUNS, years=YEARS):
    from stable_baselines3 import PPO

    pol, Zs = {}, {}
    for y in years:
        d = run_dir(y, lam, seed, root)
        pol[y] = PPO.load(d / "best.zip", device="cpu")
        st = json.loads((d / "stats.json").read_text(encoding="utf-8"))
        Zs[y] = normalize(data.F, st["mu"], st["sd"])
    return pol, Zs


def exposure_path(data: Data, pol: dict, Zs: dict, dates) -> list[float]:
    """신호일마다 그해 폴드 정책으로 E_t를 정한다. '지금 투자 비중' 입력은 직전 신호일의 E (첫날 0)."""
    prev, out = 0.0, []
    for d in dates:
        i = data.index_of(d)
        y = int(data.days[i].year)
        a = int(pol[y].predict(obs_vec(Zs[y], i, prev), deterministic=True)[0])
        prev = float(LEVELS[a])
        out.append(prev)
    return out


def with_exposure(snaps, es):
    return [replace(s, exposure=float(e)) for s, e in zip(snaps, es)]


def portfolio_eval(ctx, snaps, mode: str = "full", es=None):
    """버퍼 규칙 + M8(Step 1~3, E_t) + M9를 C 엔진으로 돌린다. es가 있으면 E_t를 그 값으로 바꾼다."""
    from common import buffer_run

    use = with_exposure(snaps, es) if es is not None else snaps
    with_m9 = mode != "off"
    pos = buffer_run(ctx, use, mode=mode, with_m9=with_m9)
    s = ctx.evaluate(pos, with_m9=with_m9)
    rep = s.pop("_rep")
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and v is not None else v) for k, v in s.items()}, rep.nav, pos


def _save_run(name: str, metrics: dict, nav: pd.Series, pos: pd.DataFrame | None, extra: dict | None = None) -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": nav.index, "nav": nav.to_numpy()}).to_parquet(EVAL / f"nav_{name}.parquet", index=False)
    if pos is not None:
        pos.to_parquet(EVAL / f"pos_{name}.parquet", index=False)
    (EVAL / f"{name}.json").write_text(json.dumps({**metrics, **(extra or {})}, ensure_ascii=False, default=float), encoding="utf-8")


# ---------------------------------------------------------------------------
# 점검
# ---------------------------------------------------------------------------

def control_book(data: Data) -> None:
    """학습 장부(매도세 0으로 둔 판)가 safe_dynamic.simulate와 같은지 (2016~2020, 고정 50%·100%)."""
    import safe_drawdown as sdd
    import safe_dynamic as sdy

    dates, end_day = sdd.window(2016, 2020)
    k0 = int(np.searchsorted(data.dec, data.index_of(dates[0])))
    if data.days[data.dec[k0]] != pd.Timestamp(dates[0]):
        raise RuntimeError(f"첫 판단일이 다르다: {data.days[data.dec[k0]].date()} vs {pd.Timestamp(dates[0]).date()}")
    stop = data.index_of(end_day)
    worst = 0.0
    for name, w in (("고정 50%", 0.5), ("지수 그대로", 1.0)):
        ref = sdy.simulate(name, dates, end_day)
        _, book, rec = play(data.px, data.dec, k0, stop, lambda i, b, w=w: w, finish=True, tax=0.0)
        nav = np.array([r[1] for r in rec])
        diff = abs(nav[-1] - ref["final_nav"])
        worst = max(worst, diff)
        print(f"  대조 {name}: 끝 잔고 {nav[-1]:.6f} vs {ref['final_nav']:.6f}, 차이 {diff:.2e}")
    if worst > 1e-9:
        raise RuntimeError(f"학습 장부 대조 실패 (차이 {worst:.2e})")
    print("대조 1: 학습 장부가 safe_dynamic 장부와 같다")


def smoke() -> None:
    from common import Ctx, buffer_run

    data = Data()
    print("특성:", json.dumps(data.meta, ensure_ascii=False))
    control_book(data)

    tmp = RUNS / "_smoke"
    t0 = time.time()
    r = train_job(data, 2016, 0.025, 0, run_dir(2016, 0.025, 0, tmp), steps=4_096, eval_every=2_048)
    per_step = (time.time() - t0) / 4_096
    print(f"짧은 학습 (폴드 2016, λ 0.025, 시드 0, 4,096스텝): {json.dumps(r, ensure_ascii=False, default=float)}")

    t1 = time.time()
    ctx = Ctx((2016,))
    snaps = ctx.snaps()
    t_load = time.time() - t1
    t2 = time.time()
    a = buffer_run(ctx, snaps, mode="full")
    b = buffer_run(ctx, with_exposure(snaps, [s.exposure for s in snaps]), mode="full")
    if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
        raise RuntimeError("E_t를 같은 값으로 바꿔 넣었는데 positions가 달라졌다")
    print("대조 2: 스냅샷 E_t 교체 경로가 공식 버퍼 실행과 같다 (2016)")
    m8, _, _ = portfolio_eval(ctx, snaps, "full")
    pol, Zs = load_policy(data, 0.025, 0, tmp, years=(2016,))
    es = exposure_path(data, pol, Zs, ctx.dates)
    ag, _, _ = portfolio_eval(ctx, snaps, "full", es)
    t_run = (time.time() - t2) / 3  # buffer_run 4번 + evaluate 2번 ≈ '실행 + 평가' 3번 몫
    print(f"2016 포트폴리오 (점검용, 4천 스텝짜리 정책): 사전등록 M8 CAGR {m8['cagr']:+.2%}, 낙폭 {m8['mdd']:.1%} / "
          f"에이전트 CAGR {ag['cagr']:+.2%}, 낙폭 {ag['mdd']:.1%}, 평균 E {np.mean(es):.0%}")
    train_total = per_step * STEPS * len(YEARS) * len(LAMS) * len(SEEDS)
    n_runs = len(LAMS) * len(SEEDS) + len(BENCH) + len(LAMS)
    eval_total = n_runs * t_run * len(YEARS) + t_load * 2
    print(f"예상 시간: 학습 워커 4개면 약 {train_total / 3600 / 4:.1f}시간 (1개면 {train_total / 3600:.1f}시간), "
          f"포트폴리오 평가 약 {eval_total / 60:.0f}분 (1년 실행으로 잰 대략값)")
    print("smoke ok")


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def yearly(nav: pd.Series) -> dict:
    last = nav.groupby(nav.index.year).last()
    return {int(y): float(v) for y, v in (last / last.shift(1).fillna(1.0) - 1.0).items()}


def capture(nav: pd.Series, px: pd.Series, segs: list) -> tuple[float, float]:
    up_m = up_i = dn_m = dn_i = 0.0
    for sg in segs:
        if sg["partial"]:
            continue
        a, b = sg["start"], sg["end"]
        rm, ri = math.log(nav[b] / nav[a]), math.log(px[b] / px[a])
        if sg["kind"] == "상승":
            up_m, up_i = up_m + rm, up_i + ri
        else:
            dn_m, dn_i = dn_m + rm, dn_i + ri
    return up_m / up_i, dn_m / dn_i


def evaluate(slot: int | None = None, nslots: int | None = None) -> None:
    """포트폴리오 평가. slot을 주면 그 몫의 에이전트(와 slot 0이면 기준)만 돌리고 끝낸다. 없으면 남은 것을 다 돌리고 보고한다."""
    from common import ALL_YEARS, Ctx
    from src.portfolio.buffer_wf_report import benchmark_summary

    import torch
    torch.set_num_threads(1)
    data = Data()
    pairs = [(l, s) for l in LAMS for s in SEEDS]
    mine = pairs if slot is None else [p for k, p in enumerate(pairs) if k % nslots == slot]
    todo = [p for p in mine if trained(*p) and not (EVAL / f"lam{p[0]:g}_s{p[1]}.json").exists()]
    untrained = [p for p in mine if not trained(*p)]
    need_bench = (slot in (None, 0)) and any(not (EVAL / f"bench_{k}.json").exists() for k in BENCH)
    if untrained:
        print(f"학습이 덜 끝난 (λ, 시드) {len(untrained)}개는 건너뛴다: {untrained[:5]}")
    ctx = snaps = None

    def ensure_ctx():
        nonlocal ctx, snaps
        if ctx is None:
            t = time.time()
            ctx = Ctx(ALL_YEARS)
            snaps = ctx.snaps()
            print(f"포트폴리오 입력 읽음 ({time.time() - t:.0f}초)", flush=True)

    if need_bench:
        ensure_ctx()
        for key, (label, mode) in BENCH.items():
            if (EVAL / f"bench_{key}.json").exists():
                continue
            t = time.time()
            m, nav, pos = portfolio_eval(ctx, snaps, mode)
            _save_run(f"bench_{key}", {**m, "label": label}, nav, pos, {"e_mean": float(pos.drop_duplicates("date")["exposure"].mean())
                                                                        if "exposure" in pos else None})
            print(f"  기준 {label}: CAGR {m['cagr']:+.2%}, 낙폭 {m['mdd']:.1%} ({time.time() - t:.0f}초)", flush=True)
            if key == "m8":
                k = benchmark_summary(ctx.inputs.bm_close, nav.index)
                (EVAL / "bench_kospi200.json").write_text(json.dumps({"cagr": float(k["cagr"]), "mdd": float(k["mdd"]),
                                                                      "label": "코스피 200 (비교 기준, 사지 않음)"}), encoding="utf-8")
    for lam, seed in todo:
        ensure_ctx()
        t = time.time()
        pol, Zs = load_policy(data, lam, seed)
        es = exposure_path(data, pol, Zs, ctx.dates)
        m, nav, pos = portfolio_eval(ctx, snaps, "full", es)
        _save_run(f"lam{lam:g}_s{seed}", m, nav, pos, {"lam": lam, "seed": seed, "e_path": es, "dates": list(ctx.dates)})
        print(f"  에이전트 λ{lam:g} 시드 {seed}: CAGR {m['cagr']:+.2%}, 낙폭 {m['mdd']:.1%}, 평균 E {np.mean(es):.0%} "
              f"({time.time() - t:.0f}초)", flush=True)
    if slot is not None:
        print("done (이 몫만 끝냈다. 모든 몫이 끝나면 인자 없이 evaluate를 한 번 더 돌려 보고서를 만든다)")
        return

    # 같은 평균 비중의 고정 E (λ마다 5시드 평균 E)
    agents = {}
    for lam, seed in pairs:
        p = EVAL / f"lam{lam:g}_s{seed}.json"
        if p.exists():
            agents[(lam, seed)] = json.loads(p.read_text(encoding="utf-8"))
    for lam in LAMS:
        ss = [agents[(lam, s)] for s in SEEDS if (lam, s) in agents]
        if len(ss) < len(SEEDS) or (EVAL / f"equal_lam{lam:g}.json").exists():
            continue
        ensure_ctx()
        w = float(np.mean([np.mean(a["e_path"]) for a in ss]))
        t = time.time()
        m, nav, pos = portfolio_eval(ctx, snaps, "full", [w] * len(snaps))
        _save_run(f"equal_lam{lam:g}", m, nav, None, {"w": w})
        print(f"  같은 평균 비중 고정 E {w:.1%} (λ{lam:g}): CAGR {m['cagr']:+.2%}, 낙폭 {m['mdd']:.1%} ({time.time() - t:.0f}초)", flush=True)
    report(data, agents)


def report(data: Data, agents: dict) -> None:
    import safe_phases as sph

    def load(name):
        p = EVAL / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def nav_of(name):
        f = pd.read_parquet(EVAL / f"nav_{name}.parquet")
        return pd.Series(f["nav"].to_numpy(), index=_ns(f["date"]))

    bench = {k: load(f"bench_{k}") for k in BENCH}
    if any(v is None for v in bench.values()):
        raise SystemExit("기준 실행이 없다. evaluate를 먼저 끝까지 돌린다")
    kospi = load("bench_kospi200")
    m8, e1, off = bench["m8"], bench["e1"], bench["off"]
    pxs = pd.Series(data.px, index=data.days)
    days = nav_of("bench_m8").index
    segs = sph.phases(pxs.reindex(days), THETA)
    caps = {k: capture(nav_of(f"bench_{k}"), pxs, segs) for k in BENCH}

    rows, summ, judg = [], [], {}
    ledger = Ledger(HERE / "ledger.jsonl")
    for (lam, seed), a in agents.items():
        up, dn = capture(nav_of(f"lam{lam:g}_s{seed}"), pxs, segs)
        rows.append({"lam": lam, "seed": seed, **{k: a.get(k) for k in ("cagr", "mdd", "ir", "annual_excess", "annual_turnover",
                                                                         "mean_cash", "final_nav")},
                     "mean_e": float(np.mean(a["e_path"])), "up_cap": up, "dn_cap": dn,
                     **{f"y{y}": v for y, v in yearly(nav_of(f"lam{lam:g}_s{seed}")).items()}})
    rd = pd.DataFrame(rows)
    rd.to_csv(HERE / f"mt_runs{SIG_TAG}.csv", index=False)

    for k, (label, _) in BENCH.items():
        b = bench[k]
        summ.append({"model": label, "kind": "기준", **{c: b.get(c) for c in ("cagr", "mdd", "ir", "annual_turnover", "mean_cash", "final_nav")},
                     "up_cap": caps[k][0], "dn_cap": caps[k][1]})
    summ.append({"model": kospi["label"], "kind": "기준", "cagr": kospi["cagr"], "mdd": kospi["mdd"]})
    for lam in LAMS:
        g = rd[rd.lam == lam] if len(rd) else rd
        eq = load(f"equal_lam{lam:g}")
        if len(g) < len(SEEDS) or eq is None:
            continue
        mean = {c: float(g[c].mean()) for c in ("cagr", "mdd", "ir", "annual_turnover", "mean_cash", "final_nav", "mean_e", "up_cap", "dn_cap")}
        both = int(((g.cagr >= e1["cagr"]) & (g.mdd < e1["mdd"])).sum())
        above_m8 = int((g.cagr > m8["cagr"]).sum())
        above_eq = int((g.cagr > eq["cagr"]).sum())
        j = {
            "(1) 두 마리 토끼 (노출 조절 끔 대비)": bool(mean["mdd"] < e1["mdd"] and mean["cagr"] >= e1["cagr"] and both >= 4),
            "(2) 지금 M8보다 나음": bool(mean["cagr"] > m8["cagr"] and mean["mdd"] < m8["mdd"] and above_m8 >= 4),
            "(3) 장을 보는 능력 (같은 평균 비중 고정 대비)": bool(mean["cagr"] > eq["cagr"] and mean["mdd"] < eq["mdd"] and above_eq >= 4),
        }
        g7 = {"mdd_reduction_rel": (off["mdd"] - mean["mdd"]) / off["mdd"], "cagr_gap": mean["cagr"] - off["cagr"]}
        g7["pass"] = bool(g7["mdd_reduction_rel"] >= float(KPI["G7"]["min_mdd_reduction_rel"])
                          and g7["cagr_gap"] >= -float(KPI["G7"]["max_cagr_giveup"]))
        name = f"에이전트 λ{lam:g}"
        judg[name] = {"lam": lam, **j, "seeds_both_vs_e1": both, "seeds_cagr_above_m8": above_m8, "seeds_cagr_above_equal": above_eq,
                      "equal_w": eq["w"], "equal_cagr": eq["cagr"], "equal_mdd": eq["mdd"], "excess_vs_equal": mean["cagr"] - eq["cagr"],
                      "G7_참고": g7}
        summ.append({"model": name, "kind": "에이전트", **mean, "cagr_min": float(g.cagr.min()), "cagr_max": float(g.cagr.max()),
                     "mdd_min": float(g.mdd.min()), "mdd_max": float(g.mdd.max()),
                     "equal_w": eq["w"], "equal_cagr": eq["cagr"], "equal_mdd": eq["mdd"], **j})
        ledger.result(f"mt|λ{lam:g}{SIG_TAG}", **{k: v for k, v in judg[name].items() if k != "lam"},
                      **{f"mean_{k}": v for k, v in mean.items()})
    sm = pd.DataFrame(summ)
    sm.to_csv(HERE / f"mt_summary{SIG_TAG}.csv", index=False)

    n3 = sum(v["(3) 장을 보는 능력 (같은 평균 비중 고정 대비)"] for v in judg.values())
    ability = {"λ_passing_(3)": n3, "λ_total": len(judg), "능력 있음": bool(len(judg) == len(LAMS) and n3 >= 3)}
    cands = [(k, v["excess_vs_equal"]) for k, v in judg.items() if v["(3) 장을 보는 능력 (같은 평균 비중 고정 대비)"]]
    pick = max(cands, key=lambda x: x[1])[0] if cands else None

    # 참고: 같은 정책을 지수에 적용 (프로젝트 밖, 판단력 진단용)
    ref = []
    k0 = int(np.searchsorted(data.dec, data.index_of(days[0]) - 1))
    stop = data.index_of(days[-1])
    for lam in LAMS:
        vals = []
        for seed in SEEDS:
            if not trained(lam, seed):
                continue
            pol, Zs = load_policy(data, lam, seed)

            def act(i, b, pol=pol, Zs=Zs):
                y = int(data.days[i].year)
                return float(LEVELS[int(pol[y].predict(obs_vec(Zs[y], i, b.weight()), deterministic=True)[0])])

            _, book, rec = play(data.px, data.dec, k0, stop, act, finish=True, tax=TAX)
            nav = np.array([r[1] for r in rec])
            w = np.array([r[2] for r in rec[1:]])
            vals.append((float(m5.cagr(np.r_[1.0, nav[2:]], trading_days=len(nav) - 2)), float(m5.max_drawdown(np.r_[1.0, nav])),
                         float(w.mean())))
        if len(vals) == len(SEEDS):
            wbar = float(np.mean([v[2] for v in vals]))
            _, _, rq = play(data.px, data.dec, k0, stop, lambda i, b: wbar, finish=True, tax=TAX)
            nq = np.array([r[1] for r in rq])
            ref.append({"lam": lam, "agent_cagr": float(np.mean([v[0] for v in vals])), "agent_mdd": float(np.mean([v[1] for v in vals])),
                        "mean_w": wbar, "fixed_cagr": float(m5.cagr(np.r_[1.0, nq[2:]], trading_days=len(nq) - 2)),
                        "fixed_mdd": float(m5.max_drawdown(np.r_[1.0, nq]))})
    import common

    (HERE / f"mt_judgment{SIG_TAG}.json").write_text(json.dumps({
        "signals": common.SIG_RUN, "period": [str(days[0].date()), str(days[-1].date())],
        "bench": {k: {c: bench[k].get(c) for c in ("cagr", "mdd", "ir")} for k in BENCH},
        "kospi200": kospi, "configs": judg, "ability": ability, "confirm_candidate": pick, "index_reference": ref},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ledger.result(f"mt|판정{SIG_TAG}", signals=common.SIG_RUN, ability=ability, confirm_candidate=pick)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(f"\n[포트폴리오 {days[0].date()} ~ {days[-1].date()}, 시험 신호 {common.SIG_RUN}] 버퍼 규칙 종목 + C 엔진 + M9 "
          f"(에이전트는 5시드 평균)")
    show = sm.copy()
    for c in [c for c in show.columns if c.startswith(("cagr", "mdd", "mean_cash", "mean_e", "equal_", "up_cap", "dn_cap"))]:
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(2)
    for c in ("ir", "annual_turnover", "final_nav"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(3)
    print(show.to_string(index=False))

    if len(rd):
        yc = [c for c in rd.columns if c.startswith("y") and c[1:].isdigit()]
        yt = (rd.groupby("lam")[yc].mean() * 100).round(1)
        yt.index = [f"에이전트 λ{l:g}" for l in yt.index]
        bt = pd.DataFrame([{f"y{y}": v * 100 for y, v in yearly(nav_of(f"bench_{k}")).items()} for k in BENCH],
                          index=[BENCH[k][0] for k in BENCH]).round(1)
        print("\n[연도별 수익률 %]")
        print(pd.concat([bt, yt])[yc].to_string())

    print("\n[판정] (실행 전 고정)")
    for k, v in judg.items():
        marks = ", ".join(f"{q} {'만족' if v[q] else '미달'}" for q in v if q.startswith("("))
        print(f"  {k}: {marks}")
        print(f"      노출 조절 끔 대비 둘 다 만족 시드 {v['seeds_both_vs_e1']}/5, M8보다 CAGR 높은 시드 {v['seeds_cagr_above_m8']}/5, "
              f"같은 평균 비중 {v['equal_w']:.0%} 고정보다 CAGR 높은 시드 {v['seeds_cagr_above_equal']}/5 "
              f"(초과 {v['excess_vs_equal'] * 100:+.2f}%p), G7 참고 {'넘음' if v['G7_참고']['pass'] else '미달'} "
              f"(낙폭 감소 {v['G7_참고']['mdd_reduction_rel']:.0%}, CAGR 차 {v['G7_참고']['cagr_gap'] * 100:+.2f}%p)")
    print(f"  장을 보는 능력: (3) 만족 λ {ability['λ_passing_(3)']}/{ability['λ_total']} → {'있음' if ability['능력 있음'] else '없음'}")
    print(f"  확인 대상 (처음 보는 데이터용): {pick or '없음'}")
    if ref:
        print("\n[참고: 같은 정책을 지수에 적용 (프로젝트 밖, 시장 판단력 진단)]")
        rt = pd.DataFrame(ref)
        for c in ("agent_cagr", "agent_mdd", "mean_w", "fixed_cagr", "fixed_mdd"):
            rt[c] = (rt[c] * 100).round(2)
        print(rt.to_string(index=False))
    print("done")


if __name__ == "__main__":
    if "--signals" in sys.argv:  # evaluate·status: 평가 시험 신호 바꾸기 (예: --signals 20261119_PM_pair32_B_test)
        k = sys.argv.index("--signals")
        use_signals(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "features":
        build_features()
    elif cmd == "smoke":
        smoke()
    elif cmd == "jobs":
        write_jobs()
    elif cmd == "worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "status":
        status()
    elif cmd == "evaluate":
        if len(sys.argv) >= 4:
            evaluate(int(sys.argv[2]), int(sys.argv[3]))
        else:
            evaluate()
    else:
        raise SystemExit("사용법: mt_agent.py features | smoke | jobs | worker <작업 파일> <슬롯> <슬롯 수> | status | evaluate [슬롯 슬롯 수]")
