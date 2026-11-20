r"""T7 비중 에이전트 v6: v4 + 현금 = CD 91일물 + 보상에 현금 보유 비용 (11/20 사용자 제안 "장기간 현금 보유는 패널티",
"v6과 v7로 둘 다 설계", "현금은 안정자산을 수익으로").

실행 (저장소 루트, PowerShell, 사용자가 돌린다, 창 1개, 30~40분):
  .\.venv\Scripts\python.exe -I explore\b09\run.py t7_agent6.py run      (짧은 점검: ... t7_agent6.py smoke)
  - 시작할 때 이전 v6 결과(runs/t7f/)를 모두 지운다.

v4와 다른 점 (실행 전 고정 11/20)
- 보상 = 100 × ln(잔고 변화, 현금 이자 포함) − 0.01 × 100 × 낙폭 증가 − κ × 100 × (1 − E) × 구간 거래일 ÷ 252.
  현금을 들고 있는 동안 연 κ×100%의 비용을 학습 보상에서만 뺀다 (평가에는 없다). 장세와 상관없이 투자를 늘리는 압력이다.
- κ ∈ {0.0255, 0.0295, 0.0335}. 현금 이자를 넣고 학습 구간(2011~2020)만으로 정했다 (11/20 B, 실행 전):
  고정 E를 1년 에피소드로 돌린 평균 보상이 가장 큰 E가 0.6·0.7·0.8이 되는 κ 구간(0.0235~0.0275, 0.0275~0.0315, 0.0315~0.0355)의
  가운데 값. 연 2.6~3.4%로 현금 이자(학습 구간 평균 약 2%)보다 크다.
- κ는 검증 2021~2023 앙상블 점수로 고른다 (v4의 λ 고르기와 같음).
- 현금 수익 = 안정자산 CD 91일물 (11/20 사용자 결정: "현금은 안정자산을 수익으로 책정", explore/b09/cash_yield.py).
  - 학습·검증 장부: 현금이 매일 CD 91일물 이자를 받는다 (세전, 보수 없음, 달력일로 쌓임).
  - C 엔진은 현금에 수익을 주지 않으므로, 엔진 일별 NAV를 엔진 원장의 일별 현금 비중으로 보정한다
    (보정 일간 수익 = 엔진 일간 수익 + 전날 현금 비중 × 그날 현금 수익). 비교 대상(T7-80-C53, 같은 평균 비중 고정)도 같게 보정하고,
    지수 혼합(46.6% + 현금)도 현금에 CD 91일물을 준다. 대조는 보정 전 값으로 한다 (E 0.466107 = 25절 T7-80-C53, 1e-12).
  - 현금 이자를 넣으면 학습 구간(2011~2020)에서는 λ와 상관없이 정적 최적 E가 0(전부 현금)이다:
    바구니 1년 평균 로그수익 +0.2% vs 현금 +2.0% (11/20 B). 그래서 κ를 현금 이자 기준으로 다시 정했다.
    정적 최적 E 0.6~0.8은 데이터에서 배운 값이 아니라 설계로 정한 기준이고, 에이전트는 그 주변 타이밍을 배운다.
- v4와 같은 것: 바구니·특성(학습 2011~2020), 관측, 행동 11단계, 학습 100,000스텝, 5,000스텝마다 검증 2021~2023 점수
  CAGR ÷ max(MDD, 2%) 최고 체크포인트, 시드 5개 앙상블, C 엔진 평가, 판정(테스트 2024~2025).
- 낙폭 벌점은 λ 0.01로 고정한다.
- 주의: 테스트 구간(2024~2025)에 2025 급등이 있다는 것을 알고 '상승장에서 더 과감하게'를 목표로 만든 설계다.
  결과는 탐색으로만 쓰고, 깨끗한 확인은 2026 홀드아웃이다.
- 탐색 결과다 (R-15). 결과: t7f_runs.csv, t7f_judgment.json, runs/t7f/, 원장 t7f|*.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

from bench_kospi import KOSPI_PATH
from common import ALL_YEARS, HERE, Ctx, Ledger, buffer_run
import exitrule as ex
import numpy as np
import pandas as pd
from src.common import io
from src.portfolio.env_check_report import FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE, kospi200, signal_dates
from src.portfolio.params import PREREG_PATH, load_prereg
from turnover import ledger_has, smooth_signals
from cash_yield import CashYield, cash_fraction
from src.portfolio.buffer_wf_report import replay, summary

SAMECASH = HERE / "runs" / "samecash"  # T7 보유 80 E = 1과 T7-80-C53의 엔진 NAV (25절 규칙 백테스트)
RUNS = HERE / "runs" / "t7f"
EVAL = RUNS / "eval"
EXT_PATH = RUNS / "basket.parquet"
PX_PATH = RUNS / "px.parquet"
FEAT_PATH = RUNS / "features.parquet"
MT_FEAT = HERE / "mt_features.parquet"

E_C53 = 1.0 - 0.533893  # 2016~2020 운용 비중과 비교 대상 T7-80-C53 (학습·보상에는 쓰지 않는다)
LEVELS = np.round(np.arange(11) / 10.0, 1)
LAMBDAS = (0.0255, 0.0295, 0.0335)  # κ: 현금 100%를 1년 들 때 연 κ×100%의 학습용 비용. 현금 이자 넣고 학습 구간만으로: 정적 최적 E 0.6·0.7·0.8
DD_LAM = 0.01  # 낙폭 벌점
SEEDS = (0, 1, 2, 3, 4)
STEPS, EVAL_EVERY, EP_LEN = 100_000, 5_000, 26
MDD_FLOOR = 0.02
TRAIN_END, VAL, TEST = 2020, (2021, 2023), (2024, 2025)
ACTIVE = tuple(range(VAL[0], TEST[1] + 1))  # 에이전트가 운용하는 해 (2021~2025)
LIVE_START = pd.Timestamp("2016-01-08")
CLIP = 5.0
N, BAND, EVERY = 80, 60, 2
IDX = ["r5", "r21", "r63", "r126", "vol20", "vol60", "ma50gap", "ma200gap", "dd126",
       "breadth50", "breadth200", "adv20", "disp20", "valtrend"]
BKT = ["b_r5", "b_r21", "b_r63", "b_vol20", "b_vol60", "b_dd"]
FEATS = IDX + BKT
PERIODS = {"val": VAL, "test": TEST, "all": (VAL[0], TEST[1])}
TRACK = "T7 비중 에이전트 v6 (현금 = CD 91일물, 보상에 현금 보유 비용)"

_cfg = load_prereg(PREREG_PATH)
assert float(_cfg["costs"]["cash_return"]) == 0.0, "부록 1 cash_return이 0이 아니다"
RATE = float(_cfg["costs"]["commission_per_side"]) + float(_cfg["costs"]["slippage_per_side"])
TAX = float(_cfg["costs"]["sell_tax"])


def lam_id(lam: float) -> str:
    return f"k{lam:g}"


def _ns(x) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(x)).as_unit("ns")


def navp(path: Path) -> pd.Series:
    f = pd.read_parquet(path)
    return pd.Series(f["nav"].to_numpy(dtype=float), index=_ns(f["date"]))


def _cm(v: pd.Series) -> tuple[float, float]:
    v = v.dropna()
    v = v / v.iloc[0]
    n = len(v) - 1
    return float(v.iloc[-1] ** (252 / n) - 1), float((1 - v / v.cummax()).max())


# ---------------------------------------------------------------------------
# 바구니와 특성 (v3와 같은 계산)
# ---------------------------------------------------------------------------

def simulate_ew(days: pd.DatetimeIndex, R: np.ndarray, col: dict, decs: list):
    """리밸런싱 신호일의 U_t 종목을 같은 비중으로, 다음 거래일 종가에 맞춘다 (첫 체결일 1, 그 전 NaN).
    교체 비용 = 편도 회전율 × (2 × 한쪽 비용 + 매도세)."""
    n = len(days)
    fill = {}
    for d, members in decs:
        k = int(days.searchsorted(d, side="right"))
        if k < n:
            fill[k] = members
    nav = np.full(n, np.nan)
    idx, w = np.zeros(0, dtype=np.int64), np.zeros(0)
    v, started = 1.0, False
    for k in range(n):
        if len(idx):
            r = R[k, idx]
            g = float(w @ r)
            v *= 1.0 + g
            w = w * (1.0 + r) / (1.0 + g)
        if k in fill:
            new = [t for t in fill[k] if t in col]
            nidx = np.array([col[t] for t in new], dtype=np.int64)
            nw = np.full(len(nidx), 1.0 / len(nidx)) if len(nidx) else np.zeros(0)
            old = dict(zip(idx.tolist(), w.tolist()))
            tgt = dict(zip(nidx.tolist(), nw.tolist()))
            tau = 0.5 * sum(abs(tgt.get(j, 0.0) - old.get(j, 0.0)) for j in set(old) | set(tgt))
            if started:
                v *= 1.0 - tau * (2.0 * RATE + TAX)
            idx, w, started = nidx, nw, True
        if started:
            nav[k] = v
    return nav


def build_data() -> None:
    """바구니(2011~2015 U_t 동일가중 + 2016~ T7 보유 80 E = 1 엔진 NAV)와 리밸런싱일 특성을 처음부터 만든다."""
    import pyarrow.parquet as pq

    t0 = time.time()
    bm = kospi200(io.read_table("index"))
    u = io.read_table("universe")
    cal = [d for d in signal_dates(u, bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE) if 2011 <= int(d[:4]) <= 2025]
    days = _ns(bm.index)
    days = days[(days >= pd.Timestamp(cal[0])) & (days <= pd.Timestamp("2025-12-30"))]
    u = u[u["in_universe"].astype(bool)][["date", "ticker"]].copy()
    u["date"], u["ticker"] = _ns(u["date"]), u["ticker"].astype(str)
    mem = {pd.Timestamp(d): g["ticker"].to_numpy() for d, g in u.groupby("date")}
    tbl = pq.read_table(io.ROOT / "data" / "processed" / "prices.parquet", columns=["date", "ticker", "close"],
                        filters=[("ticker", "in", sorted(set(u["ticker"]))), ("date", ">=", days[0]), ("date", "<=", days[-1])])
    px = tbl.to_pandas()
    del tbl
    px["date"], px["ticker"] = _ns(px["date"]), px["ticker"].astype(str)
    close = px.pivot(index="date", columns="ticker", values="close").reindex(days).astype(float)
    del px
    close = close.where(close > 0).ffill()
    R = close.pct_change(fill_method=None).fillna(0.0).to_numpy()
    col = {t: k for k, t in enumerate(close.columns)}
    cal_old = [pd.Timestamp(d) for d in cal if int(d[:4]) <= 2015]
    cal_new = [d for d in cal if int(d[:4]) >= 2016]
    dec_old = [(d, mem[d]) for d in cal_old[0::2]]
    ew = pd.Series(simulate_ew(days, R, col, dec_old), index=days)

    e1 = navp(SAMECASH / "nav_e1.parquet")
    j0 = e1.index[0]  # 2016-01-11
    first_fill = ew.dropna().index[0]
    bpx = pd.concat([pd.Series([1.0], index=_ns([days[0]])), ew[(ew.index >= first_fill) & (ew.index < j0)], float(ew[j0]) * e1])
    if bpx.index.has_duplicates or not bpx.index.is_monotonic_increasing or bpx.isna().any():
        raise RuntimeError("이어 붙인 바구니 잔고가 이상하다")
    reb_new = cal_new[0::2]
    dec_dates = _ns([str(d.date()) for d, _ in dec_old] + reb_new)
    dec = bpx.index.get_indexer(dec_dates)
    if (dec < 0).any():
        raise RuntimeError("리밸런싱일이 바구니 달력에 없다")
    s = np.log(bpx)
    r1 = s.diff()
    bf = pd.DataFrame(index=bpx.index)
    for k in (5, 21, 63):
        bf[f"b_r{k}"] = s - s.shift(k)
    bf["b_vol20"] = r1.rolling(20, min_periods=20).std() * math.sqrt(252)
    bf["b_vol60"] = r1.rolling(60, min_periods=60).std() * math.sqrt(252)
    bf["b_dd"] = s - s.cummax()
    mt = pd.read_parquet(MT_FEAT)
    mt.index = _ns(mt["date"])
    feats = pd.concat([mt.reindex(dec_dates)[IDX].reset_index(drop=True), bf.reindex(dec_dates)[BKT].reset_index(drop=True)], axis=1)
    out = pd.concat([pd.DataFrame({"date": dec_dates, "k": dec, "year": dec_dates.year}), feats], axis=1)
    out["ok"] = np.isfinite(out[FEATS].to_numpy(dtype=float)).all(axis=1)
    okv = out["ok"].to_numpy()
    first_ok = int(np.argmax(okv))
    if not okv[first_ok:].all():
        raise RuntimeError("특성이 빠진 리밸런싱일이 처음 몇 개 뒤에도 있다")
    RUNS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": days, "ew": ew.to_numpy()}).to_parquet(EXT_PATH, index=False)
    pd.DataFrame({"date": bpx.index, "px": bpx.to_numpy()}).to_parquet(PX_PATH, index=False)
    out.to_parquet(FEAT_PATH, index=False)

    # 점검: 2016~ 장부 E 고정 vs 엔진 (25절)
    c53 = navp(SAMECASH / "nav_cash.parquet")
    live = out[out["date"] >= LIVE_START]
    if len(live) != len(reb_new) or live["date"].iloc[0] != LIVE_START:
        raise RuntimeError("2016년 뒤 리밸런싱일이 평가 신호일과 다르다")
    for e, ref, name in ((E_C53, c53, "E 0.466107 (T7-80-C53)"), (1.0, e1, "E 1")):
        b = Book(bpx.to_numpy(), int(live["k"].iloc[0]), 0.0)
        rec = []
        dl = live["k"].to_numpy(dtype=np.int64)
        for k in range(len(dl)):
            j = int(dl[k + 1]) if k + 1 < len(dl) else len(bpx) - 1
            b.run(j, e, rec)
        a = pd.Series([v for _, v in rec], index=bpx.index[[d for d, _ in rec]])
        n = len(a) - 1
        ca, ma = float(a.iloc[-1] ** (252 / n) - 1), float((1 - a / a.cummax()).max())
        rr = ref[ref.index >= j0]
        cr, mr = float(rr.iloc[-1] ** (252 / (len(rr) - 1)) - 1), float((1 - rr / rr.cummax()).max())
        print(f"  점검 2016~ 장부 {name}: CAGR {ca:+.4f}·MDD {ma:.4f} | 엔진 {cr:+.4f}·{mr:.4f}")
        if abs(ca - cr) > 0.002 or abs(ma - mr) > 0.005:
            raise RuntimeError(f"장부({name})가 엔진과 너무 다르다")
    co, mo = _cm(ew[ew.index <= pd.Timestamp("2015-12-30")])
    n_train = int((out["ok"] & (out["year"] <= TRAIN_END)).sum())
    print(f"바구니·특성 ({time.time() - t0:.0f}초): 2011~2015 동일가중 CAGR {co:+.4f}·MDD {mo:.4f}, "
          f"학습 리밸런싱일 {n_train}개 ({out['date'][out['ok']].iloc[0].date()} ~ {out['date'][out['year'] <= TRAIN_END].iloc[-1].date()})",
          flush=True)


class Data:
    def __init__(self) -> None:
        w = pd.read_parquet(PX_PATH)
        self.days = _ns(w["date"])
        self.px = w["px"].to_numpy(dtype=float)
        self.rc = CashYield().daily(self.days)  # 현금 = CD 91일물
        f = pd.read_parquet(FEAT_PATH)
        f["date"] = _ns(f["date"])
        self.f = f.sort_values("date").reset_index(drop=True)
        self.lv = self.f[self.f["date"] >= LIVE_START].reset_index(drop=True)  # 평가 리밸런싱일 2016~

    def train_rows(self) -> pd.DataFrame:
        return self.f[self.f["ok"] & (self.f["year"] <= TRAIN_END)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 장부와 학습 환경
# ---------------------------------------------------------------------------

class Book:
    """바구니(잔고 px) + 현금. 신호일 종가에 정한 비중을 다음 거래일 종가에 맞춘다 (줄일 때 매도세).
    rc가 있으면 현금이 매일 rc[d]만큼 불어난다 (CD 91일물, cash_yield.CashYield.daily)."""

    __slots__ = ("px", "rc", "i", "units", "cash", "nav", "peak")

    def __init__(self, px: np.ndarray, i0: int, w0: float = 0.0, rc: np.ndarray | None = None) -> None:
        self.px, self.rc, self.i = px, rc, int(i0)
        self.nav, self.peak = 1.0, 1.0
        self.units = w0 / px[self.i]
        self.cash = 1.0 - w0

    def weight(self) -> float:
        return self.units * self.px[self.i] / self.nav

    def dd(self) -> float:
        return 1.0 - self.nav / self.peak

    def run(self, j: int, target: float, rec: list | None = None) -> tuple[float, float]:
        """i 종가에서 정한 target을 i+1 종가에 체결하고 j 종가까지 간다. 반환: (ln 잔고 변화, 낙폭 증가)."""
        nav0, dd0, first = self.nav, self.dd(), self.i + 1
        for d in range(first, j + 1):
            if self.rc is not None:
                self.cash *= 1.0 + self.rc[d]
            p = self.px[d]
            nav = self.units * p + self.cash
            if d == first:
                w = self.units * p / nav
                trade = (target - w) * nav
                nav -= abs(trade) * RATE + max(0.0, -trade) * TAX
                self.units = target * nav / p
                self.cash = nav - self.units * p
            self.nav = nav
            self.peak = max(self.peak, nav)
            if rec is not None:
                rec.append((d, nav))
        self.i = int(j)
        return math.log(self.nav / nav0), max(0.0, self.dd() - dd0)


def norm_stats(X: np.ndarray):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def normalize(X: np.ndarray, mu, sd) -> np.ndarray:
    return np.clip((X - np.asarray(mu)) / np.asarray(sd), -CLIP, CLIP)


def obs_vec(Z: np.ndarray, p: int, book: Book) -> np.ndarray:
    return np.append(Z[p], book.weight()).astype(np.float32)


def make_env_class():
    import gymnasium as gym
    from gymnasium import spaces

    class T7Env(gym.Env):
        """보상 = 100 × ln(잔고 변화, 현금 이자 포함) − 0.01 × 100 × 낙폭 증가 − κ × 100 × (1 − E) × 구간 거래일 ÷ 252. 학습 리밸런싱일 안에서 무작위 시작, EP_LEN 스텝 뒤 끝."""

        def __init__(self, px, Z, dec, lam, rc=None):
            super().__init__()
            self.px, self.Z, self.dec, self.lam = px, Z, np.asarray(dec, dtype=np.int64), float(lam)
            self.rc = rc
            self.observation_space = spaces.Box(-np.inf, np.inf, (Z.shape[1] + 1,), np.float32)
            self.action_space = spaces.Discrete(len(LEVELS))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.p = int(self.np_random.integers(0, len(self.dec) - EP_LEN))
            self.t = 0
            self.book = Book(self.px, int(self.dec[self.p]), float(LEVELS[int(self.np_random.integers(0, len(LEVELS)))]), rc=self.rc)
            return obs_vec(self.Z, self.p, self.book), {}

        def step(self, action):
            i_, j_ = int(self.dec[self.p]), int(self.dec[self.p + 1])
            e_ = float(LEVELS[int(action)])
            lv, dv = self.book.run(j_, e_)
            r = 100.0 * lv - DD_LAM * 100.0 * dv - self.lam * 100.0 * (1.0 - e_) * (j_ - i_) / 252.0
            self.p += 1
            self.t += 1
            truncated = self.t >= EP_LEN or self.p + 1 >= len(self.dec)
            return obs_vec(self.Z, self.p, self.book), float(r), False, truncated, {}

    return T7Env


# ---------------------------------------------------------------------------
# 검증 (엔진 밖 장부, 2021~2023)
# ---------------------------------------------------------------------------

def decide(models, o: np.ndarray) -> float:
    return float(np.mean([LEVELS[int(m.predict(o, deterministic=True)[0])] for m in models]))


def path_from(data: Data, models, mu, sd, start_year: int, end_year: int, finish: int | None = None, rec: list | None = None):
    """start_year 첫 리밸런싱일부터 end_year 마지막 리밸런싱일까지 E를 정한다 (시작 비중 E 0.466107). 반환: (E 목록, 장부)."""
    f = data.f
    rows = np.flatnonzero((f["year"] >= start_year).to_numpy() & (f["year"] <= end_year).to_numpy())
    X = normalize(f[FEATS].to_numpy(dtype=float), mu, sd)
    dec = f["k"].to_numpy(dtype=np.int64)
    book, es = Book(data.px, int(dec[rows[0]]), E_C53, rc=data.rc), []
    for n_, k in enumerate(rows):
        e = decide(models, obs_vec(X, k, book))
        es.append(e)
        if n_ + 1 < len(rows):
            j = int(dec[rows[n_ + 1]])
        else:
            j = finish if finish is not None else (int(dec[k + 1]) if k + 1 < len(dec) else len(data.px) - 1)
        if j > book.i:
            book.run(j, e, rec)
    return es, book


def val_metrics(data: Data, models, mu, sd) -> dict:
    """검증 2021~2023: 2021 첫 리밸런싱일 E 0.466107에서 시작, 2023년 마지막 거래일에서 끝."""
    last = int(np.flatnonzero(data.days.year <= VAL[1])[-1])
    rec = []
    es, _ = path_from(data, models, mu, sd, VAL[0], VAL[1], finish=last, rec=rec)
    nav = pd.Series([v for _, v in rec], index=data.days[[d for d, _ in rec]])
    n = len(nav) - 1
    cagr = float(nav.iloc[-1] ** (252 / n) - 1)
    mdd = float((1 - nav / nav.cummax()).max())
    return {"cagr": cagr, "mdd": mdd, "score": cagr / max(mdd, MDD_FLOOR), "mean_e": float(np.mean(es))}


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def train_job(data: Data, lam: float, seed: int, out: Path, steps: int = STEPS, every: int = EVAL_EVERY) -> dict:
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    torch.set_num_threads(1)
    g = data.train_rows()
    mu, sd = norm_stats(g[FEATS].to_numpy(dtype=float))
    Z = normalize(g[FEATS].to_numpy(dtype=float), mu, sd)
    env = make_env_class()(data.px, Z, g["k"].to_numpy(dtype=np.int64), lam, data.rc)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    class ValCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best = -np.inf

        def _check(self):
            v = val_metrics(data, [self.model], mu, sd)
            log.append({"timesteps": int(self.num_timesteps), **v})
            if v["score"] > self.best:
                self.best = v["score"]
                self.model.save(out / "best.zip")

        def _on_step(self) -> bool:
            if self.num_timesteps % every == 0:
                self._check()
            return True

    t0 = time.time()
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0)
    cb = ValCallback()
    model.learn(total_timesteps=int(steps), callback=cb)
    model.save(out / "last.zip")
    lg = pd.DataFrame(log)
    lg.to_csv(out / "eval_log.csv", index=False)
    best = lg.loc[lg["score"].idxmax()]
    (out / "stats.json").write_text(json.dumps({"cols": FEATS, "mu": list(map(float, mu)), "sd": list(map(float, sd)), "lam": lam,
                                                "train": [str(g["date"].iloc[0].date()), str(g["date"].iloc[-1].date())],
                                                "train_rows": int(len(g))}), encoding="utf-8")
    res = {"lam": lam, "seed": seed, "steps": int(steps), "train_rows": int(len(g)),
           "train": [str(g["date"].iloc[0].date()), str(g["date"].iloc[-1].date())],
           "best_timesteps": int(best["timesteps"]), "best_val_score": float(best["score"]), "best_val_cagr": float(best["cagr"]),
           "best_val_mdd": float(best["mdd"]), "best_val_mean_e": float(best["mean_e"]), "seconds": round(time.time() - t0, 1),
           "sb3": stable_baselines3.__version__, "torch": torch.__version__}
    (out / "done.json").write_text(json.dumps(res), encoding="utf-8")
    return res


def load_best(lam: float):
    from stable_baselines3 import PPO

    models = [PPO.load(RUNS / lam_id(lam) / f"s{s}" / "best.zip", device="cpu") for s in SEEDS]
    st = json.loads((RUNS / lam_id(lam) / f"s{SEEDS[0]}" / "stats.json").read_text(encoding="utf-8"))
    if st["cols"] != FEATS:
        raise RuntimeError("특성 열이 다르다")
    return models, np.array(st["mu"]), np.array(st["sd"])


# ---------------------------------------------------------------------------
# 테스트 (C 엔진)
# ---------------------------------------------------------------------------

def live_es(data: Data, models, mu, sd) -> list[float]:
    """평가 리밸런싱일(2016~2025) E: 2016~2020은 E 0.466107, 2021부터 에이전트 (2021 첫 리밸런싱일 E 0.466107에서 시작한 장부)."""
    es_active, _ = path_from(data, models, mu, sd, ACTIVE[0], ACTIVE[-1])
    yr = data.lv["year"].to_numpy()
    pre = [E_C53] * int((yr < ACTIVE[0]).sum())
    if len(pre) + len(es_active) != len(data.lv):
        raise RuntimeError("운용 리밸런싱일 수가 맞지 않다")
    return pre + es_active


def full_es(es_reb: list[float], n_dates: int) -> list[float]:
    if len(es_reb) != (n_dates + 1) // 2:
        raise RuntimeError("리밸런싱일 수가 신호일 수와 맞지 않다")
    return [float(es_reb[i // 2]) for i in range(n_dates)]


def period_stats(nav: pd.Series, y0: int, y1: int, fills: pd.DatetimeIndex) -> tuple[float, float]:
    start = fills[fills.year == y0][0]
    s = nav[(nav.index >= start) & (nav.index.year <= y1)]
    s = s / s.iloc[0]
    n = len(s) - 1
    return float(s.iloc[-1] ** (252 / n) - 1), float((1 - s / s.cummax()).max())


def run_cash(ctx, snaps, b, es: list[float]):
    """T7 보유 80 + 리밸런싱 주 E (turnover.run과 같은 흐름) + 엔진 원장의 일별 현금 비중."""
    use = [replace(sn, exposure=float(e)) for sn, e in zip(snaps, es)]
    pos = buffer_run(ctx, use, buffer=b, mode="full", with_m9=True, trade_every=EVERY)
    eng = ctx.engine(True, ledger=True)
    rep = replay(pos, ctx.dates, eng, ctx.last_period)
    sm = summary(rep, ctx.inputs.bm_close, pos)
    m = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v) for k, v in sm.items()}
    return m, rep.nav, cash_fraction(eng.ledger)


def run_all(smoke_only: bool = False) -> None:
    t_all = time.time()
    if RUNS.exists() and not smoke_only:
        shutil.rmtree(RUNS)  # 이전 v4 결과를 지우고 처음부터
    print("1) 바구니·특성을 처음부터 만든다", flush=True)
    build_data()
    data = Data()
    if smoke_only:
        sm = RUNS / "_smoke"
        if sm.exists():
            shutil.rmtree(sm)
        r = train_job(data, 1.0, 0, sm, steps=2_048, every=1_024)
        print(json.dumps(r, ensure_ascii=False))
        print("smoke ok")
        return

    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("plan", f"t7f|{lam_id(LAMBDAS[0])}"):
        for lam in LAMBDAS:
            ledger.plan(f"t7f|{lam_id(lam)}", TRACK,
                        f"v6: 현금 = CD 91일물, 처음부터 학습 (2011~2020, v4와 같은 바구니), 보상 = 100 × ln(잔고 변화, 현금 이자 포함) − 0.01 × 100 × 낙폭 증가 − κ × 100 × (1 − E) × 구간 거래일 ÷ 252 (κ = {lam:g}), "
                        f"E 11단계, PPO {STEPS:,}스텝, {EVAL_EVERY:,}스텝마다 검증 2021~2023 점수 최고 체크포인트, "
                        "5시드 앙상블, 테스트 2024~2025 C 엔진",
                        "사용자 결정(11/20): 상승장에서 더 과감하게 투자하도록 보상 변경, 현금은 안정자산(CD 91일물) 수익")

    print(f"2) 학습: κ {len(LAMBDAS)}개 × 시드 {len(SEEDS)}개, 각 {STEPS:,}스텝, {EVAL_EVERY:,}스텝마다 검증", flush=True)
    jobs = [(lam, s) for lam in LAMBDAS for s in SEEDS]
    for n_, (lam, s) in enumerate(jobs, 1):
        r = train_job(data, lam, s, RUNS / lam_id(lam) / f"s{s}")
        print(f"  [{n_}/{len(jobs)}] κ {lam:g} 시드 {s}: {r['steps']:,}스텝 ({r['seconds']:.0f}초), 학습 {r['train'][0]} ~ {r['train'][1]} "
              f"({r['train_rows']}개), 검증 최고 {r['best_timesteps']:,}스텝 점수 {r['best_val_score']:+.2f} "
              f"(CAGR {r['best_val_cagr']:+.2%}, MDD {r['best_val_mdd']:.1%}, 평균 E {r['best_val_mean_e']:.2f})", flush=True)

    print("3) 검증 2021~2023으로 κ 고르기 (앙상블, 엔진 밖 장부, 현금 = CD 91일물)", flush=True)
    sel, best_models = {}, {}
    for lam in LAMBDAS:
        models, mu, sd = load_best(lam)
        best_models[lam] = (models, mu, sd)
        sel[lam] = val_metrics(data, models, mu, sd)
        print(f"  κ {lam:g}: 점수 {sel[lam]['score']:+.2f} (CAGR {sel[lam]['cagr']:+.2%}, MDD {sel[lam]['mdd']:.1%}, 평균 E {sel[lam]['mean_e']:.2f})",
              flush=True)
    pick = max(LAMBDAS, key=lambda l: sel[l]["score"])
    print(f"  → 고른 κ = {pick:g}", flush=True)

    print("4) 테스트 2024~2025 (C 엔진)", flush=True)
    dates = [d for d in signal_dates(io.read_table("universe"), kospi200(io.read_table("index")).index, FIRST_SIGNAL_DATE,
                                     LAST_SIGNAL_DATE) if 2016 <= int(d[:4]) <= 2025]
    ctx = Ctx(ALL_YEARS)
    if list(ctx.dates) != dates:
        raise RuntimeError("Ctx 신호일이 다르다")
    snaps = ctx.snaps(signals=smooth_signals(ctx.signals, 4), key="sm4")
    b = replace(ctx.buffer, enter_rank_max=N, exit_rank_greater_than=N + BAND, max_holdings=N)
    k200 = ctx.inputs.bm_close.copy()
    k200.index = _ns(k200.index)
    cal = k200.index[k200.index <= pd.Timestamp("2025-12-30")]
    fills = pd.DatetimeIndex([cal[cal.searchsorted(d, side="right")] for d in _ns(dates)])
    yr_reb = data.lv["year"].to_numpy()
    EVAL.mkdir(parents=True, exist_ok=True)
    results = {}
    CY = CashYield()

    def engine(name: str, es_reb: list[float]) -> dict:
        t = time.time()
        m, nav_raw, cfrac = run_cash(ctx, snaps, b, full_es(es_reb, len(dates)))
        nav_raw = nav_raw.copy()
        nav_raw.index = _ns(nav_raw.index)
        nav = CY.adjust(nav_raw, cfrac)  # 현금 = CD 91일물로 보정한 NAV (이 뒤의 지표는 모두 보정 값)
        pd.DataFrame({"date": nav.index, "nav": nav.to_numpy(), "nav_raw": nav_raw.to_numpy(),
                      "cash_frac": cfrac.reindex(nav.index).ffill().to_numpy()}).to_parquet(EVAL / f"nav_{name}.parquet", index=False)
        es_a = np.array(es_reb)[yr_reb >= ACTIVE[0]]
        n10 = len(nav) - 1
        r = {"name": name, "cagr10": float(nav.iloc[-1] ** (252 / n10) - 1), "mdd10": float((1 - nav / nav.cummax()).max()),
             "cagr10_raw": float(m["cagr"]), "mdd10_raw": float(m["mdd"]), "mean_cash": float(cfrac.mean()),
             "turnover10": float(m["annual_turnover"]),
             "mean_e": float(es_a.mean()), "switches": int((np.abs(np.diff(es_a)) > 1e-9).sum())}
        for p, (y0, y1) in PERIODS.items():
            r[f"cagr_{p}"], r[f"mdd_{p}"] = period_stats(nav, y0, y1, fills)
        r["yearly"] = {int(y): float(v) for y, v in (lambda l: l / l.shift(1) - 1)(nav.groupby(nav.index.year).last()).items()
                       if y >= ACTIVE[0]}
        r["window_mean_e"] = {}
        for wname, a, z in [w_ for w_ in ex.WINDOWS if int(w_[1][:4]) >= ACTIVE[0]]:
            msk = ((data.lv["date"] >= pd.Timestamp(a)) & (data.lv["date"] <= pd.Timestamp(z))).to_numpy()
            r["window_mean_e"][wname] = float(np.mean(np.array(es_reb)[msk])) if msk.any() else None
        results[name] = r
        (EVAL / f"{name}.json").write_text(json.dumps({**r, "es": es_reb}, ensure_ascii=False, default=float), encoding="utf-8")
        print(f"  {name}: 테스트 {r['cagr_test']:+.2%}·{r['mdd_test']:.1%}, 검증 {r['cagr_val']:+.2%}·{r['mdd_val']:.1%}, "
              f"평균 E {r['mean_e']:.2f} ({time.time() - t:.0f}초)", flush=True)
        return r

    c53 = engine("c53", [E_C53] * len(data.lv))
    ref = json.loads((SAMECASH / "cash.json").read_text(encoding="utf-8"))
    d = abs(c53["cagr10_raw"] - ref["summary_cagr"])
    print(f"  대조 (현금 보정 전): E 고정 CAGR {c53['cagr10_raw']:+.6%} vs 25절 T7-80-C53 {ref['summary_cagr']:+.6%} (차이 {d:.1e}); "
          f"현금 = CD 91일물로 보정하면 {c53['cagr10']:+.4%}", flush=True)
    if d > 1e-12:
        raise RuntimeError("E 고정 판이 25절 T7-80-C53과 다르다")
    for lam in LAMBDAS:
        models, mu, sd = best_models[lam]
        names = [("ens", models)] + ([(f"s{s}", [models[i]]) for i, s in enumerate(SEEDS)] if lam == pick else [])
        for tag, ms in names:
            es_reb = live_es(data, ms, mu, sd)
            engine(f"{lam_id(lam)}_{tag}", es_reb)
            mean_e = float(np.mean(np.array(es_reb)[yr_reb >= ACTIVE[0]]))
            engine(f"{lam_id(lam)}_{tag}_fixed", [E_C53 if y < ACTIVE[0] else mean_e for y in yr_reb])

    k = pd.read_parquet(KOSPI_PATH)
    kospi = pd.Series(k["close"].to_numpy(dtype=float), index=_ns(k["date"]))
    ref_rows = {}
    for nm, s in (("kospi", kospi), ("k200", k200), ("e1", navp(SAMECASH / "nav_e1.parquet"))):
        s = s[(s.index >= fills[0]) & (s.index <= pd.Timestamp("2025-12-30"))]
        ref_rows[nm] = {f"{x}_{p}": v for p, (y0, y1) in PERIODS.items() for x, v in zip(("cagr", "mdd"), period_stats(s, y0, y1, fills))}
        if nm in ("kospi", "k200"):  # 지수 46.6% + 현금(CD 91일물), 매일 같은 비중, 비용 없음
            mx = CY.mix(s, E_C53)
            ref_rows[f"{nm}_mix"] = {f"{x}_{p}": v for p, (y0, y1) in PERIODS.items()
                                     for x, v in zip(("cagr", "mdd"), period_stats(mx, y0, y1, fills))}

    P = lam_id(pick)
    ens, fx, base = results[f"{P}_ens"], results[f"{P}_ens_fixed"], results["c53"]
    seeds_beat = int(sum(results[f"{P}_s{s}"]["cagr_test"] > results[f"{P}_s{s}_fixed"]["cagr_test"] for s in SEEDS))
    q1 = "(1) 타이밍 실력 (테스트 2024~2025: 앙상블 CAGR·MDD 모두 같은 평균 고정보다 낫고 시드 4/5 이상)"
    q2 = "(2) 고정 모델 대비 (테스트 2024~2025: CAGR > T7-80-C53, MDD < T7-80-C53)"
    j = {"고른 κ (검증 2021~2023 앙상블 점수 최고)": pick, "검증 점수": {f"{l:g}": sel[l] for l in LAMBDAS},
         q1: bool(ens["cagr_test"] > fx["cagr_test"] and ens["mdd_test"] < fx["mdd_test"] and seeds_beat >= 4),
         q2: bool(ens["cagr_test"] > base["cagr_test"] and ens["mdd_test"] < base["mdd_test"]),
         "자기 고정보다 CAGR 높은 시드 수 (테스트)": seeds_beat}
    j["교체 후보"] = bool(j[q1] and j[q2])
    flat = []
    for n_, r in list(results.items()) + [(k_, {"name": k_, **v}) for k_, v in ref_rows.items()]:
        f_ = {k_: v for k_, v in r.items() if not isinstance(v, dict)}
        f_.update({f"y{y}": v for y, v in r.get("yearly", {}).items()})
        f_.update({f"e:{w}": v for w, v in r.get("window_mean_e", {}).items()})
        flat.append(f_)
    rd = pd.DataFrame(flat)
    rd.to_csv(HERE / "t7f_runs.csv", index=False)
    train = {f"{lam_id(l)}_s{s}": json.loads((RUNS / lam_id(l) / f"s{s}" / "done.json").read_text(encoding="utf-8"))
             for l in LAMBDAS for s in SEEDS}
    (HERE / "t7f_judgment.json").write_text(json.dumps({"judgment": j, "results": results, "refs": ref_rows, "train": train},
                                                      ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not ledger_has("result", "t7f|판정"):
        for lam in LAMBDAS:
            r = results[f"{lam_id(lam)}_ens"]
            ledger.result(f"t7f|{lam_id(lam)}", **{k_: r[k_] for k_ in ("cagr_val", "mdd_val", "cagr_test", "mdd_test", "cagr_all",
                                                                       "mdd_all", "mean_e", "turnover10")})
        ledger.result("t7f|판정", **{k_: v for k_, v in j.items() if k_ != "검증 점수"})

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    show = rd.copy()
    for c in [c for c in show.columns if c.startswith(("cagr", "mdd")) or (c.startswith("y") and c[1:].isdigit())]:
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(1)
    for c in ["mean_e", "turnover10"] + [c for c in show.columns if c.startswith("e:")]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    print("\n[기간별 CAGR·MDD %: val = 검증 2021~2023, test = 테스트 2024~2025, all = 2021~2025]")
    print(show[[c for c in ["name", "cagr_val", "mdd_val", "cagr_test", "mdd_test", "cagr_all", "mdd_all", "mean_e", "switches",
                            "turnover10"] if c in show.columns]].to_string(index=False))
    print("\n[연도별 수익률 %]")
    print(show[["name"] + [c for c in show.columns if c.startswith("y") and c[1:].isdigit()]].to_string(index=False))
    print("\n[판정] (실행 전 고정)")
    for q, v in j.items():
        if q != "검증 점수":
            print(f"  {q}: {v if not isinstance(v, bool) else ('만족' if v else '미달')}")
    print(f"done ({(time.time() - t_all) / 60:.0f}분)")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run_all()
    elif cmd == "smoke":
        run_all(smoke_only=True)
    else:
        raise SystemExit(f"모르는 명령: {cmd} (run | smoke)")


if __name__ == "__main__":
    main()
