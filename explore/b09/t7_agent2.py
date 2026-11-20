r"""T7 비중 에이전트 v2: 학습 기간을 2011년부터로 늘린다 (11/20 사용자 결정: "우리 데이터가 2011년부터 있다, 10년치 학습하고 검증은 2021~2025").

v1(t7_agent.py, docs 28절)과 다른 것은 학습 바구니의 기간뿐이다. 판정·평가·행동·보상·학습 방법은 v1과 같다.
- v1 결과를 본 뒤의 설계다: v1은 2020 폭락 때 E 0.96이었다 (폴드 2020이 2016~2019만 배워 큰 폭락을 본 적이 없음).
  v2는 2011 폭락·2015 하락이 학습에 들어간다. 2021년 폴드는 2011~2020 10년을 배운다.
- 2019~2020(고르기)과 2019~2025 판정 (3)은 그대로 둔다. 2020 폭락을 판정에서 빼지 않는다.

실행 (저장소 루트, PowerShell). 모두 사용자가 돌린다.
  1) 2011~2015 바구니 (2~5분):  .\.venv\Scripts\python.exe -I explore\b09\run.py t7_agent2.py basket
  2) 준비·점검 (1분):           .\.venv\Scripts\python.exe -I explore\b09\run.py t7_agent2.py prep
  3) 짧은 학습 점검 (1분):       .\.venv\Scripts\python.exe -I explore\b09\run.py t7_agent2.py smoke
  4) 학습 (작업 105개, 창 4개):   .\.venv\Scripts\python.exe -I explore\b09\run.py t7_agent2.py jobs
                                .\.venv\Scripts\python.exe -I explore\b09\run.py t7_agent2.py worker explore\b09\jobs_t7b.txt 0 4
                                (창 4개에서 끝에서 두 번째 숫자 0·1·2·3). 진행: ... t7_agent2.py status
  5) 평가 (C 엔진, 30분 안팎, 창 1개): .\.venv\Scripts\python.exe -I explore\b09\run.py t7_agent2.py evaluate
  끊기면 같은 명령을 다시 돌린다. 끝난 학습·평가는 건너뛴다.

학습 바구니 (실행 전 고정 11/20)
- 2011-01-07 ~ 2015-12-04: A 폴드 신호 20261117_A_rlsig_y2016 (2011~2013 교차 적합, 2014~2015 폴드 2016 모델, 2016년 뒤 정보 없음).
  15절 실전 바구니와 같은 출처다. 교차 적합이라 실제보다 낙관적일 수 있다 (17·21·23절). 그 판 바구니 수익을 prep이 찍는다.
- 바구니 규칙은 T7 보유 80과 같다: rank_score 4주 평균 순위, 진입 rank ≤ 80, 청산 rank > 140, 최대 80종목, 같은 비중,
  crash_flag = 1 제외, 2개 신호일마다(2011-01-07부터 짝수 번째) 리밸런싱, 다음 거래일 종가에 맞춘다.
  교체 비용 = 편도 회전율 × (2 × 한쪽 비용 + 매도세). 엔진 밖 계산이라 종목·업종 한도와 M9는 없다 (15절과 같음).
  2015-12-04 뒤 신호가 없는 주(A purge)는 리밸런싱 없이 들고 간다.
- 2016-01-11부터는 v1과 같은 T7 보유 80 E = 1 엔진 NAV를 이어 붙인다 (그날 엔진 첫 매수 비용 포함).
- 점검: 같은 엔진 밖 계산을 2016~2025 시험 신호로 돌려 엔진 NAV와 비교한다 (일간 수익 상관 ≥ 0.98, CAGR 차 ≤ 1%p, MDD 차 ≤ 3%p, 아니면 멈춤).
- 학습 행: 특성이 모두 있는 첫 리밸런싱일(2011년 4월 무렵)부터. 2015-12-04와 2016-01-08 사이 한 구간은 5주다.

나머지 (관측, 행동, 보상, PPO, walk-forward 2019~2025, 앙상블, C 엔진 평가, 대조, 판정)는 v1(t7_agent.py) 머리말과 같다.
- 판정 (실행 전 고정, v1과 같음): λ는 2019~2020 앙상블 CAGR/MDD 최고. 고른 판이 2021~2025에서
  (1) 같은 평균 비중 고정보다 CAGR↑·MDD↓, 시드 4/5 이상 (2) T7-80-C53보다 CAGR↑·MDD↓ (3) 2019~2025에서도 (2) → '교체 후보'.
- 탐색 결과다 (R-15). 결과: t7b_runs.csv, t7b_judgment.json, runs/t7b/, 원장 t7b|*.
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
from common import ALL_YEARS, HERE, SIG_RUN, Ctx, Ledger
import exitrule as ex
import numpy as np
import pandas as pd
from src.common import io
from src.portfolio.env_check_report import FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE, kospi200, signal_dates
from src.portfolio.params import PREREG_PATH, load_prereg
from turnover import ledger_has, run, smooth_signals

SAMECASH = HERE / "runs" / "samecash"
RUNS = HERE / "runs" / "t7b"
EVAL = RUNS / "eval"
PX_PATH = HERE / "t7b_px.parquet"
EXT_PATH = HERE / "t7b_basket.parquet"  # 엔진 밖 바구니 (2011~ 학습용, 2016~ 점검용)
FEAT_PATH = HERE / "t7b_features.parquet"
JOBS = HERE / "jobs_t7b.txt"
MT_FEAT = HERE / "mt_features.parquet"

E_C53 = 1.0 - 0.533893  # T7-80-C53의 E (samecash.py와 같은 값)
LEVELS = np.array(sorted(set(np.round(np.arange(11) / 10.0, 1).tolist()) | {E_C53}))  # 고정 모델 값을 그대로 고를 수 있게
LAMBDAS = (0.5, 1.0, 2.0)
SEEDS = (0, 1, 2, 3, 4)
TEST_YEARS = tuple(range(2019, 2026))
STEPS, EP_LEN = 20_000, 26
CLIP = 5.0
N, BAND, EVERY = 80, 60, 2
IDX = ["r5", "r21", "r63", "r126", "vol20", "vol60", "ma50gap", "ma200gap", "dd126",
       "breadth50", "breadth200", "adv20", "disp20", "valtrend"]
BKT = ["b_r5", "b_r21", "b_r63", "b_vol20", "b_vol60", "b_dd"]
FEATS = IDX + BKT
PERIODS = {"select": (2019, 2020), "confirm": (2021, 2025), "all": (2019, 2025)}
PERIOD_LABEL = {"select": "2019~2020 (고르기)", "confirm": "2021~2025 (확인)", "all": "2019~2025"}
TRACK = "T7 비중 에이전트 v2 (학습 2011~, 리밸런싱마다 현금 보유율)"
RLSIG_2011 = "20261117_A_rlsig_y2016"
LIVE_START = pd.Timestamp("2016-01-08")  # 평가(운용) 첫 리밸런싱일

_cfg = load_prereg(PREREG_PATH)
assert float(_cfg["costs"]["cash_return"]) == 0.0, "부록 1 cash_return이 0이 아니다"
RATE = float(_cfg["costs"]["commission_per_side"]) + float(_cfg["costs"]["slippage_per_side"])
TAX = float(_cfg["costs"]["sell_tax"])


def lam_id(lam: float) -> str:
    return f"l{lam:g}"


def _ns(x) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(x)).as_unit("ns")


def navp(path: Path) -> pd.Series:
    f = pd.read_parquet(path)
    return pd.Series(f["nav"].to_numpy(dtype=float), index=_ns(f["date"]))


def signal_list() -> list[str]:
    """2016~2025 신호일 (common.Ctx와 같은 계산)."""
    u, bm = io.read_table("universe"), kospi200(io.read_table("index"))
    return [d for d in signal_dates(u, bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE) if 2016 <= int(d[:4]) <= 2025]


# ---------------------------------------------------------------------------
# 장부
# ---------------------------------------------------------------------------

class Book:
    """바구니(잔고 px) + 현금. 신호일 종가에 정한 비중을 다음 거래일 종가에 맞춘다 (현금 수익 0, 줄일 때 매도세)."""

    __slots__ = ("px", "i", "units", "cash", "nav", "peak", "turn")

    def __init__(self, px: np.ndarray, i0: int, w0: float = 0.0) -> None:
        self.px, self.i = px, int(i0)
        self.nav, self.peak, self.turn = 1.0, 1.0, 0.0
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
            p = self.px[d]
            nav = self.units * p + self.cash
            if d == first:
                w = self.units * p / nav
                trade = (target - w) * nav
                nav -= abs(trade) * RATE + max(0.0, -trade) * TAX
                self.units = target * nav / p
                self.cash = nav - self.units * p
                self.turn += abs(target - w)
            self.nav = nav
            self.peak = max(self.peak, nav)
            if rec is not None:
                rec.append((d, nav))
        self.i = int(j)
        return math.log(self.nav / nav0), max(0.0, self.dd() - dd0)


def reward(lv: float, dv: float, lf: float, df: float, lam: float) -> float:
    return 100.0 * (lv - lf) - lam * 100.0 * (dv - df)


def norm_stats(X: np.ndarray):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def normalize(X: np.ndarray, mu, sd) -> np.ndarray:
    return np.clip((X - np.asarray(mu)) / np.asarray(sd), -CLIP, CLIP)


def obs_vec(Z: np.ndarray, p: int, book: Book) -> np.ndarray:
    return np.append(Z[p], book.weight()).astype(np.float32)


def play_fixed(px: np.ndarray, dec: np.ndarray, e: float, finish: int) -> pd.Series:
    """dec[0]부터 리밸런싱마다 같은 E로 맞춘 장부의 일별 잔고 (dec[0] 다음 날부터)."""
    book, rec = Book(px, int(dec[0]), 0.0), []
    for k in range(len(dec)):
        j = int(dec[k + 1]) if k + 1 < len(dec) else finish
        if j > book.i:
            book.run(j, e, rec)
    return pd.Series([v for _, v in rec], index=[d for d, _ in rec])


# ---------------------------------------------------------------------------
# 준비
# ---------------------------------------------------------------------------

class Data:
    def __init__(self) -> None:
        w = pd.read_parquet(PX_PATH)
        self.days = _ns(w["date"])
        self.px = w["px"].to_numpy(dtype=float)
        f = pd.read_parquet(FEAT_PATH)
        f["date"] = _ns(f["date"])
        self.f = f.sort_values("date").reset_index(drop=True)
        self.lv = self.f[self.f["date"] >= LIVE_START].reset_index(drop=True)  # 평가(운용) 리밸런싱일 2016~

    def train_rows(self, y: int) -> pd.DataFrame:
        g = self.f[self.f["ok"] & (self.f["date"].dt.year <= y - 1)].reset_index(drop=True)
        if len(g) < EP_LEN + 1:
            raise RuntimeError(f"폴드 {y}: 학습 리밸런싱일 {len(g)}개 < 에피소드 {EP_LEN} + 1")
        return g


def t7_book(sig: pd.DataFrame) -> dict:
    """신호일 → (종목, 4주 평균 순위, crash_flag) 배열 (순위 오름차순, 같으면 종목코드 순)."""
    sm = smooth_signals(sig, 4)
    s = pd.DataFrame({"date": _ns(sm["date"]), "ticker": sm["ticker"].astype(str).to_numpy(),
                      "rank": sm["rank"].to_numpy(dtype=np.int64), "flag": sm["crash_flag"].to_numpy(dtype=np.int64)})
    s = s.sort_values(["date", "rank", "ticker"], kind="stable")
    return {pd.Timestamp(d): (g["ticker"].to_numpy(), g["rank"].to_numpy(), g["flag"].to_numpy())
            for d, g in s.groupby("date", sort=True)}


def choose_t7(entry, held: set) -> list:
    """T7 보유 80 단계 2~3 (쿨다운 없음): 유지 rank ≤ 140, 추가 rank ≤ 80, 최대 80, crash_flag = 0."""
    tick, rank, flag = entry
    keep = [t for t, r, f in zip(tick, rank, flag) if f == 0 and t in held and r <= N + BAND]
    room = max(N - len(keep), 0)
    add = [t for t, r, f in zip(tick, rank, flag) if f == 0 and t not in held and r <= N][:room]
    return keep + add


def simulate_t7(days: pd.DatetimeIndex, R: np.ndarray, col: dict, decs: list):
    """리밸런싱 신호일 종가에 고르고 다음 거래일 종가에 같은 비중으로 맞추는 바구니의 일별 잔고 (첫 체결일 1, 그 전 NaN).
    교체 비용 = 편도 회전율 × (2 × 한쪽 비용 + 매도세). 15절 bt_agent.simulate와 같은 계산에 T7 보유 80 규칙."""
    n = len(days)
    fill = {}
    for d, entry in decs:
        k = int(days.searchsorted(d, side="right"))
        if k < n:
            fill[k] = (d, entry)
    nav = np.full(n, np.nan)
    held, idx, w = set(), np.zeros(0, dtype=np.int64), np.zeros(0)
    v, started, turns, sizes, no_price = 1.0, False, [], [], 0
    for k in range(n):
        if len(idx):
            r = R[k, idx]
            g = float(w @ r)
            v *= 1.0 + g
            w = w * (1.0 + r) / (1.0 + g)
        if k in fill:
            d, entry = fill[k]
            picked = choose_t7(entry, held)
            new = [t for t in picked if t in col]
            no_price += len(picked) - len(new)
            nidx = np.array([col[t] for t in new], dtype=np.int64)
            nw = np.full(len(nidx), 1.0 / len(nidx)) if len(nidx) else np.zeros(0)
            old = dict(zip(idx.tolist(), w.tolist()))
            tgt = dict(zip(nidx.tolist(), nw.tolist()))
            tau = 0.5 * sum(abs(tgt.get(j, 0.0) - old.get(j, 0.0)) for j in set(old) | set(tgt))
            if started:
                v *= 1.0 - tau * (2.0 * RATE + TAX)
                turns.append(tau)
            held, idx, w, started = set(new), nidx, nw, True
            sizes.append(len(new))
        if started:
            nav[k] = v
    return nav, turns, sizes, no_price


def _cm(v: pd.Series) -> tuple[float, float]:
    v = v.dropna()
    v = v / v.iloc[0]
    n = len(v) - 1
    return float(v.iloc[-1] ** (252 / n) - 1), float((1 - v / v.cummax()).max())


def basket() -> None:
    """2011~2015 학습 바구니(엔진 밖, rlsig_y2016)와 점검용 2016~ 엔진 밖 바구니(시험 신호)를 만든다."""
    import pyarrow.parquet as pq

    t0 = time.time()
    bm = kospi200(io.read_table("index"))
    cal = [d for d in signal_dates(io.read_table("universe"), bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)
           if 2011 <= int(d[:4]) <= 2025]
    days = _ns(bm.index)
    days = days[(days >= pd.Timestamp(cal[0])) & (days <= pd.Timestamp("2025-12-30"))]
    old, new = t7_book(io.read_table("signals", run_id=RLSIG_2011)), t7_book(io.read_table("signals", run_id=SIG_RUN))
    need = set()
    for b in (old, new):
        for tick, rank, _ in b.values():
            need.update(tick[rank <= N + BAND].tolist())
    tbl = pq.read_table(io.ROOT / "data" / "processed" / "prices.parquet", columns=["date", "ticker", "close"],
                        filters=[("ticker", "in", sorted(need)), ("date", ">=", days[0]), ("date", "<=", days[-1])])
    px = tbl.to_pandas()
    del tbl
    px["date"], px["ticker"] = _ns(px["date"]), px["ticker"].astype(str)
    close = px.pivot(index="date", columns="ticker", values="close").reindex(days).astype(float)
    del px
    close = close.where(close > 0).ffill()
    R = close.pct_change(fill_method=None).fillna(0.0).to_numpy()
    col = {t: k for k, t in enumerate(close.columns)}
    print(f"가격 읽음: 종목 {close.shape[1]}, 거래일 {len(days)} ({time.time() - t0:.0f}초)", flush=True)
    cal_old = [pd.Timestamp(d) for d in cal if int(d[:4]) <= 2015]
    cal_new = [pd.Timestamp(d) for d in cal if int(d[:4]) >= 2016]
    dec_old = [(d, old[d]) for k, d in enumerate(cal_old) if k % 2 == 0 and d in old]
    if any(d not in new for d in cal_new[0::2]):
        raise RuntimeError("2016년 뒤 리밸런싱일에 시험 신호가 없는 날이 있다")
    dec_new = [(d, new[d]) for d in cal_new[0::2]]
    nav_old, t_o, s_o, np_o = simulate_t7(days, R, col, dec_old)
    nav_new, t_n, s_n, np_n = simulate_t7(days, R, col, dec_new)
    pd.DataFrame({"date": days, "old": nav_old, "check": nav_new}).to_parquet(EXT_PATH, index=False)
    EXT_PATH.with_suffix(".json").write_text(json.dumps({
        "old_decisions": [str(d.date()) for d, _ in dec_old], "old_last": str(dec_old[-1][0].date()),
        "old_turnover": float(np.mean(t_o)), "old_holdings": float(np.mean(s_o)), "old_no_price": int(np_o),
        "new_turnover": float(np.mean(t_n)), "new_holdings": float(np.mean(s_n)), "new_no_price": int(np_n)},
        ensure_ascii=False), encoding="utf-8")

    e1 = navp(SAMECASH / "nav_e1.parquet")
    chk = pd.Series(nav_new, index=days).reindex(e1.index)
    ce, me = _cm(e1)
    cc, mc = _cm(chk)
    ra, rb = chk.pct_change().dropna(), e1.pct_change().dropna()
    corr = float(np.corrcoef(ra, rb.reindex(ra.index))[0, 1])
    print(f"점검 2016~2025: 엔진 밖 {cc:+.4f}·{mc:.4f} vs 엔진 E = 1 {ce:+.4f}·{me:.4f}, 일간 수익 상관 {corr:.4f}, "
          f"평균 종목 {np.mean(s_n):.1f}, 리밸런싱당 편도 교체 {np.mean(t_n):.1%}")
    if corr < 0.98 or abs(cc - ce) > 0.01 or abs(mc - me) > 0.03:
        raise RuntimeError("엔진 밖 바구니가 엔진과 너무 다르다")
    k = pd.read_parquet(KOSPI_PATH)
    kospi = pd.Series(k["close"].to_numpy(dtype=float), index=_ns(k["date"]))
    o = pd.Series(nav_old, index=days)
    o = o[(o.index <= pd.Timestamp("2015-12-30"))]
    ko = kospi[(kospi.index >= o.dropna().index[0]) & (kospi.index <= pd.Timestamp("2015-12-30"))]
    co, mo = _cm(o)
    ck, mk = _cm(ko)
    print(f"학습 바구니 2011~2015 (rlsig_y2016, 엔진 밖): CAGR {co:+.4f}·MDD {mo:.4f} (코스피 종합 {ck:+.4f}·{mk:.4f}), "
          f"리밸런싱 {len(dec_old)}번 (마지막 {dec_old[-1][0].date()}), 평균 종목 {np.mean(s_o):.1f}, 가격 없는 종목 {np_o}")
    print(f"basket ok ({time.time() - t0:.0f}초)")


def prep() -> None:
    dates = signal_list()  # 2016~2025 (평가 신호일)
    reb = dates[0::2]
    e1 = navp(SAMECASH / "nav_e1.parquet")
    c53 = navp(SAMECASH / "nav_cash.parquet")
    ext = pd.read_parquet(EXT_PATH)
    ext.index = _ns(ext["date"])
    meta = json.loads(EXT_PATH.with_suffix(".json").read_text(encoding="utf-8"))
    j0 = e1.index[0]  # 2016-01-11: 엔진 바구니 첫 체결일
    old = ext["old"]
    first_fill = old.dropna().index[0]
    d0 = ext.index[0]  # 2011-01-07
    px = pd.concat([pd.Series([1.0], index=_ns([d0])), old[(old.index >= first_fill) & (old.index < j0)],
                    float(old[j0]) * e1])
    if px.index.has_duplicates or not px.index.is_monotonic_increasing or px.isna().any():
        raise RuntimeError("이어 붙인 바구니 잔고가 이상하다")
    days = px.index
    dec_dates = _ns(meta["old_decisions"] + reb)
    dec = days.get_indexer(dec_dates)
    if (dec < 0).any():
        raise RuntimeError("리밸런싱일이 바구니 달력에 없다")

    s = np.log(px)
    r1 = s.diff()
    bf = pd.DataFrame(index=days)
    for k in (5, 21, 63):
        bf[f"b_r{k}"] = s - s.shift(k)
    bf["b_vol20"] = r1.rolling(20, min_periods=20).std() * math.sqrt(252)
    bf["b_vol60"] = r1.rolling(60, min_periods=60).std() * math.sqrt(252)
    bf["b_dd"] = s - s.cummax()
    mt = pd.read_parquet(MT_FEAT)
    mt.index = _ns(mt["date"])
    feats = pd.concat([mt.reindex(dec_dates)[IDX].reset_index(drop=True), bf.reindex(dec_dates)[BKT].reset_index(drop=True)],
                      axis=1)
    out = pd.DataFrame({"date": dec_dates, "k": dec, "year": dec_dates.year})
    out = pd.concat([out, feats], axis=1)
    out["ok"] = np.isfinite(out[FEATS].to_numpy(dtype=float)).all(axis=1)
    okv = out["ok"].to_numpy()
    first_ok = int(np.argmax(okv))
    if not okv[first_ok:].all():
        raise RuntimeError("특성이 빠진 리밸런싱일이 처음 몇 개 뒤에도 있다")
    pd.DataFrame({"date": days, "px": px.to_numpy()}).to_parquet(PX_PATH, index=False)
    out.to_parquet(FEAT_PATH, index=False)

    # 점검: 2016~ 장부 E 고정 vs 엔진 (25절). 2016~ 구간은 v1과 같은 엔진 바구니다
    live = out[out["date"] >= LIVE_START]
    dl = live["k"].to_numpy(dtype=np.int64)
    if len(live) != len(reb) or live["date"].iloc[0] != LIVE_START:
        raise RuntimeError("2016년 뒤 리밸런싱일이 평가 신호일과 다르다")

    def cm(v: pd.Series):  # 장부와 엔진 NAV 모두 2016-01-08(체결 전) 잔고 1 기준이다 (v1과 같은 식)
        v = v[v.index >= j0]
        n = len(v) - 1
        return float(v.iloc[-1] ** (252 / n) - 1), float((1 - v / v.cummax()).max())

    n_old = len(meta["old_decisions"])
    print(f"리밸런싱일 {len(out)}개 (2011~2015 {n_old}, 2016~2025 {len(reb)}), 특성이 모두 있는 첫 리밸런싱일 "
          f"{out['date'].iloc[first_ok].date()}")
    for e, ref, name in ((E_C53, c53, "E = 0.466107 (T7-80-C53)"), (1.0, e1, "E = 1")):
        a = play_fixed(px.to_numpy(), dl, e, len(px) - 1)
        a.index = days[a.index]
        ca, ma = cm(a)
        cr, mr = cm(ref)
        ra, rr = a.pct_change().dropna(), ref.pct_change().dropna()
        corr = float(np.corrcoef(ra, rr.reindex(ra.index))[0, 1])
        print(f"  장부 2016~ {name}: CAGR {ca:+.4f}·MDD {ma:.4f} | 엔진 {cr:+.4f}·{mr:.4f} | 일간 수익 상관 {corr:.4f}")
        if abs(ca - cr) > 0.002 or abs(ma - mr) > 0.005 or corr < 0.999:
            raise RuntimeError(f"장부({name})가 엔진과 너무 다르다")
    for y in TEST_YEARS:
        print(f"  폴드 {y}: 학습 리밸런싱일 {int((out['ok'] & (out['year'] <= y - 1)).sum())}개")
    print("prep ok")


# ---------------------------------------------------------------------------
# 학습 환경과 학습
# ---------------------------------------------------------------------------

def make_env_class():
    import gymnasium as gym
    from gymnasium import spaces

    class T7Env(gym.Env):
        """학습 리밸런싱일 안에서 무작위 시작, EP_LEN 스텝 뒤 끝. 보상은 머리말 식."""

        def __init__(self, px, Z, dec, lam):
            super().__init__()
            self.px, self.Z, self.dec, self.lam = px, Z, np.asarray(dec, dtype=np.int64), float(lam)
            if len(self.dec) < EP_LEN + 1:
                raise ValueError(f"학습 리밸런싱일 {len(self.dec)}개 < 에피소드 {EP_LEN} + 1")
            self.observation_space = spaces.Box(-np.inf, np.inf, (Z.shape[1] + 1,), np.float32)
            self.action_space = spaces.Discrete(len(LEVELS))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.p = int(self.np_random.integers(0, len(self.dec) - EP_LEN))
            w0 = float(LEVELS[int(self.np_random.integers(0, len(LEVELS)))])
            self.t = 0
            self.a = Book(self.px, int(self.dec[self.p]), w0)
            self.fix = Book(self.px, int(self.dec[self.p]), w0)
            return obs_vec(self.Z, self.p, self.a), {}

        def step(self, action):
            j = int(self.dec[self.p + 1])
            lv, dv = self.a.run(j, float(LEVELS[int(action)]))
            lf, df = self.fix.run(j, E_C53)
            r = reward(lv, dv, lf, df, self.lam)
            self.p += 1
            self.t += 1
            truncated = self.t >= EP_LEN or self.p + 1 >= len(self.dec)
            return obs_vec(self.Z, self.p, self.a), float(r), False, truncated, {}

    return T7Env


def run_dir(lam: float, y: int, seed: int, root: Path = RUNS) -> Path:
    return root / lam_id(lam) / f"y{y}" / f"s{seed}"


def train_job(data: Data, lam: float, y: int, seed: int, out: Path, steps: int = STEPS) -> dict:
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    g = data.train_rows(y)
    X = g[FEATS].to_numpy(dtype=float)
    mu, sd = norm_stats(X)
    Z = normalize(X, mu, sd)
    dec = g["k"].to_numpy(dtype=np.int64)
    env = make_env_class()(data.px, Z, dec, lam)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0)
    model.learn(total_timesteps=int(steps))
    model.save(out / "final.zip")
    # 진단 (고르는 데 쓰지 않는다): 학습 구간 전체를 고정 비중에서 시작해 그대로 운용
    book, fix, es, rw = Book(data.px, int(dec[0]), E_C53), Book(data.px, int(dec[0]), E_C53), [], []
    for k in range(len(dec) - 1):
        e = float(LEVELS[int(model.predict(obs_vec(Z, k, book), deterministic=True)[0])])
        es.append(e)
        lv, dv = book.run(int(dec[k + 1]), e)
        lf, df = fix.run(int(dec[k + 1]), E_C53)
        rw.append(reward(lv, dv, lf, df, lam))
    (out / "stats.json").write_text(json.dumps({"cols": FEATS, "mu": list(map(float, mu)), "sd": list(map(float, sd)),
                                                "lam": lam, "train": [str(g["date"].iloc[0].date()), str(g["date"].iloc[-1].date())]}),
                                    encoding="utf-8")
    es_a = np.array(es)
    res = {"lam": lam, "year": y, "seed": seed, "steps": int(steps), "train_rows": int(len(dec)),
           "insample_reward": float(np.sum(rw)), "insample_mean_e": float(es_a.mean()),
           "insample_low": float((es_a <= 0.2).mean()), "insample_high": float((es_a >= 0.8).mean()),
           "insample_switches": int((np.diff(es_a) != 0).sum()), "seconds": round(time.time() - t0, 1),
           "sb3": stable_baselines3.__version__, "torch": torch.__version__}
    (out / "done.json").write_text(json.dumps(res), encoding="utf-8")  # 마지막에 쓴다 (없으면 다시 학습)
    return res


def smoke() -> None:
    data = Data()
    sm = RUNS / "_smoke"
    if sm.exists():
        shutil.rmtree(sm)
    r = train_job(data, 1.0, TEST_YEARS[0], 0, sm / "l1", steps=2_048)
    print(json.dumps(r, ensure_ascii=False))
    from stable_baselines3 import PPO

    pol = {TEST_YEARS[0]: [PPO.load(sm / "l1" / "final.zip", device="cpu")]}
    st = json.loads((sm / "l1" / "stats.json").read_text(encoding="utf-8"))
    es = live_path(data, pol, {TEST_YEARS[0]: (np.array(st["mu"]), np.array(st["sd"]))})
    yr = data.lv["date"].dt.year.to_numpy()
    print(f"운용 점검: 2019 리밸런싱일 {int((yr == 2019).sum())}개, E 평균 {np.mean(np.array(es)[yr == 2019]):.2f}, "
          f"2016~2018 E가 모두 0.466107: {bool(np.allclose(np.array(es)[yr < 2019], E_C53))}")
    print("smoke ok")


def write_jobs() -> None:
    lines = [f"{lam:g} {y} {s}" for y in TEST_YEARS for lam in LAMBDAS for s in SEEDS]
    JOBS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("plan", f"t7b|{lam_id(LAMBDAS[0])}"):
        for lam in LAMBDAS:
            ledger.plan(f"t7b|{lam_id(lam)}", TRACK,
                        f"v2: 학습 바구니 2011~ (2011~2015 rlsig_y2016 엔진 밖 + 2016~ T7 보유 80 E = 1 엔진 NAV), 리밸런싱(2주)마다 E 12단계. 보상 = T7-80-C53 장부 대비 초과 로그수익 "
                        f"− {lam:g} × 초과 낙폭 증가. PPO {STEPS:,}스텝 끝 정책, 5시드 앙상블, walk-forward 2019~2025 (학습 = 2011년부터 그해 전 바구니). "
                        "평가 = C 엔진 T7 보유 80 + 리밸런싱 주 E",
                        "사용자 결정(11/20): RL 에이전트를 학습시켜 리밸런싱 주기마다 현금 보유율을 바꾼다. scratch부터 다시 학습")
    print(f"{JOBS}: 작업 {len(lines)}개 (판 {len(LAMBDAS)} × 시험 연도 {len(TEST_YEARS)} × 시드 {len(SEEDS)})")


def worker(jobs_path: str, slot: int, nslots: int) -> None:
    data = Data()
    jobs = [l.split() for l in Path(jobs_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    mine = [j for k, j in enumerate(jobs) if k % nslots == slot]
    print(f"워커 {slot}/{nslots}: 작업 {len(mine)}개", flush=True)
    for n, (lam, y, seed) in enumerate(mine, 1):
        out = run_dir(float(lam), int(y), int(seed))
        if (out / "done.json").exists():
            continue
        try:
            r = train_job(data, float(lam), int(y), int(seed), out)
            print(f"[{n}/{len(mine)}] " + json.dumps(r, ensure_ascii=False, default=float), flush=True)
        except Exception as e:  # 다음 작업으로 간다. 다시 돌리면 이 작업을 다시 시도한다
            print(f"[{n}/{len(mine)}] " + json.dumps({"job": [lam, y, seed], "error": repr(e)}, ensure_ascii=False), flush=True)
    print("done", flush=True)


def all_trained() -> bool:
    return all((run_dir(l, y, s) / "done.json").exists() for l in LAMBDAS for y in TEST_YEARS for s in SEEDS)


def status() -> None:
    done = [(l, y, s) for l in LAMBDAS for y in TEST_YEARS for s in SEEDS if (run_dir(l, y, s) / "done.json").exists()]
    secs = [json.loads((run_dir(l, y, s) / "done.json").read_text(encoding="utf-8"))["seconds"] for l, y, s in done]
    total = len(LAMBDAS) * len(TEST_YEARS) * len(SEEDS)
    print(f"학습: {len(done)}/{total} 끝" + (f" (평균 {np.mean(secs):.0f}초/작업)" if secs else ""))
    for l in LAMBDAS:
        print(f"  λ {l:g}: " + ", ".join(f"{y} {sum((run_dir(l, y, s) / 'done.json').exists() for s in SEEDS)}/5" for y in TEST_YEARS))


# ---------------------------------------------------------------------------
# 운용 경로 (엔진 밖 장부로 E 결정) → C 엔진 평가
# ---------------------------------------------------------------------------

def load_policies(lam: float, seeds) -> tuple[dict, dict]:
    from stable_baselines3 import PPO

    pols, stats = {}, {}
    for y in TEST_YEARS:
        pols[y] = [PPO.load(run_dir(lam, y, s) / "final.zip", device="cpu") for s in seeds]
        st = json.loads((run_dir(lam, y, seeds[0]) / "stats.json").read_text(encoding="utf-8"))
        if st["cols"] != FEATS:
            raise RuntimeError(f"λ {lam:g} y{y}: 특성 열이 다르다")
        stats[y] = (np.array(st["mu"]), np.array(st["sd"]))
    return pols, stats


def live_path(data: Data, pols: dict, stats: dict) -> list[float]:
    """2016-01-08부터 리밸런싱일마다 E를 정한다. 정책이 없는 해(2016~2018)는 E = 0.466107. 지금 비중은 자기 장부로 잰다."""
    f = data.lv
    X = f[FEATS].to_numpy(dtype=float)
    yr = f["date"].dt.year.to_numpy()
    dec = f["k"].to_numpy(dtype=np.int64)
    Zy = {y: normalize(X, *stats[y]) for y in pols}
    book, es = Book(data.px, int(dec[0]), 0.0), []
    for k in range(len(dec)):
        y = int(yr[k])
        if y in pols:
            o = obs_vec(Zy[y], k, book)
            e = float(np.mean([LEVELS[int(p.predict(o, deterministic=True)[0])] for p in pols[y]]))
        else:
            e = E_C53
        es.append(e)
        j = int(dec[k + 1]) if k + 1 < len(dec) else len(data.px) - 1
        if j > book.i:
            book.run(j, e)
    return es


def full_es(es_reb: list[float], n_dates: int) -> list[float]:
    """리밸런싱일(짝수 번째 신호일) E → 신호일 전체 E. 사이 주 값은 엔진이 쓰지 않는다 (직전 리밸런싱 값을 적어 둔다)."""
    if len(es_reb) != (n_dates + 1) // 2:
        raise RuntimeError("리밸런싱일 수가 신호일 수와 맞지 않다")
    return [float(es_reb[i // 2]) for i in range(n_dates)]


def _save(name: str, metrics: dict, nav: pd.Series, es: list[float]) -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": nav.index, "nav": nav.to_numpy()}).to_parquet(EVAL / f"nav_{name}.parquet", index=False)
    (EVAL / f"{name}.json").write_text(json.dumps({**metrics, "es": es}, ensure_ascii=False, default=float), encoding="utf-8")


def _load(name: str):
    p = EVAL / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def period_stats(nav: pd.Series, y0: int, y1: int, fills: pd.DatetimeIndex) -> tuple[float, float]:
    start = fills[fills.year == y0][0]
    s = nav[(nav.index >= start) & (nav.index.year <= y1)]
    s = s / s.iloc[0]
    n = len(s) - 1
    return float(s.iloc[-1] ** (252 / n) - 1), float((1 - s / s.cummax()).max())


def evaluate() -> None:
    if not all_trained():
        print("아직 끝나지 않은 학습이 있다 (status로 확인)")
        sys.exit(1)
    t0 = time.time()
    data = Data()
    dates = signal_list()
    ctx = Ctx(ALL_YEARS)
    if list(ctx.dates) != dates:
        raise RuntimeError("Ctx 신호일이 리밸런싱 계산의 신호일과 다르다")
    snaps = ctx.snaps(signals=smooth_signals(ctx.signals, 4), key="sm4")
    b = replace(ctx.buffer, enter_rank_max=N, exit_rank_greater_than=N + BAND, max_holdings=N)
    k200 = ctx.inputs.bm_close.copy()
    k200.index = _ns(k200.index)
    cal = k200.index[k200.index <= pd.Timestamp("2025-12-30")]
    fills = pd.DatetimeIndex([cal[cal.searchsorted(d, side="right")] for d in _ns(dates)])
    k = pd.read_parquet(KOSPI_PATH)
    kospi = pd.Series(k["close"].to_numpy(dtype=float), index=_ns(k["date"]))
    yr_reb = data.lv["date"].dt.year.to_numpy()
    print(f"입력 읽음 ({time.time() - t0:.0f}초)", flush=True)

    def engine(name: str, es_reb: list[float]) -> dict:
        r = _load(name)
        if r is None:
            t = time.time()
            m, nav = run(ctx, snaps, b, EVERY, full_es(es_reb, len(dates)))
            nav = nav.copy()
            nav.index = _ns(nav.index)
            _save(name, m, nav, es_reb)
            r = _load(name)
            print(f"  {name}: CAGR {m['cagr']:+.2%}, MDD {m['mdd']:.1%}, 회전율 {m['annual_turnover']:.1f} ({time.time() - t:.0f}초)",
                  flush=True)
        return r

    # 대조: 고정 E = 0.466107 = 25절 T7-80-C53
    c53 = engine("c53", [E_C53] * len(data.lv))
    ref = json.loads((SAMECASH / "cash.json").read_text(encoding="utf-8"))
    d = abs(c53["cagr"] - ref["summary_cagr"])
    print(f"대조: E 고정 CAGR {c53['cagr']:+.6%} vs 25절 T7-80-C53 {ref['summary_cagr']:+.6%} (차이 {d:.1e})")
    if d > 1e-12:
        raise RuntimeError("E 고정 판이 25절 T7-80-C53과 다르다")

    paths = {}
    for lam in LAMBDAS:
        pols, stats = load_policies(lam, SEEDS)
        paths[f"{lam_id(lam)}_ens"] = live_path(data, pols, stats)
        for i, s in enumerate(SEEDS):
            paths[f"{lam_id(lam)}_s{s}"] = live_path(data, {y: [pols[y][i]] for y in TEST_YEARS}, stats)
    for name, es_reb in paths.items():
        engine(name, es_reb)
        mean_e = float(np.mean(np.array(es_reb)[yr_reb >= TEST_YEARS[0]]))
        engine(f"{name}_fixed", [E_C53 if y < TEST_YEARS[0] else mean_e for y in yr_reb])

    navs = {n: navp(EVAL / f"nav_{n}.parquet") for n in ["c53"] + [x for p in paths for x in (p, f"{p}_fixed")]}
    navs["e1"] = navp(SAMECASH / "nav_e1.parquet")
    navs["kospi"] = kospi[(kospi.index >= fills[0]) & (kospi.index <= pd.Timestamp("2025-12-30"))]
    navs["k200"] = k200[(k200.index >= fills[0]) & (k200.index <= pd.Timestamp("2025-12-30"))]

    def row(name: str) -> dict:
        nav = navs[name]
        out = {"name": name}
        for p, (y0, y1) in PERIODS.items():
            c, m_ = period_stats(nav, y0, y1, fills)
            out[f"cagr_{p}"], out[f"mdd_{p}"] = c, m_
        out["yearly"] = {int(y): float(v) for y, v in (lambda l: l / l.shift(1) - 1)(nav.groupby(nav.index.year).last()).items()
                         if y >= TEST_YEARS[0]}
        out["windows"] = {w: v for w, v in ex.windows(nav).items() if w not in ("2018 하락",)}
        r = _load(name)
        if r is not None:
            es_a = np.array(r["es"])[yr_reb >= TEST_YEARS[0]]
            out.update(mean_e=float(es_a.mean()), low=float((es_a <= 0.2).mean()), high=float((es_a >= 0.8).mean()),
                       switches=int((np.abs(np.diff(es_a)) > 1e-9).sum()), turnover10=r["annual_turnover"], cagr10=r["cagr"],
                       mdd10=r["mdd"])
            wd = {}
            for wname, a, z in ex.WINDOWS[1:]:
                msk = (data.lv["date"] >= pd.Timestamp(a)) & (data.lv["date"] <= pd.Timestamp(z))
                wd[wname] = float(np.mean(np.array(r["es"])[msk.to_numpy()])) if msk.any() else None
            out["window_mean_e"] = wd
        return out

    rows = {n: row(n) for n in navs}
    pick = max(LAMBDAS, key=lambda l: rows[f"{lam_id(l)}_ens"]["cagr_select"] / max(rows[f"{lam_id(l)}_ens"]["mdd_select"], 1e-9))
    P = lam_id(pick)
    ens, fx, base = rows[f"{P}_ens"], rows[f"{P}_ens_fixed"], rows["c53"]
    seeds_beat = int(sum(rows[f"{P}_s{s}"]["cagr_confirm"] > rows[f"{P}_s{s}_fixed"]["cagr_confirm"] for s in SEEDS))
    j = {"고른 λ (2019~2020 앙상블 CAGR/MDD 최고)": pick,
         "(1) 타이밍 실력 (2021~2025: 앙상블 CAGR·MDD 모두 같은 평균 고정보다 낫고 시드 4/5 이상)":
             bool(ens["cagr_confirm"] > fx["cagr_confirm"] and ens["mdd_confirm"] < fx["mdd_confirm"] and seeds_beat >= 4),
         "(2) 고정 모델 대비 (2021~2025: CAGR > T7-80-C53, MDD < T7-80-C53)":
             bool(ens["cagr_confirm"] > base["cagr_confirm"] and ens["mdd_confirm"] < base["mdd_confirm"]),
         "(3) 2019~2025 전체에서도 (2)": bool(ens["cagr_all"] > base["cagr_all"] and ens["mdd_all"] < base["mdd_all"]),
         "자기 고정보다 CAGR 높은 시드 수 (2021~2025)": seeds_beat}
    j["교체 후보"] = bool(j[[q for q in j if q.startswith("(1)")][0]] and j[[q for q in j if q.startswith("(2)")][0]]
                     and j[[q for q in j if q.startswith("(3)")][0]])
    flat = []
    for n, r in rows.items():
        f = {k_: v for k_, v in r.items() if not isinstance(v, dict)}
        f.update({f"y{y}": v for y, v in r["yearly"].items()})
        f.update({f"w:{w}": v for w, v in r["windows"].items()})
        f.update({f"e:{w}": v for w, v in r.get("window_mean_e", {}).items()})
        flat.append(f)
    rd = pd.DataFrame(flat)
    rd.to_csv(HERE / "t7b_runs.csv", index=False)
    (HERE / "t7b_judgment.json").write_text(json.dumps({"judgment": j, "rows": rows}, ensure_ascii=False, indent=1, default=float),
                                           encoding="utf-8")
    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("result", "t7b|판정"):
        for lam in LAMBDAS:
            r = rows[f"{lam_id(lam)}_ens"]
            ledger.result(f"t7b|{lam_id(lam)}", **{k_: r[k_] for k_ in ("cagr_select", "mdd_select", "cagr_confirm", "mdd_confirm",
                                                                       "cagr_all", "mdd_all", "mean_e", "turnover10")})
        ledger.result("t7b|판정", **j)

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    show = rd.copy()
    for c in [c for c in show.columns if c.startswith(("cagr", "mdd", "w:", "low", "high")) or (c.startswith("y") and c[1:].isdigit())]:
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(1)
    for c in ["mean_e", "turnover10"] + [c for c in show.columns if c.startswith("e:")]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    cols = ["name", "cagr_select", "mdd_select", "cagr_confirm", "mdd_confirm", "cagr_all", "mdd_all", "mean_e", "low", "high",
            "switches", "turnover10"]
    print("\n[기간별 CAGR·MDD %, 평균 E(2019~2025), E ≤ 0.2·≥ 0.8 비율 %, 바꾼 횟수, 10년 연 회전율]")
    print(show[[c for c in cols if c in show.columns]].to_string(index=False))
    print("\n[하락·상승 구간 수익률 %]")
    print(show[["name"] + [c for c in show.columns if c.startswith("w:")]].to_string(index=False))
    print("\n[하락·상승 구간 평균 E]")
    print(show[show["name"].str.contains("_ens$|_s\\d$", regex=True)][["name"] + [c for c in show.columns if c.startswith("e:")]]
          .to_string(index=False))
    print("\n[연도별 수익률 %]")
    print(show[["name"] + [c for c in show.columns if c.startswith("y") and c[1:].isdigit()]].to_string(index=False))
    print("\n[판정] (실행 전 고정)")
    for q, v in j.items():
        print(f"  {q}: {v if not isinstance(v, bool) else ('만족' if v else '미달')}")
    print("done")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "basket":
        basket()
    elif cmd == "prep":
        prep()
    elif cmd == "smoke":
        smoke()
    elif cmd == "jobs":
        write_jobs()
    elif cmd == "worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "status":
        status()
    elif cmd == "evaluate":
        evaluate()
    else:
        raise SystemExit(f"모르는 명령: {cmd} (basket | prep | smoke | jobs | worker <작업 파일> <슬롯> <슬롯 수> | status | evaluate)")


if __name__ == "__main__":
    main()
