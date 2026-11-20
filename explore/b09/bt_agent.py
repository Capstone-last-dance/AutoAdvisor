r"""바구니 타이밍: 지금 포트폴리오 종목 바구니의 투자 비중을 정하는 RL 에이전트 (11/19 사용자 결정).

질문: 투자 비중을 정할 때 코스피 200(지수) 대신 포트폴리오 종목 바구니 자체의 움직임을 보고 배우면
더 제때 줄이고 늘려서 수익이 늘어나나. 하락장이면 0%(전부 현금)도 된다. ETF는 쓰지 않는다 (최후의 수단).

실행 (저장소 루트, PowerShell). 모두 사용자가 돌린다.
  1) 바구니·특성 (한 번, 2~5분):  .\.venv\Scripts\python.exe -I explore\b09\run.py bt_agent.py basket
  2) 점검 (5분 안팎):             .\.venv\Scripts\python.exe -I explore\b09\run.py bt_agent.py smoke
  3) 학습 (작업 150개):           .\.venv\Scripts\python.exe -I explore\b09\run.py bt_agent.py jobs
                                  .\.venv\Scripts\python.exe -I explore\b09\run.py bt_agent.py worker explore\b09\jobs_bt.txt 0 4
                                  (창 4개에서 끝에서 두 번째 숫자 0·1·2·3). 진행: ... bt_agent.py status
  4) 평가 (C 엔진, 10분 안팎):     .\.venv\Scripts\python.exe -I explore\b09\run.py bt_agent.py evaluate
  끊기면 같은 명령을 다시 돌린다. 끝난 학습·평가는 건너뛴다.

바구니 (실행 전 고정 11/19)
- 버퍼 규칙과 같은 20종목 (부록 1 값): 신호일마다 보유 종목 중 rank ≤ 40은 유지하고, 빈자리는 rank ≤ 20 종목을 rank 순으로
  채운다 (최대 20). crash_flag = 1은 뺀다. 같은 비중, 다음 거래일 종가에 맞춘다. 수정종가(거래 없는 날은 직전 값).
  교체 비용 = 편도 회전율 × (2 × 한쪽 비용 0.115% + 매도세 0.20%). 엔진 밖 계산이라 종목·업종 한도와 M9 쿨다운은 없다.
- 사용자 제안은 '상위 30종목'이었다. 2016~2025 확인(스크래치)에서 버퍼 규칙 20종목이 실제 B 포트폴리오(M8·M9 끔)와
  주간 수익 상관 0.999 (CAGR −4.5% vs −4.3%), 매주 상위 30종목은 0.966·교체 주 56%라 버퍼 규칙 판을 쓴다.
- 학습 바구니 (폴드 Y): A 폴드 신호 20261117_A_rlsig_y{Y}의 rank (2011 ~ Y−1년 11~12월, A purge 기준, 교차 적합).
- 실전 바구니 (시험 특성·장부): 2015-12-04까지 rlsig_y2016, 2016-01-08부터 A v7 시험 신호(20261117_A_wfnews).
  2016-01-08에는 보유를 비우고 새로 고른다 (B 포트폴리오가 현금 100%에서 시작하는 것과 같게).

관측 (신호일 t 종가까지) — 세 판을 나란히 비교한다
- idx (지수만): mt_agent와 같은 시장 특성 14개 (코스피 200 9개 + 시가총액 상위 200 코스피 종목의 시장 폭 5개, mt_features.parquet)
- bkt (바구니만): 바구니 잔고의 5·21·63·126거래일 로그수익률, 20·60일 변동성, 50·200일선 대비 거리, 126일 고점 대비 낙폭,
  그날 고른 바구니 종목 중 자기 50·200일선 위 비율, 바구니 종목 20일 로그수익률의 흩어짐 (12개)
- both (둘 다): 26개
- 모든 판에 '지금 투자 비중' 1개를 더한다. 특성은 폴드 학습 신호일의 평균·표준편차로 표준화(±5).
- 학습 신호일은 세 판이 같다 (26개 특성이 모두 계산되는 첫 신호일부터).

학습 (실행 전 고정 11/19)
- 행동: 매주(신호일) 바구니 투자 비중 0·25·50·75·100% 중 하나. 신호일 종가에 정하고 다음 거래일 종가에 맞춘다.
- 보상 = 100 × [ln(에이전트 잔고 변화) − ln(바구니 100% 보유 변화)]: 100% 들고 있었을 때보다 더 번 만큼.
  비중을 바꿀 때 비용: 한쪽 0.115%, 줄일 때 매도세 0.20%. 현금 수익 0. 낙폭 벌점 없음.
- PPO(MlpPolicy, SB3 기본값), cpu, 스레드 1, 시드 0~4, 20,000스텝, 에피소드 52주(학습 구간 안 무작위 시작, 시작 비중 무작위).
- 체크포인트를 고르지 않는다: 20,000스텝 끝의 정책을 쓴다. (mt_agent는 검증 1년 최고점을 골라 한 해에 맞춘 정책이 됐고,
  그 최고점의 중앙값이 7,500~12,500스텝이었다. 끝까지 50,000스텝을 학습하면 학습 연도에 과적합될 수 있어 20,000으로 둔다.)
- Walk-forward: 시험 연도 Y = 2016~2025. 학습 = 폴드 Y 신호일 전부 (2011 ~ Y−1년). 신호일이 속한 해의 폴드 정책을 쓴다.
- 앙상블: 신호일마다 5시드 정책의 비중을 평균한다 (5% 단위). 앙상블과 시드는 각자 자기 장부(실전 바구니)로 '지금 비중'을 잰다.

평가 (B 포트폴리오, 2016~2025 연속, C 엔진, M9 포함)
- 에이전트 비중 E_t를 버퍼 + M8 Step 1~3 + M9 포트폴리오에 적용한다 (사전등록 E_t 대신, 하한 없음).
- 비교: B 100% (노출 조절 끔, M9 켬), 사전등록 M8, M8·M9 끔, 같은 평균 비중 고정 (앙상블·시드마다 자기 평균 E), 코스피 200(참고).
- 판정 (실행 전 고정 11/19). 판마다:
  (1) 타이밍 실력: 앙상블 CAGR > 같은 평균 비중 고정, 앙상블 MDD < 같은 평균 비중 고정, 시드 5개 중 4개 이상이 자기 고정보다 CAGR 높음.
  (2) 수익: 앙상블 CAGR > B 100%.
  (3) 가설 '자기 종목으로 배우면 낫다' (판 사이): bkt 앙상블 CAGR > idx 앙상블 CAGR, 그리고 bkt 시드 평균 CAGR > idx 시드 평균 CAGR.
  참고: 끝 잔고 > 1(원금보다 늘었나), MDD, IR, 연도별, B 100% 국면(θ=10%) 상승·하락 구간 평균 비중과 따라감, 바구니에 직접 적용(엔진 밖).
- 확인 대상: (1)·(2)를 만족한 판 중 앙상블 CAGR이 가장 높은 것. 2016~2025는 이미 여러 번 본 구간이고 판이 3개라,
  넘어도 처음 보는 데이터(forward test, PM이 허락하면 2026 홀드아웃)로 확인해야 한다.
- 탐색 결과다 (R-15). 결과: bt_summary.csv, bt_runs.csv, bt_judgment.json, runs/bt/eval/, 원장.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

from common import HERE, SIG_RUN, Ledger
import numpy as np
import pandas as pd
from src.common import io
from src.portfolio.env_check_report import kospi200
from src.portfolio.params import PREREG_PATH, load_prereg

RLSIG = "20261117_A_rlsig"  # 폴드 학습 신호 (B-07 공식 학습과 같다)
MT_FEAT = HERE / "mt_features.parquet"  # 지수·시장 폭 특성 (mt_agent.py features로 만든 것)
SERIES_PATH = HERE / "bt_series.parquet"
FEAT_PATH = HERE / "bt_features.parquet"
HOLD_PATH = HERE / "bt_holdings.parquet"
META_PATH = HERE / "bt_meta.json"
RUNS = HERE / "runs" / "bt"
EVAL = RUNS / "eval"
JOBS = HERE / "jobs_bt.txt"
OFFICIAL_NAV = HERE.parents[1] / "reports" / "20261118_B_buffer_wf" / "daily_nav.csv"
END = "2025-12-31"  # 2026 홀드아웃은 쓰지 않는다
LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
SEEDS = (0, 1, 2, 3, 4)
YEARS = tuple(range(2016, 2026))
STEPS, EP_LEN = 20_000, 52
IDX = ["r5", "r21", "r63", "r126", "vol20", "vol60", "ma50gap", "ma200gap", "dd126",
       "breadth50", "breadth200", "adv20", "disp20", "valtrend"]
BKT = ["b_r5", "b_r21", "b_r63", "b_r126", "b_vol20", "b_vol60", "b_ma50gap", "b_ma200gap", "b_dd126",
       "b_breadth50", "b_breadth200", "b_disp20"]
VARIANTS = {"idx": IDX, "bkt": BKT, "both": IDX + BKT}
LABEL = {"idx": "지수만", "bkt": "바구니만", "both": "둘 다"}
CLIP = 5.0
THETA = 0.10
BENCH = {"e1": ("B 100% (노출 조절 끔, M9 켬)", "no_exposure"), "m8": ("B + 사전등록 M8 + M9", "full"),
         "off": ("B, M8·M9 끔", "off")}

_cfg = load_prereg(PREREG_PATH)
assert float(_cfg["costs"]["cash_return"]) == 0.0, "부록 1 cash_return이 0이 아니다"
RATE = float(_cfg["costs"]["commission_per_side"]) + float(_cfg["costs"]["slippage_per_side"])
TAX = float(_cfg["costs"]["sell_tax"])
_buf = _cfg["portfolio"]["buffer_rule"]
ENTER, EXIT, HOLD_MAX = int(_buf["enter_rank_max"]), int(_buf["exit_rank_greater_than"]), int(_buf["max_holdings"])


def _ns(x) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(x)).as_unit("ns")


# ---------------------------------------------------------------------------
# 바구니와 특성
# ---------------------------------------------------------------------------

def rank_book(run_id: str) -> dict:
    """신호일 → (종목, rank, crash_flag) 배열 (rank 오름차순, 같으면 종목코드 순)."""
    s = io.read_table("signals", run_id=run_id)
    s = pd.DataFrame({"date": _ns(s["date"]), "ticker": s["ticker"].astype(str).to_numpy(),
                      "rank": s["rank"].to_numpy(dtype=np.int64), "flag": s["crash_flag"].to_numpy(dtype=np.int64)})
    s = s.sort_values(["date", "rank", "ticker"], kind="stable")
    return {pd.Timestamp(d): (g["ticker"].to_numpy(), g["rank"].to_numpy(), g["flag"].to_numpy())
            for d, g in s.groupby("date", sort=True)}


def choose(entry, held: set) -> list:
    """버퍼 규칙 단계 2~3 (쿨다운 없음): 유지 rank ≤ EXIT, 추가 rank ≤ ENTER, 최대 HOLD_MAX, crash_flag = 0."""
    tick, rank, flag = entry
    keep = [t for t, r, f in zip(tick, rank, flag) if f == 0 and t in held and r <= EXIT]
    room = max(HOLD_MAX - len(keep), 0)
    add = [t for t, r, f in zip(tick, rank, flag) if f == 0 and t not in held and r <= ENTER][:room]
    return keep + add


def simulate(days: pd.DatetimeIndex, R: np.ndarray, col: dict, decs: list, reset=None):
    """신호일 종가에 고르고 다음 거래일 종가에 같은 비중으로 맞추는 바구니의 일별 잔고 (시작 1, 첫 체결 전 NaN)."""
    n = len(days)
    fill = {}
    for d, entry in decs:
        k = int(days.searchsorted(d, side="right"))
        if k < n:
            fill[k] = (d, entry)
    nav = np.full(n, np.nan)
    held, idx, w = set(), np.zeros(0, dtype=np.int64), np.zeros(0)
    v, started, rec, turns, sizes, no_price = 1.0, False, {}, [], [], 0
    for k in range(n):
        if len(idx):
            r = R[k, idx]
            g = float(w @ r)
            v *= 1.0 + g
            w = w * (1.0 + r) / (1.0 + g)
        if k in fill:
            d, entry = fill[k]
            if reset is not None and d == reset:
                held = set()
            picked = choose(entry, held)
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
            rec[d] = new
            sizes.append(len(new))
        if started:
            nav[k] = v
    return nav, rec, turns, sizes, no_price


def basket_feats(days, nav, C, MA50, MA200, L20, col, rec, ddates) -> pd.DataFrame:
    s = pd.Series(np.log(nav), index=days)
    navs = pd.Series(nav, index=days)
    r1 = s.diff()
    f = pd.DataFrame(index=days)
    for k in (5, 21, 63, 126):
        f[f"b_r{k}"] = s - s.shift(k)
    f["b_vol20"] = r1.rolling(20, min_periods=20).std() * math.sqrt(252)
    f["b_vol60"] = r1.rolling(60, min_periods=60).std() * math.sqrt(252)
    f["b_ma50gap"] = s - np.log(navs.rolling(50, min_periods=50).mean())
    f["b_ma200gap"] = s - np.log(navs.rolling(200, min_periods=200).mean())
    f["b_dd126"] = s - np.log(navs.rolling(126, min_periods=126).max())
    ix = _ns(ddates)
    out = f.reindex(ix)
    b50, b200, dsp, last = [], [], [], []
    for d in ix:
        k = days.get_loc(d)
        ts = rec.get(d, last)  # 그날 고른 바구니 (체결일이 달력 밖이면 직전 바구니)
        last = ts
        j = np.array([col[t] for t in ts if t in col], dtype=np.int64)
        c, m50, m200, l20 = C[k, j], MA50[k, j], MA200[k, j], L20[k, j]
        v50, v200, vl = np.isfinite(c) & np.isfinite(m50), np.isfinite(c) & np.isfinite(m200), np.isfinite(l20)
        b50.append(float((c[v50] > m50[v50]).mean()) if v50.any() else np.nan)
        b200.append(float((c[v200] > m200[v200]).mean()) if v200.any() else np.nan)
        dsp.append(float(np.std(l20[vl], ddof=1)) if vl.sum() >= 2 else np.nan)
    out["b_breadth50"], out["b_breadth200"], out["b_disp20"] = b50, b200, dsp
    out.index.name = "date"
    return out


def _cagr_mdd(nav: pd.Series) -> tuple[float, float]:
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / yrs) - 1.0), float((1.0 - nav / nav.cummax()).max())


def build_basket() -> None:
    import pyarrow.parquet as pq

    t0 = time.time()
    days = _ns(kospi200(io.read_table("index")).index)
    days = days[days <= pd.Timestamp(END)]
    books = {y: rank_book(f"{RLSIG}_y{y}") for y in YEARS}
    test = rank_book(SIG_RUN)
    need = set()
    for b in list(books.values()) + [test]:
        for tick, rank, _ in b.values():
            need.update(tick[rank <= EXIT].tolist())
    tbl = pq.read_table(io.ROOT / "data" / "processed" / "prices.parquet", columns=["date", "ticker", "close"],
                        filters=[("ticker", "in", sorted(need)), ("date", ">=", days[0]), ("date", "<=", days[-1])])
    px = tbl.to_pandas()
    del tbl
    px["date"] = _ns(px["date"])
    px["ticker"] = px["ticker"].astype(str)
    close = px.pivot(index="date", columns="ticker", values="close").reindex(days).astype(float)
    del px
    close = close.where(close > 0).ffill()
    R = close.pct_change(fill_method=None).fillna(0.0).to_numpy()
    col = {t: k for k, t in enumerate(close.columns)}
    C = close.to_numpy()
    MA50 = close.rolling(50, min_periods=50).mean().to_numpy()
    MA200 = close.rolling(200, min_periods=200).mean().to_numpy()
    L20 = np.log(close / close.shift(20)).to_numpy()
    print(f"가격 읽음: 종목 {close.shape[1]}, 거래일 {len(days)} ({time.time() - t0:.0f}초)", flush=True)
    mt = pd.read_parquet(MT_FEAT)
    mt.index = _ns(mt["date"])
    mt = mt[IDX]

    first_test = min(test)
    specs = {f"f{y}": (sorted(books[y].items()), None) for y in YEARS}
    specs["live"] = (sorted([(d, e) for d, e in books[2016].items() if d < first_test] + list(test.items())), first_test)
    wide = pd.DataFrame(index=days)
    feats, meta_series = [], {}
    for name, (decs, reset) in specs.items():
        nav, rec, turns, sizes, no_price = simulate(days, R, col, decs, reset)
        wide[name] = nav
        fb = basket_feats(days, nav, C, MA50, MA200, L20, col, rec, [d for d, _ in decs])
        fr = pd.concat([mt.reindex(fb.index), fb], axis=1)
        ok = fr[IDX + BKT].notna().all(axis=1).to_numpy()
        fr.insert(0, "series", name)
        if name == "live":
            fr["test"] = fr.index >= first_test
            fr["train"] = False
            if not ok[fr["test"].to_numpy()].all():
                raise RuntimeError("실전 바구니의 시험 신호일에 비어 있는 특성이 있다")
        else:
            fr["train"] = ok
            fr["test"] = False
        feats.append(fr.reset_index())
        navs = pd.Series(nav, index=days)
        if name == "live":
            lo, hi = pd.Timestamp("2016-01-11"), days[-1]
            c, m = _cagr_mdd(navs[(navs.index >= lo) & (navs.index <= hi)])
            seg = "2016-01-11 ~ " + str(hi.date())
            holds = pd.DataFrame([(d, t) for d, ts in rec.items() if d >= first_test for t in ts], columns=["date", "ticker"])
        else:
            tr = fr.index[ok]
            c, m = _cagr_mdd(navs[(navs.index >= tr[0]) & (navs.index <= tr[-1])])
            seg = f"{tr[0].date()} ~ {tr[-1].date()} (학습 신호일 {int(ok.sum())}개)"
        meta_series[name] = {"decisions": len(decs), "first": str(decs[0][0].date()), "last": str(decs[-1][0].date()),
                             "segment": seg, "cagr": c, "mdd": m, "weekly_turnover": float(np.mean(turns)),
                             "mean_holdings": float(np.mean(sizes)), "picked_without_price": int(no_price)}
    wide.index.name = "date"
    wide.reset_index().to_parquet(SERIES_PATH, index=False)
    pd.concat(feats, ignore_index=True).to_parquet(FEAT_PATH, index=False)
    holds.to_parquet(HOLD_PATH, index=False)
    meta = {"days": [str(days[0].date()), str(days[-1].date())], "tickers": int(close.shape[1]), "rule": {
        "enter": ENTER, "exit": EXIT, "max": HOLD_MAX, "rate": RATE, "tax": TAX}, "series": meta_series,
        "seconds": round(time.time() - t0, 1)}
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'바구니':>6} | {'구간':<44} | {'CAGR':>7} | {'MDD':>6} | 주간 교체 | 평균 종목")
    for k, v in meta_series.items():
        print(f"{k:>6} | {v['segment']:<44} | {v['cagr']:+7.1%} | {v['mdd']:6.1%} | {v['weekly_turnover']:9.0%} | {v['mean_holdings']:.1f}")
    print(f"가격 없는 종목을 고른 횟수 (빼고 계산): {sum(v['picked_without_price'] for v in meta_series.values())}")
    print("done")


class Data:
    """bt_series(달력 × 바구니 잔고), bt_features(바구니 × 신호일 특성)."""

    def __init__(self) -> None:
        w = pd.read_parquet(SERIES_PATH)
        self.days = _ns(w["date"])
        self.nav = {c: w[c].to_numpy(dtype=float) for c in w.columns if c != "date"}
        f = pd.read_parquet(FEAT_PATH)
        f["date"] = _ns(f["date"])
        self.f = f
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    def _rows(self, series: str, flag: str):
        g = self.f[(self.f["series"] == series) & self.f[flag]].sort_values("date").reset_index(drop=True)
        dec = self.days.get_indexer(_ns(g["date"]))
        if (dec < 0).any() or not np.isfinite(self.nav[series][dec]).all():
            raise RuntimeError(f"{series}: 신호일이 달력에 없거나 잔고가 비었다")
        return self.nav[series], dec, g

    def fold(self, y: int):
        return self._rows(f"f{y}", "train")

    def live(self):
        return self._rows("live", "test")


def norm_stats(X: np.ndarray):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def normalize(X: np.ndarray, mu, sd) -> np.ndarray:
    return np.clip((X - np.asarray(mu)) / np.asarray(sd), -CLIP, CLIP)


def obs_vec(Z: np.ndarray, p: int, weight: float) -> np.ndarray:
    return np.append(Z[p], weight).astype(np.float32)


# ---------------------------------------------------------------------------
# 장부와 학습 환경
# ---------------------------------------------------------------------------

class Book:
    """바구니(잔고 px) + 현금. 신호일 종가에 정한 비중을 다음 거래일 종가에 맞춘다 (현금 수익 0, 줄일 때 매도세)."""

    __slots__ = ("px", "i", "units", "cash", "nav", "turn")

    def __init__(self, px: np.ndarray, i0: int, w0: float = 0.0) -> None:
        self.px, self.i = px, int(i0)
        self.nav, self.turn = 1.0, 0.0
        self.units = w0 / px[self.i]
        self.cash = 1.0 - w0

    def weight(self) -> float:
        return self.units * self.px[self.i] / self.nav

    def run(self, j: int, target: float, rec: list | None = None) -> float:
        """i 종가에서 정한 target을 i+1 종가에 체결하고 j 종가까지 간다. 반환: ln 잔고 변화."""
        nav0, first = self.nav, self.i + 1
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
            if rec is not None:
                rec.append((d, nav))
        self.i = int(j)
        return math.log(self.nav / nav0)


def play(px: np.ndarray, dec: np.ndarray, act, w0: float = 0.0, finish: int | None = None, rec: list | None = None):
    """dec[0]부터 신호일 k마다 act(k, book)으로 비중을 정한다. finish가 있으면 마지막 결정도 그날까지 간다.
    반환: (E 목록, 초과 보상 목록, book)."""
    book = Book(px, int(dec[0]), w0)
    es, rw = [], []
    for k in range(len(dec)):
        e = float(act(k, book))
        es.append(e)
        j = int(dec[k + 1]) if k + 1 < len(dec) else finish
        if j is None or j <= book.i:
            break
        i = book.i
        lr = book.run(j, e, rec)
        rw.append(100.0 * (lr - math.log(px[j] / px[i])))
    return es, rw, book


def make_env_class():
    import gymnasium as gym
    from gymnasium import spaces

    class BasketEnv(gym.Env):
        """학습 신호일 안에서 무작위 시작, EP_LEN 스텝 뒤 끝. 보상 = 100% 보유 대비 초과 로그수익 × 100."""

        def __init__(self, px, Z, dec):
            super().__init__()
            self.px, self.Z, self.dec = px, Z, dec
            if len(dec) < EP_LEN + 1:
                raise ValueError(f"학습 신호일 {len(dec)}개 < 에피소드 {EP_LEN} + 1")
            self.observation_space = spaces.Box(-np.inf, np.inf, (Z.shape[1] + 1,), np.float32)
            self.action_space = spaces.Discrete(len(LEVELS))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.p = int(self.np_random.integers(0, len(self.dec) - EP_LEN))
            w0 = float(LEVELS[int(self.np_random.integers(0, len(LEVELS)))])
            self.t = 0
            self.book = Book(self.px, int(self.dec[self.p]), w0)
            return obs_vec(self.Z, self.p, self.book.weight()), {}

        def step(self, action):
            i, j = int(self.dec[self.p]), int(self.dec[self.p + 1])
            lr = self.book.run(j, float(LEVELS[int(action)]))
            reward = 100.0 * (lr - math.log(self.px[j] / self.px[i]))
            self.p += 1
            self.t += 1
            truncated = self.t >= EP_LEN or self.p + 1 >= len(self.dec)
            return obs_vec(self.Z, self.p, self.book.weight()), float(reward), False, truncated, {}

    return BasketEnv


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def run_dir(v: str, y: int, seed: int, root: Path = RUNS) -> Path:
    return root / v / f"y{y}" / f"s{seed}"


def train_job(data: Data, v: str, y: int, seed: int, out: Path, steps: int = STEPS) -> dict:
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    px, dec, g = data.fold(y)
    X = g[VARIANTS[v]].to_numpy(dtype=float)
    mu, sd = norm_stats(X)
    Z = normalize(X, mu, sd)
    env = make_env_class()(px, Z, dec)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0)
    model.learn(total_timesteps=int(steps))
    model.save(out / "final.zip")

    def act(k, b):
        return float(LEVELS[int(model.predict(obs_vec(Z, k, b.weight()), deterministic=True)[0])])

    es, rw, _ = play(px, dec, act)  # 학습 구간 그대로 운용 (진단용, 고르는 데 쓰지 않는다)
    yr = g["date"].dt.year.to_numpy()
    last = yr[:-1] == y - 1
    (out / "stats.json").write_text(json.dumps({"cols": VARIANTS[v], "mu": list(map(float, mu)), "sd": list(map(float, sd)),
                                                "train": [str(g['date'].iloc[0].date()), str(g['date'].iloc[-1].date())]}),
                                    encoding="utf-8")
    es_a = np.array(es)
    res = {"variant": v, "year": y, "seed": seed, "steps": int(steps), "train_rows": len(dec),
           "insample_excess": float(np.sum(rw)), "insample_mean_e": float(es_a.mean()),
           "insample_e0": float((es_a == 0).mean()), "insample_e1": float((es_a == 1).mean()),
           "insample_switches": int((np.diff(es_a) != 0).sum()),
           "last_year_excess": float(np.sum(np.array(rw)[last])) if last.any() else None,
           "seconds": round(time.time() - t0, 1), "sb3": stable_baselines3.__version__, "torch": torch.__version__}
    (out / "done.json").write_text(json.dumps(res), encoding="utf-8")  # 마지막에 쓴다 (없으면 다시 학습)
    return res


def write_jobs() -> None:
    lines = [f"{v} {y} {s}" for y in YEARS for v in VARIANTS for s in SEEDS]
    JOBS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger = Ledger(HERE / "ledger.jsonl")
    for v, cols in VARIANTS.items():
        ledger.plan(f"bt|{v}", "바구니 타이밍 (포트폴리오 종목으로 배우는 비중 에이전트)",
                    f"관측 {LABEL[v]} ({len(cols)}개 + 지금 비중). 주간 E_t 0~100% 5단계, 보상 = 바구니 100% 보유 대비 초과 로그수익, "
                    f"PPO {STEPS:,}스텝 끝 정책, 5시드 앙상블. 학습 = 버퍼 규칙 20종목 바구니(폴드 rlsig), "
                    "평가 = 버퍼 + M8 Step 1~3 + M9 + C 엔진에 E_t 적용, walk-forward 2016~2025",
                    "사용자 결정(11/19): ETF는 최후의 수단, 지금 포트폴리오 종목에 집중. 지수 대신 포트폴리오 종목으로 배우면 "
                    "제때 줄이고 늘려 수익이 늘어나는지. 하락장이면 0%도 된다")
    print(f"{JOBS}: 작업 {len(lines)}개 (판 {len(VARIANTS)} × 시험 연도 {len(YEARS)} × 시드 {len(SEEDS)})")


def worker(jobs_path: str, slot: int, nslots: int) -> None:
    data = Data()
    jobs = [l.split() for l in Path(jobs_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    mine = [j for k, j in enumerate(jobs) if k % nslots == slot]
    print(f"워커 {slot}/{nslots}: 작업 {len(mine)}개", flush=True)
    for n, (v, y, seed) in enumerate(mine, 1):
        out = run_dir(v, int(y), int(seed))
        if (out / "done.json").exists():
            continue
        try:
            r = train_job(data, v, int(y), int(seed), out)
            print(f"[{n}/{len(mine)}] " + json.dumps(r, ensure_ascii=False, default=float), flush=True)
        except Exception as e:  # 다음 작업으로 간다. 다시 돌리면 이 작업을 다시 시도한다
            print(f"[{n}/{len(mine)}] " + json.dumps({"job": [v, y, seed], "error": repr(e)}, ensure_ascii=False), flush=True)
    print("done", flush=True)


def trained(v: str, seed: int) -> bool:
    return all((run_dir(v, y, seed) / "done.json").exists() for y in YEARS)


def status() -> None:
    n = sum((run_dir(v, y, s) / "done.json").exists() for v in VARIANTS for y in YEARS for s in SEEDS)
    secs = [json.loads((run_dir(v, y, s) / "done.json").read_text(encoding="utf-8"))["seconds"]
            for v in VARIANTS for y in YEARS for s in SEEDS if (run_dir(v, y, s) / "done.json").exists()]
    total = len(VARIANTS) * len(YEARS) * len(SEEDS)
    print(f"학습: {n}/{total} 끝" + (f" (평균 {np.mean(secs):.0f}초/작업)" if secs else ""))
    for v in VARIANTS:
        print(f"  {v}: " + ", ".join(f"시드 {s} {sum((run_dir(v, y, s) / 'done.json').exists() for y in YEARS)}/10" for s in SEEDS))
    names = [f"bench_{k}" for k in BENCH] + [f"{v}_{t}" for v in VARIANTS for t in [f"s{s}" for s in SEEDS] + ["ens"]]
    print(f"평가: {sum((EVAL / f'{n_}.json').exists() for n_ in names)}/{len(names)} (고정 비중 비교 제외)")


# ---------------------------------------------------------------------------
# 정책 → 실전 바구니 장부 → E_t
# ---------------------------------------------------------------------------

def load_policies(v: str, seeds) -> tuple[dict, dict]:
    from stable_baselines3 import PPO

    pols, stats = {}, {}
    for y in YEARS:
        pols[y] = [PPO.load(run_dir(v, y, s) / "final.zip", device="cpu") for s in seeds]
        st = json.loads((run_dir(v, y, seeds[0]) / "stats.json").read_text(encoding="utf-8"))
        if st["cols"] != VARIANTS[v]:
            raise RuntimeError(f"{v} y{y}: 특성 열이 다르다")
        stats[y] = (np.array(st["mu"]), np.array(st["sd"]))
    return pols, stats


def test_path(data: Data, v: str, pols: dict, stats: dict) -> tuple[list, list]:
    """실전 바구니로 시험 신호일마다 그해 폴드 정책(들)의 비중 평균을 정한다. 첫 신호일은 현금 100%(지금 비중 0)."""
    px, dec, g = data.live()
    X = g[VARIANTS[v]].to_numpy(dtype=float)
    Zy = {y: normalize(X, *stats[y]) for y in YEARS}
    yr = g["date"].dt.year.to_numpy()

    def act(k, book):
        o = obs_vec(Zy[int(yr[k])], k, book.weight())
        return float(np.mean([LEVELS[int(p.predict(o, deterministic=True)[0])] for p in pols[int(yr[k])]]))

    rec = []
    es, _, _ = play(px, dec, act, 0.0, finish=len(data.days) - 1, rec=rec)
    return es, rec


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
    clean = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v)
             for k, v in s.items()}
    return clean, rep.nav, pos


def _save(name: str, metrics: dict, nav: pd.Series, extra: dict | None = None, pos: pd.DataFrame | None = None) -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": nav.index, "nav": nav.to_numpy()}).to_parquet(EVAL / f"nav_{name}.parquet", index=False)
    if pos is not None:
        pos.to_parquet(EVAL / f"pos_{name}.parquet", index=False)
    (EVAL / f"{name}.json").write_text(json.dumps({**metrics, **(extra or {})}, ensure_ascii=False, default=float),
                                       encoding="utf-8")


def _done(name: str) -> bool:
    return (EVAL / f"{name}.json").exists()


# ---------------------------------------------------------------------------
# 점검
# ---------------------------------------------------------------------------

def smoke() -> None:
    from common import Ctx, buffer_run

    data = Data()
    for k, v in data.meta["series"].items():
        print(f"  바구니 {k}: {v['segment']}, CAGR {v['cagr']:+.1%}, MDD {v['mdd']:.1%}, 주간 교체 {v['weekly_turnover']:.0%}")

    # 대조 1: 실전 바구니가 공식 B 포트폴리오(M8·M9 끔)와 같은 종목을 고르고 비슷하게 움직이나
    off = pd.read_parquet(HERE.parents[1] / "positions" / "wf_20261118_B_buffer_wf_off.parquet")
    off = off[(off["ticker"] != "CASH") & (off["target_weight"] > 0)]
    b_set = {pd.Timestamp(d): set(g["ticker"].astype(str)) for d, g in off.groupby(_ns(off["date"]))}
    h = pd.read_parquet(HOLD_PATH)
    l_set = {pd.Timestamp(d): set(g["ticker"].astype(str)) for d, g in h.groupby(_ns(h["date"]))}
    common_d = sorted(set(b_set) & set(l_set))
    same = np.mean([b_set[d] == l_set[d] for d in common_d])
    jac = np.mean([len(b_set[d] & l_set[d]) / max(len(b_set[d] | l_set[d]), 1) for d in common_d])
    nav_o = pd.read_csv(OFFICIAL_NAV, parse_dates=["date"]).set_index("date")["buffer_off"]
    nav_o.index = _ns(nav_o.index)
    live = pd.Series(data.nav["live"], index=data.days).reindex(nav_o.index)
    wk = lambda s: s.resample("W-FRI").last().pct_change().dropna()  # noqa: E731
    corr = pd.concat([wk(live), wk(nav_o)], axis=1).dropna().corr().iloc[0, 1]
    c_l, m_l = _cagr_mdd(live)
    c_o, m_o = _cagr_mdd(nav_o)
    print(f"대조 1: 실전 바구니 vs 공식 B(M8·M9 끔, C 엔진) 2016~2025 — 같은 종목 집합인 신호일 {same:.1%}, 평균 겹침 {jac:.1%}, "
          f"주간 수익 상관 {corr:.4f}, CAGR {c_l:+.2%} vs {c_o:+.2%}, MDD {m_l:.1%} vs {m_o:.1%}")
    if corr < 0.98:
        raise RuntimeError("실전 바구니가 공식 B 포트폴리오와 너무 다르다")

    # 대조 2: 처음부터 100% 들고 계속 100%면 매매가 없어 초과 보상이 0이다
    px, dec, g = data.fold(2016)
    _, rw, _ = play(px, dec, lambda k, b: 1.0, w0=1.0)
    print(f"대조 2: 폴드 2016 바구니 100% 보유의 초과 보상 합 {sum(rw):.2e}")
    if abs(sum(rw)) > 1e-9:
        raise RuntimeError("100% 보유인데 초과 보상이 0이 아니다")

    # 대조 3: 엔진 경로. E = 1로 바꿔 넣으면 노출 조절 끔과 같고, E = 0이면 현금 100%다 (2016)
    t1 = time.time()
    ctx = Ctx((2016,))
    snaps = ctx.snaps()
    t_load = time.time() - t1
    t2 = time.time()
    a = buffer_run(ctx, snaps, mode="no_exposure")
    b = buffer_run(ctx, with_exposure(snaps, [1.0] * len(snaps)), mode="full")
    if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
        raise RuntimeError("E = 1로 바꾼 positions가 노출 조절 끔과 다르다")
    z = buffer_run(ctx, with_exposure(snaps, [0.0] * len(snaps)), mode="full")
    if set(z["ticker"]) != {"CASH"} or not np.allclose(z["target_weight"], 1.0):
        raise RuntimeError("E = 0인데 종목을 산다")
    t_run = (time.time() - t2) / 3 * 2  # positions 만들기 + 엔진 재생 ≈ 2배
    print("대조 3: E = 1 → 노출 조절 끔과 같은 positions, E = 0 → 현금 100% (2016)")

    t3 = time.time()
    r = train_job(data, "both", 2016, 0, RUNS / "_smoke" / "both" / "y2016" / "s0", steps=4_096)
    per_step = (time.time() - t3) / 4_096
    print(f"짧은 학습 (both, 폴드 2016, 시드 0, 4,096스텝): {json.dumps(r, ensure_ascii=False, default=float)}")
    pd.set_option("display.width", 250)
    print("폴드 2016 학습 신호일 특성 요약:")
    print(g[IDX + BKT].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    total = per_step * STEPS * len(VARIANTS) * len(YEARS) * len(SEEDS)
    n_runs = len(BENCH) + len(VARIANTS) * (len(SEEDS) + 1) * 2
    print(f"예상 시간: 학습 워커 4개면 약 {total / 3600 / 4 * 1.2:.1f}시간 (1개면 {total / 3600 * 1.2:.1f}시간), "
          f"평가 약 {(n_runs * t_run * len(YEARS) + t_load * 3) / 60:.0f}분 (1년 실행으로 잰 대략값)")
    print("smoke ok")


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def zigzag(s: pd.Series, theta: float) -> list:
    """θ 이상 움직인 고점·저점을 번갈아 찾는다 (사후 구분, 평가 전용, safe_phases.py와 같다)."""
    piv, mode = [], None
    hi = lo = float(s.iloc[0])
    hi_d = lo_d = s.index[0]
    for d, p in s.items():
        p = float(p)
        if mode in (None, "up"):
            if p > hi:
                hi, hi_d = p, d
            if p <= hi * (1.0 - theta):
                piv.append((hi_d, "peak"))
                mode, lo, lo_d = "down", p, d
                continue
        if mode in (None, "down"):
            if p < lo:
                lo, lo_d = p, d
            if p >= lo * (1.0 + theta):
                piv.append((lo_d, "trough"))
                mode, hi, hi_d = "up", p, d
    return piv


def phases(s: pd.Series, theta: float) -> list:
    piv = zigzag(s, theta)
    bounds = [(s.index[0], None)] + piv + [(s.index[-1], None)]
    segs = []
    for k in range(len(bounds) - 1):
        (a, _), (b, kind_b) = bounds[k], bounds[k + 1]
        if a == b:
            continue
        partial = k == 0 or k == len(bounds) - 2
        kind = "상승" if kind_b == "peak" else "하락" if kind_b == "trough" else ("하락" if bounds[k][1] == "peak" else "상승")
        segs.append({"start": a, "end": b, "kind": kind, "partial": partial})
    return segs


def yearly(nav: pd.Series) -> dict:
    last = nav.groupby(nav.index.year).last()
    return {int(y): float(v) for y, v in (last / last.shift(1).fillna(1.0) - 1.0).items()}


def evaluate() -> None:
    from common import ALL_YEARS, Ctx
    from src.portfolio.buffer_wf_report import benchmark_summary

    import torch
    torch.set_num_threads(1)
    missing = [(v, y, s) for v in VARIANTS for y in YEARS for s in SEEDS if not (run_dir(v, y, s) / "done.json").exists()]
    if missing:
        raise SystemExit(f"학습이 덜 끝났다 ({len(missing)}개, 예: {missing[:3]}). worker를 모두 끝낸 뒤 다시 돌린다")
    data = Data()
    _, _, g_live = data.live()
    t = time.time()
    ctx = Ctx(ALL_YEARS)
    snaps = ctx.snaps()
    if list(_ns(ctx.dates)) != list(_ns(g_live["date"])):
        raise RuntimeError("실전 바구니 신호일과 평가 신호일이 다르다")
    print(f"포트폴리오 입력 읽음 ({time.time() - t:.0f}초)", flush=True)

    for key, (label, mode) in BENCH.items():
        if _done(f"bench_{key}"):
            continue
        t = time.time()
        m, nav, pos = portfolio_eval(ctx, snaps, mode)
        _save(f"bench_{key}", {**m, "label": label}, nav)
        print(f"  기준 {label}: CAGR {m['cagr']:+.2%}, MDD {m['mdd']:.1%} ({time.time() - t:.0f}초)", flush=True)
        if key == "e1":
            k = benchmark_summary(ctx.inputs.bm_close, nav.index)
            (EVAL / "bench_kospi200.json").write_text(json.dumps({"cagr": float(k["cagr"]), "mdd": float(k["mdd"]),
                                                                  "label": "코스피 200 (참고, 사지 않음)"}), encoding="utf-8")

    for v in VARIANTS:
        runs = [(f"{v}_s{s}", (s,)) for s in SEEDS] + [(f"{v}_ens", SEEDS)]
        if all(_done(n) and _done(f"{n}_fixed") for n, _ in runs):
            continue
        t = time.time()
        pols_all, stats = load_policies(v, SEEDS)
        print(f"  {v}: 정책 {len(SEEDS) * len(YEARS)}개 읽음 ({time.time() - t:.0f}초)", flush=True)
        for name, seeds in runs:
            if not _done(name):
                t = time.time()
                pols = {y: [pols_all[y][SEEDS.index(s)] for s in seeds] for y in YEARS}
                es, rec = test_path(data, v, pols, stats)
                m, nav, pos = portfolio_eval(ctx, snaps, "full", es)
                bk = pd.Series([n_ for _, n_ in rec], index=data.days[[d for d, _ in rec]])
                bk = bk[(bk.index >= nav.index[0]) & (bk.index <= nav.index[-1])]
                b_c, b_m = _cagr_mdd(pd.concat([pd.Series([1.0], index=[nav.index[0] - pd.Timedelta(days=1)]), bk]))
                _save(name, m, nav, {"variant": v, "seeds": list(seeds), "e_path": es, "dates": list(ctx.dates),
                                     "basket_cagr": b_c, "basket_mdd": b_m}, pos if name.endswith("_ens") else None)
                print(f"  {name}: CAGR {m['cagr']:+.2%}, MDD {m['mdd']:.1%}, 평균 E {np.mean(es):.0%} "
                      f"(바구니에 직접 {b_c:+.2%}) ({time.time() - t:.0f}초)", flush=True)
            if not _done(f"{name}_fixed"):
                t = time.time()
                w = float(np.mean(json.loads((EVAL / f"{name}.json").read_text(encoding="utf-8"))["e_path"]))
                m, nav, _ = portfolio_eval(ctx, snaps, "full", [w] * len(snaps))
                _save(f"{name}_fixed", m, nav, {"w": w})
                print(f"  {name}_fixed (고정 {w:.1%}): CAGR {m['cagr']:+.2%}, MDD {m['mdd']:.1%} ({time.time() - t:.0f}초)", flush=True)
    report()


def report() -> None:
    def load(name):
        return json.loads((EVAL / f"{name}.json").read_text(encoding="utf-8"))

    def nav_of(name):
        f = pd.read_parquet(EVAL / f"nav_{name}.parquet")
        return pd.Series(f["nav"].to_numpy(), index=_ns(f["date"]))

    bench = {k: load(f"bench_{k}") for k in BENCH}
    kospi = load("bench_kospi200")
    b100 = nav_of("bench_e1")
    segs = [s for s in phases(b100, THETA) if not s["partial"]]
    up_mask = pd.Series(False, index=b100.index)
    dn_mask = pd.Series(False, index=b100.index)
    for s in segs:
        (up_mask if s["kind"] == "상승" else dn_mask).loc[s["start"]:s["end"]] = True

    def capture(nav):
        out = {}
        for kind in ("상승", "하락"):
            sm = sum(math.log(nav[s["end"]] / nav[s["start"]]) for s in segs if s["kind"] == kind)
            sb = sum(math.log(b100[s["end"]] / b100[s["start"]]) for s in segs if s["kind"] == kind)
            out[kind] = sm / sb if sb else float("nan")
        return out["상승"], out["하락"]

    rows = []
    for v in VARIANTS:
        for name, kind, seed in [(f"{v}_s{s}", "시드", s) for s in SEEDS] + [(f"{v}_ens", "앙상블", None)]:
            a, fx, nav = load(name), load(f"{name}_fixed"), nav_of(name)
            e = pd.Series(a["e_path"], index=_ns(a["dates"])).reindex(nav.index, method="ffill")
            um, dm = up_mask.reindex(e.index, fill_value=False), dn_mask.reindex(e.index, fill_value=False)
            up, dn = capture(nav)
            rows.append({"variant": v, "kind": kind, "seed": seed, **{k: a.get(k) for k in (
                "cagr", "mdd", "ir", "annual_turnover", "mean_cash", "final_nav")}, "mean_e": float(np.mean(a["e_path"])),
                "fixed_w": fx["w"], "fixed_cagr": fx["cagr"], "fixed_mdd": fx["mdd"],
                "e_up": float(e[um].mean()), "e_down": float(e[dm].mean()), "up_cap": up, "dn_cap": dn,
                "basket_cagr": a["basket_cagr"], "basket_mdd": a["basket_mdd"],
                **{f"y{y}": r for y, r in yearly(nav).items()}})
    rd = pd.DataFrame(rows)
    rd.to_csv(HERE / "bt_runs.csv", index=False)

    e1 = bench["e1"]
    judg, summ = {}, []
    for k, (label, _) in BENCH.items():
        up, dn = capture(nav_of(f"bench_{k}"))
        summ.append({"model": label, **{c: bench[k].get(c) for c in ("cagr", "mdd", "ir", "annual_turnover", "mean_cash",
                                                                     "final_nav")}, "up_cap": up, "dn_cap": dn})
    summ.append({"model": kospi["label"], "cagr": kospi["cagr"], "mdd": kospi["mdd"]})
    for v in VARIANTS:
        ens = rd[(rd.variant == v) & (rd.kind == "앙상블")].iloc[0]
        sd = rd[(rd.variant == v) & (rd.kind == "시드")]
        beat = int((sd.cagr > sd.fixed_cagr).sum())
        j = {"(1) 타이밍 실력 (같은 평균 비중 고정 대비)": bool(ens.cagr > ens.fixed_cagr and ens.mdd < ens.fixed_mdd and beat >= 4),
             "(2) 수익 (B 100% 대비)": bool(ens.cagr > e1["cagr"])}
        judg[v] = {**j, "seeds_cagr_above_own_fixed": beat, "ens_cagr": float(ens.cagr), "ens_mdd": float(ens.mdd),
                   "ens_fixed_w": float(ens.fixed_w), "ens_fixed_cagr": float(ens.fixed_cagr), "ens_fixed_mdd": float(ens.fixed_mdd),
                   "seed_mean_cagr": float(sd.cagr.mean()), "ens_final_nav": float(ens.final_nav),
                   "ens_e_up": float(ens.e_up), "ens_e_down": float(ens.e_down)}
        summ.append({"model": f"에이전트 {LABEL[v]} (앙상블)", **{c: ens[c] for c in ("cagr", "mdd", "ir", "annual_turnover",
                                                                                   "mean_cash", "final_nav", "mean_e",
                                                                                   "e_up", "e_down", "up_cap", "dn_cap")},
                     "fixed_w": ens.fixed_w, "fixed_cagr": ens.fixed_cagr, "fixed_mdd": ens.fixed_mdd,
                     "seed_cagr_min": float(sd.cagr.min()), "seed_cagr_max": float(sd.cagr.max()), **j})
    h3 = bool(judg["bkt"]["ens_cagr"] > judg["idx"]["ens_cagr"] and judg["bkt"]["seed_mean_cagr"] > judg["idx"]["seed_mean_cagr"])
    cands = [(v, judg[v]["ens_cagr"]) for v in VARIANTS
             if judg[v]["(1) 타이밍 실력 (같은 평균 비중 고정 대비)"] and judg[v]["(2) 수익 (B 100% 대비)"]]
    pick = max(cands, key=lambda x: x[1])[0] if cands else None
    sm = pd.DataFrame(summ)
    sm.to_csv(HERE / "bt_summary.csv", index=False)
    (HERE / "bt_judgment.json").write_text(json.dumps({
        "period": [str(b100.index[0].date()), str(b100.index[-1].date())], "signals": SIG_RUN, "fold_signals": RLSIG,
        "bench": {k: {c: bench[k].get(c) for c in ("cagr", "mdd", "ir")} for k in BENCH}, "kospi200": kospi,
        "variants": judg, "(3) 자기 종목으로 배우면 낫다 (bkt > idx)": h3, "confirm_candidate": pick,
        "phases": {"theta": THETA, "of": "B 100% 잔고", "segments": len(segs)}},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    ledger = Ledger(HERE / "ledger.jsonl")
    for v in VARIANTS:
        ledger.result(f"bt|{v}", **judg[v])
    ledger.result("bt|판정", **{"(3) 자기 종목으로 배우면 낫다 (bkt > idx)": h3}, confirm_candidate=pick)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(f"\n[B 포트폴리오 {b100.index[0].date()} ~ {b100.index[-1].date()}, 버퍼 + M8 Step 1~3 + M9 + C 엔진]")
    show = sm.copy()
    for c in [c for c in show.columns if c.startswith(("cagr", "mdd", "mean_cash", "mean_e", "e_up", "e_down", "fixed_w",
                                                       "fixed_cagr", "fixed_mdd", "seed_cagr", "up_cap", "dn_cap"))]:
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(1)
    for c in ("ir", "annual_turnover", "final_nav"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(3)
    print(show.drop(columns=[c for c in show.columns if c.startswith("(")]).to_string(index=False))
    yc = [c for c in rd.columns if c.startswith("y") and c[1:].isdigit()]
    yt = rd[rd.kind == "앙상블"].set_index("variant")[yc] * 100
    yt.index = [f"에이전트 {LABEL[v]}" for v in yt.index]
    bt = pd.DataFrame([{f"y{y}": r * 100 for y, r in yearly(nav_of(f"bench_{k}")).items()} for k in BENCH],
                      index=[BENCH[k][0] for k in BENCH])
    print("\n[연도별 수익률 %]")
    print(pd.concat([bt, yt])[yc].round(1).to_string())
    print("\n[시드별 CAGR % (괄호: 자기 평균 비중으로 고정했을 때)]")
    for v in VARIANTS:
        sd = rd[(rd.variant == v) & (rd.kind == "시드")]
        print(f"  {LABEL[v]}: " + ", ".join(f"{r.cagr * 100:+.1f} ({r.fixed_cagr * 100:+.1f}, E {r.mean_e:.0%})" for r in sd.itertuples()))
    print(f"\n[판정] (실행 전 고정) — 국면: B 100% 잔고 θ={THETA:.0%} 구간 {len(segs)}개")
    for v in VARIANTS:
        jv = judg[v]
        marks = ", ".join(f"{q} {'만족' if jv[q] else '미달'}" for q in jv if q.startswith("("))
        print(f"  {LABEL[v]}: {marks}")
        print(f"      앙상블 CAGR {jv['ens_cagr']:+.2%} vs 같은 평균 비중 {jv['ens_fixed_w']:.0%} 고정 {jv['ens_fixed_cagr']:+.2%}, "
              f"MDD {jv['ens_mdd']:.1%} vs {jv['ens_fixed_mdd']:.1%}, 자기 고정보다 CAGR 높은 시드 {jv['seeds_cagr_above_own_fixed']}/5, "
              f"B 100% {e1['cagr']:+.2%}, 끝 잔고 {jv['ens_final_nav']:.3f}, 상승·하락 구간 평균 비중 {jv['ens_e_up']:.0%}·{jv['ens_e_down']:.0%}")
    print(f"  (3) 자기 종목으로 배우면 낫다 (바구니만 > 지수만): {'만족' if h3 else '미달'}")
    print(f"  확인 대상 (처음 보는 데이터용): {pick or '없음'}")
    print("done")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "basket":
        build_basket()
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
    elif cmd == "report":
        report()
    else:
        raise SystemExit("사용법: bt_agent.py basket | smoke | jobs | worker <작업 파일> <슬롯> <슬롯 수> | status | evaluate | report")
