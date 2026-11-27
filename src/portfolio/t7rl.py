r"""B-09 탐색 고정 모델 T7-80-RL: T7 보유 80 + 리밸런싱마다 RL(PPO 5시드)이 정하는 주식 비중.

설계와 결정 기록은 docs/b09_explore.md 35절이다. 탐색 결과(R-15)이고, KPI 판정은 사전등록 설계 그대로다.

종목 구성 (T7-80-C53과 같다, 26절 규칙 1~7)
- 순위: A 신호 rank_score를 종목별 최근 4개 신호일(t 포함) 평균으로 바꾸고 U_t 안에서 다시 매긴다 (동점은 종목코드 순).
- 격주: 첫 신호일을 0번으로 셀 때 짝수 번째가 리밸런싱 주, 홀수 번째가 사이 주다.
- 리밸런싱 주: 버퍼 규칙 (보유 중 rank ≤ 140 유지, 80종목까지 rank ≤ 80을 순위대로 더함, 같은 비중)
  → crash_flag·M9 쿨다운 차단 → 종목 10%·업종 30% 한도 → 모든 종목 비중에 E_t를 곱한다.
- 사이 주: 실제 보유 비중 그대로. U_t 밖·crash_flag·쿨다운 종목만 팔고, 한도 넘는 몫만 줄인다. E_t를 다시 곱하지 않는다.
- 체결: C 엔진 (t+1 종가, 사전등록 비용·매도세, M9 청산). 현금 100%에서 시작하고 현금 수익은 0이다 (공식).

E_t (리밸런싱 주) = 고정 PPO 정책 5개(v12, λ 0.25, 시드 0~4)가 고른 비중(0~1, 11단계)의 평균
- 폴드 Y 정책: 학습 2011~Y−3 (바구니 + 나스닥·닛케이 1990~2010), 검증 Y−2~Y−1 점수 최고 체크포인트.
  홀드아웃(2026)과 forward는 폴드 2026 (학습 2011~2023, 검증 2024~2025, 명세 4.7절의 최종 모델 구간).
- 관측 11개 = 특성 10개를 폴드 학습 자료 평균·표준편차로 표준화해 ±5에서 자른 값 + 내부 장부의 지금 비중.
  - 바구니 자기 가격 특성 8개: 5·21·63·126일 로그수익, 20·60일 일간 로그수익 표준편차 × √252,
    252일 최고 대비 로그 낙폭, 200일 평균 대비 괴리. 바구니 가격 = T7 보유 80을 E = 1로 돌린 C 엔진 일별 NAV.
  - VIX 2개: FRED VIXCLS의 로그와 21행 로그 변화. 행은 VIXCLS·BAA10Y·DEXKOUS·NASDAQCOM 중 하나라도 값이 있는 미국
    날짜다 (앞 값으로 채움, 2009년부터, 학습 때와 같음). 결정일 t에는 t보다 앞선 마지막 미국 날짜 값을 쓴다.
- 내부 장부: 바구니 가격과 현금(CD 91일물, FRED IR3TIB01KRM156N 월평균, 달력일 복리)으로 만든 장부다.
  첫 결정일에 비중 0.466107에서 시작하고, t에 정한 E를 t+1 종가에 맞춘다 (사전등록 비용·매도세).
  관측의 '지금 비중'만 여기서 온다. 실제 체결은 C 엔진이 한다.

고정 파일 (reports/20261120_B_t7rl, DEC-37처럼 git에 둔다): 폴드 2016~2026 정책 55개(best.zip)와 표준화 값,
2011~2025 바구니 가격, FRED 미국 4계열(2008~2025), CD 91일물 월평균(~2025-12). 모든 파일의 SHA-256은 manifest.json에 있다.
2026 홀드아웃 실행(`run_holdout`)은 C 가드를 먼저 지난다 (개봉 2026-12-01 이후 + 허용 플래그, DEC-38).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.common import io
from src.common.guards import DataAccessPolicy, assert_outcome_dates_allowed
from src.portfolio.buffer_rule import buffer_rule_weights
from src.portfolio.params import BufferRuleParams, RiskLayerParams
from src.portfolio.risk_layer import check_invariants, positions_frame, risk_layer_weights

MODEL_ID = "T7-80-RL"
RUN_ID = "20261120_B_t7rl"
RUN_DIR = io.ROOT / "reports" / RUN_ID

N, BAND, EVERY, SMOOTH_WEEKS = 80, 60, 2, 4  # 보유 80, 청산 rank > 140, 격주, 4주 평균 순위
E_C53 = 1.0 - 0.533893  # T7-80-C53의 고정 E. 내부 장부의 첫 비중이기도 하다
LEVELS = np.round(np.arange(11) / 10.0, 1)  # 정책 행동 0~10 → 비중 0, 0.1, ..., 1
CLIP = 5.0
SEEDS = (0, 1, 2, 3, 4)
FOLDS = tuple(range(2016, 2027))  # 2016~2025는 walk-forward 재현, 2026은 홀드아웃·forward
FINAL_FOLD = 2026
OWN = ("o_r5", "o_r21", "o_r63", "o_r126", "o_vol20", "o_vol60", "o_dd252", "o_ma200")
FEATS = OWN + ("g_vix", "g_vix_chg")
US_SERIES = ("VIXCLS", "BAA10Y", "DEXKOUS", "NASDAQCOM")  # VIX 행 달력을 정하는 4계열 (학습 때와 같음)
US_START = pd.Timestamp("2009-01-01")
FROZEN_END = pd.Timestamp("2025-12-31")  # 고정 파일의 마지막 날 (학습·튜닝 끝, 4.7절)


# ---------------------------------------------------------------------------
# 종목 구성 (T7 보유 80)
# ---------------------------------------------------------------------------

def t7_buffer(buffer: BufferRuleParams) -> BufferRuleParams:
    """사전등록 버퍼 규칙에서 보유 수만 바꾼다: 진입 rank ≤ 80, 청산 rank > 140, 최대 80종목."""
    return replace(buffer, enter_rank_max=N, exit_rank_greater_than=N + BAND, max_holdings=N)


def rank_from_score(df: pd.DataFrame, col: str) -> pd.Series:
    """날짜별 U_t 안 순위 (1이 최고, 동점은 종목코드 순)."""
    r = df.sort_values(["date", col, "ticker"], ascending=[True, False, True], kind="stable")
    return r.groupby("date").cumcount().add(1).reindex(df.index).astype("int32")


def smooth_signals(sig: pd.DataFrame, weeks: int = SMOOTH_WEEKS) -> pd.DataFrame:
    """rank_score를 종목별 최근 weeks개 신호일(t 포함) 평균으로 바꾸고 순위를 다시 매긴다. 앞 신호가 적으면 있는 만큼.
    sig에 들어 있는 신호일만 평균에 쓴다 (평균의 시작 = sig의 첫 신호일)."""
    s = sig[["date", "ticker", "rank_score"]].sort_values(["ticker", "date"])
    sm = s.groupby("ticker")["rank_score"].transform(lambda x: x.rolling(weeks, min_periods=1).mean())
    out = sig.copy()
    out["rank_score"] = sm.reindex(sig.index).to_numpy()
    out["rank"] = rank_from_score(out, "rank_score").to_numpy()
    return out


def expand(es_reb: Sequence[float], n_dates: int) -> list[float]:
    """리밸런싱 주의 E(짝수 번째 신호일마다 하나)를 모든 신호일에 편다. 사이 주에는 직전 리밸런싱 주의 E가 기록된다."""
    if len(es_reb) != (n_dates + 1) // 2:
        raise ValueError(f"리밸런싱 E {len(es_reb)}개가 신호일 {n_dates}개와 맞지 않는다")
    return [float(es_reb[i // 2]) for i in range(n_dates)]


def build_positions(snaps: Sequence[Any], engine: Any, buffer: BufferRuleParams, risk: RiskLayerParams,
                    exposures: Sequence[float], trade_every: int = EVERY) -> pd.DataFrame:
    """신호일마다 목표 비중(positions, 9.3절 열)을 만든다. exposures[i] = i번째 신호일의 E (expand 결과).

    사이 주의 실제 보유 비중은 엔진이 직전 목표를 체결한 결과(weights_obs)로 안다. engine은 새로 만든 C 엔진이다."""
    if len(exposures) != len(snaps):
        raise ValueError("exposures와 신호일 수가 다르다")
    state, holdings, obs = engine.initial_state(), set(), {}
    frames = []
    for i, snap in enumerate(snaps):
        cooldown = snap.cooldown
        hold_week = trade_every > 1 and i % trade_every != 0 and i > 0
        if hold_week:  # 사이 주: 들고 있는 비중 그대로 (U_t 밖 종목은 0)
            w_raw = pd.Series(obs, dtype="float64")
            w_raw = w_raw[w_raw.index.isin(snap.tickers)]
        else:
            sig = pd.DataFrame({"ticker": snap.tickers, "rank": snap.rank, "crash_flag": snap.crash_flag.astype(np.int8)})
            w_raw = buffer_rule_weights(holdings, sig, snap.tickers, snap.tickers[cooldown], buffer)
        w = np.zeros(len(snap.tickers))
        if len(w_raw):
            w[np.array([snap.index[t] for t in w_raw.index], dtype=np.int64)] = w_raw.to_numpy()
        e = float(exposures[i])
        blocked = snap.crash_flag | cooldown
        # 사이 주의 w_raw는 이미 E가 곱해진 실제 비중이라 E를 다시 곱하지 않는다 (차단·한도만)
        w = risk_layer_weights(w, blocked, snap.sector_code, 1.0 if hold_week else e, risk, "full")
        cash = check_invariants(w, snap.sector_code, risk, "full")
        targets = {str(snap.tickers[j]): float(w[j]) for j in np.flatnonzero(w > 0.0)}
        frames.append(positions_frame(snap.date, targets, cash, "buffer", e, snap.regime_mc))
        if i + 1 < len(snaps):
            res = engine.run_period(state, targets, snap.date, snaps[i + 1].date)
            state = res.state
            obs = {t: float(v) for t, v in res.weights_obs.items() if v > 0.0}
            holdings = set(obs)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 관측 특성
# ---------------------------------------------------------------------------

def basket_features(px: pd.Series) -> pd.DataFrame:
    """바구니 일별 가격 → 자기 가격 특성 8개 (같은 날짜 색인)."""
    v = px.to_numpy(dtype=float)
    s = pd.Series(np.log(v), index=px.index)
    r1 = s.diff()
    out = pd.DataFrame({f"o_r{k}": s - s.shift(k) for k in (5, 21, 63, 126)}, index=px.index)
    out["o_vol20"] = r1.rolling(20, min_periods=20).std() * math.sqrt(252)
    out["o_vol60"] = r1.rolling(60, min_periods=60).std() * math.sqrt(252)
    out["o_dd252"] = s - s.rolling(252, min_periods=1).max()
    p = pd.Series(v, index=px.index)
    out["o_ma200"] = p / p.rolling(200, min_periods=200).mean() - 1
    return out[list(OWN)]


def read_fred_csv(path: str | Path, end: pd.Timestamp | None = None) -> pd.Series:
    """FRED 내려받기 CSV (날짜, 값) → 값이 있는 날만 남긴 Series ('.'·빈 칸은 뺀다)."""
    f = pd.read_csv(path)
    s = pd.Series(pd.to_numeric(f[f.columns[1]], errors="coerce").to_numpy(), index=pd.to_datetime(f[f.columns[0]]))
    if end is not None:
        s = s[s.index <= end]
    return s.dropna().sort_index()


def us_table(raw: Mapping[str, pd.Series]) -> pd.DataFrame:
    """미국 날짜 × (g_vix, g_vix_chg). raw = US_SERIES 4계열 (값 있는 날만)."""
    miss = [k for k in US_SERIES if k not in raw]
    if miss:
        raise ValueError(f"FRED 계열이 없다: {miss}")
    us = pd.DataFrame({k: raw[k] for k in US_SERIES}).sort_index().ffill()
    us = us[us.index >= US_START]
    lv = np.log(us["VIXCLS"])
    return pd.DataFrame({"g_vix": lv, "g_vix_chg": lv.diff(21)}, index=us.index)


def us_features(table: pd.DataFrame, decision_dates: Sequence[Any]) -> pd.DataFrame:
    """결정일 t마다 t보다 앞선 마지막 미국 날짜의 값 (미국 종가는 한국 장 마감 전에 나온다)."""
    d = pd.DatetimeIndex(pd.to_datetime(list(decision_dates)))
    idx = table.index.searchsorted(d, side="left") - 1
    if (idx < 0).any():
        raise ValueError("결정일 앞에 미국 값이 없다")
    out = table.iloc[idx].reset_index(drop=True)
    out.insert(0, "us_date", table.index[idx])
    out.insert(0, "date", d)
    if (out["us_date"] >= out["date"]).any():
        raise ValueError("미국 값이 결정일과 같거나 뒤다")
    return out


def feature_matrix(px: pd.Series, us: pd.DataFrame, decision_dates: Sequence[Any]) -> pd.DataFrame:
    """결정일 × 특성 10개 (FEATS 순서). px 색인에 결정일이 모두 있어야 한다."""
    d = pd.DatetimeIndex(pd.to_datetime(list(decision_dates)))
    own = basket_features(px)
    pos = own.index.get_indexer(d)
    if (pos < 0).any():
        raise ValueError("결정일이 바구니 가격 달력에 없다")
    f = own.iloc[pos].reset_index(drop=True)
    g = us_features(us, d)
    f["g_vix"], f["g_vix_chg"] = g["g_vix"].to_numpy(), g["g_vix_chg"].to_numpy()
    f.insert(0, "us_date", g["us_date"].to_numpy())
    f.insert(0, "date", d)
    return f


# ---------------------------------------------------------------------------
# 현금 (CD 91일물)과 내부 장부
# ---------------------------------------------------------------------------

def read_cd91(path: str | Path) -> pd.Series:
    """month(YYYY-MM), cd91_pct(연 %) → 월 → 연금리(소수)."""
    m = pd.read_csv(path)
    return pd.Series(m["cd91_pct"].to_numpy(dtype=float) / 100.0, index=pd.PeriodIndex(m["month"], freq="M"))


def cd91_from_fred_csv(path: str | Path) -> pd.Series:
    """FRED IR3TIB01KRM156N 내려받기 CSV (월 첫날, 연 %) → read_cd91과 같은 형식 (월 → 연금리 소수)."""
    f = pd.read_csv(path)
    v = pd.to_numeric(f[f.columns[1]], errors="coerce").to_numpy(dtype=float) / 100.0
    s = pd.Series(v, index=pd.PeriodIndex(pd.to_datetime(f[f.columns[0]]), freq="M"))
    return s.dropna()


def cash_daily(days: Sequence[Any], monthly: pd.Series) -> np.ndarray:
    """거래일 → r[k] = 거래일 k−1 → k의 현금 수익 = (1 + k−1이 속한 달 연금리)^(달력일 / 365) − 1, r[0] = 0."""
    d = pd.DatetimeIndex(pd.to_datetime(list(days)))
    r = np.zeros(len(d))
    for k in range(1, len(d)):
        p = d[k - 1].to_period("M")
        if p not in monthly.index:
            raise KeyError(f"CD 91일물 금리가 없는 달: {p}")
        r[k] = (1.0 + float(monthly[p])) ** ((d[k] - d[k - 1]).days / 365.0) - 1.0
    return r


def extend_monthly(monthly: pd.Series, last_month: pd.Period) -> tuple[pd.Series, list[str]]:
    """last_month까지 빈 달을 직전 달 값으로 채운다 (발표가 늦은 달). 반환: (채운 시계열, 채운 달 목록)."""
    out = monthly.sort_index()
    filled = []
    p = out.index.max()
    while p < last_month:
        p = p + 1
        if p not in out.index:
            out = pd.concat([out, pd.Series([float(out.iloc[-1])], index=pd.PeriodIndex([p], freq="M"))])
            filled.append(str(p))
    return out.sort_index(), filled


class Book:
    """바구니(가격 px) + 현금(rc) 장부. 결정일 종가에 정한 비중을 다음 거래일 종가에 맞춘다 (줄일 때 매도세)."""

    __slots__ = ("px", "rc", "i", "units", "cash", "nav", "rate", "tax")

    def __init__(self, px: np.ndarray, i0: int, w0: float, rc: np.ndarray, rate: float, tax: float) -> None:
        self.px, self.rc, self.i, self.rate, self.tax = px, rc, int(i0), float(rate), float(tax)
        self.nav = 1.0
        self.units = w0 / px[self.i]
        self.cash = 1.0 - w0

    def weight(self) -> float:
        return self.units * self.px[self.i] / self.nav

    def run(self, j: int, target: float) -> None:
        """i 종가에 정한 target을 i+1 종가에 체결하고 j 종가까지 간다."""
        first = self.i + 1
        for d in range(first, j + 1):
            self.cash *= 1.0 + self.rc[d]
            p = self.px[d]
            nav = self.units * p + self.cash
            if d == first:
                w = self.units * p / nav
                trade = (target - w) * nav
                nav -= abs(trade) * self.rate + max(0.0, -trade) * self.tax
                self.units = target * nav / p
                self.cash = nav - self.units * p
            self.nav = nav
        self.i = int(j)


def cost_rates(cfg: Mapping[str, Any]) -> tuple[float, float]:
    """사전등록 비용: (한쪽 수수료 + 슬리피지, 매도세)."""
    c = cfg["costs"]
    return float(c["commission_per_side"]) + float(c["slippage_per_side"]), float(c["sell_tax"])


# ---------------------------------------------------------------------------
# 정책
# ---------------------------------------------------------------------------

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(run_dir: Path = RUN_DIR) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def verify_files(rel_paths: Sequence[str], run_dir: Path = RUN_DIR) -> None:
    """고정 파일의 SHA-256이 manifest.json과 같은지 확인한다. 다르면 멈춘다."""
    files = manifest(run_dir)["files"]
    for rel in rel_paths:
        if rel not in files:
            raise ValueError(f"manifest에 없는 파일: {rel}")
        got = sha256(run_dir / rel)
        if got != files[rel]:
            raise ValueError(f"고정 파일이 바뀌었다: {rel} ({got[:12]} ≠ {files[rel][:12]})")


@dataclass(frozen=True)
class FoldPolicy:
    """한 폴드의 PPO 정책 5개와 표준화 값."""

    fold: int
    models: tuple
    mu: np.ndarray
    sd: np.ndarray

    def observation(self, x: np.ndarray, weight: float) -> np.ndarray:
        z = np.clip((np.asarray(x, dtype=float) - self.mu) / self.sd, -CLIP, CLIP)
        return np.append(z, weight).astype(np.float32)

    def actions(self, x: np.ndarray, weight: float) -> list[int]:
        o = self.observation(x, weight)
        return [int(m.predict(o, deterministic=True)[0]) for m in self.models]

    def exposure(self, x: np.ndarray, weight: float) -> tuple[float, list[int]]:
        a = self.actions(x, weight)
        return float(np.mean([LEVELS[k] for k in a])), a


def fold_files(fold: int) -> list[str]:
    return [f"models/y{fold}/stats.json"] + [f"models/y{fold}/seed_{s}/best.zip" for s in SEEDS]


def load_policy(fold: int, run_dir: Path = RUN_DIR, verify: bool = True) -> FoldPolicy:
    from stable_baselines3 import PPO

    if verify:
        verify_files(fold_files(fold), run_dir)
    st = json.loads((run_dir / "models" / f"y{fold}" / "stats.json").read_text(encoding="utf-8"))
    if tuple(st["cols"]) != FEATS:
        raise ValueError(f"폴드 {fold}: 특성 열이 다르다 {st['cols']}")
    models = tuple(PPO.load(run_dir / "models" / f"y{fold}" / f"seed_{s}" / "best.zip", device="cpu") for s in SEEDS)
    return FoldPolicy(int(fold), models, np.asarray(st["mu"], dtype=float), np.asarray(st["sd"], dtype=float))


def exposure_path(px: pd.Series, features: pd.DataFrame, policy_for: Callable[[pd.Timestamp], FoldPolicy],
                  rc: np.ndarray, rate: float, tax: float, w0: float = E_C53) -> pd.DataFrame:
    """결정일마다 E를 정한다. features = feature_matrix 결과 (결정일 오름차순). 장부는 첫 결정일 비중 w0에서 시작해
    E를 따라간다 (해가 바뀌어 폴드가 바뀌어도 장부는 잇는다). 반환: 결정일, 폴드, 장부 비중, 시드별 행동, E."""
    days = pd.DatetimeIndex(px.index)
    p = px.to_numpy(dtype=float)
    dd = pd.DatetimeIndex(features["date"])
    dec = days.get_indexer(dd)
    if (dec < 0).any() or not (np.diff(dec) > 0).all():
        raise ValueError("결정일이 바구니 가격 달력에 없거나 오름차순이 아니다")
    if len(rc) != len(p):
        raise ValueError("현금 수익 길이가 바구니 가격과 다르다")
    X = features[list(FEATS)].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("결정일 특성에 빈 값이 있다")
    book = Book(p, int(dec[0]), w0, rc, rate, tax)
    rows = []
    for k, d in enumerate(dd):
        pol = policy_for(d)
        w = book.weight()
        e, acts = pol.exposure(X[k], w)
        rows.append({"date": d, "fold": pol.fold, "book_weight": w, **{f"a{s}": a for s, a in zip(SEEDS, acts)},
                     "exposure": e})
        j = int(dec[k + 1]) if k + 1 < len(dec) else len(p) - 1
        if j > book.i:
            book.run(j, e)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 고정 입력
# ---------------------------------------------------------------------------

INPUT_FILES = ("inputs/basket_px.csv", "inputs/fred_us.csv", "inputs/cd91_monthly.csv")


def frozen_inputs(run_dir: Path = RUN_DIR, verify: bool = True) -> tuple[pd.Series, dict[str, pd.Series], pd.Series]:
    """(바구니 가격 2011~2025, FRED 미국 4계열 ~2025-12-31, CD 91일물 월평균 ~2025-12)."""
    if verify:
        verify_files(INPUT_FILES, run_dir)
    b = pd.read_csv(run_dir / "inputs" / "basket_px.csv", parse_dates=["date"])
    px = pd.Series(b["px"].to_numpy(dtype=float), index=pd.DatetimeIndex(b["date"]).as_unit("ns"), name="px")
    u = pd.read_csv(run_dir / "inputs" / "fred_us.csv", parse_dates=["date"]).set_index("date")
    raw = {k: u[k].dropna() for k in US_SERIES}
    return px, raw, read_cd91(run_dir / "inputs" / "cd91_monthly.csv")


# ---------------------------------------------------------------------------
# 2026 홀드아웃 (C가 12/11 개봉 뒤 PM 승인 때 한 번 실행)
# ---------------------------------------------------------------------------

def holdout_access_policy(cfg: Mapping[str, Any]) -> DataAccessPolicy:
    s = cfg["splits"]
    h0, h1 = (date.fromisoformat(v) for v in s["holdout_signal_range"])
    return DataAccessPolicy(date(2025, 12, 31), h0, h1, date.fromisoformat(s["holdout_open_date"]))


def chain_basket(px_frozen: pd.Series, e1_nav: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """고정 바구니 가격(~2025-12-30) 뒤에 2026 E = 1 실행 NAV를 잇는다. 첫 체결일 전 거래일은 그대로(현금),
    첫 체결일부터 = 마지막 고정 가격 × NAV (NAV는 첫 체결 직전 1)."""
    last = px_frozen.index[-1]
    days = pd.DatetimeIndex(calendar)
    days = days[(days > last) & (days <= e1_nav.index[-1])]
    nav = e1_nav.reindex(days)
    first_fill = e1_nav.index[0]
    nav[days < first_fill] = 1.0
    if nav.isna().any():
        raise ValueError("E = 1 NAV에 빈 거래일이 있다")
    new = float(px_frozen.iloc[-1]) * nav
    return pd.concat([px_frozen, new.rename("px")])


@dataclass
class PeriodRun:
    dates: list[str]  # 신호일
    decisions: pd.DataFrame  # 리밸런싱 결정일마다 E (exposure_path)
    positions: dict[str, pd.DataFrame]  # "t7rl", "t7c53", "t7e1" (E = 1, 바구니 가격용)
    basket_px: pd.Series  # 이어 붙인 바구니 가격 (특성용)
    record: dict[str, Any]


def build_period(dates: Sequence[str], snaps: Sequence[Any], make_engine: Callable[[bool], Any], calendar: Sequence[Any],
                 cfg: Mapping[str, Any], px_before: pd.Series, raw: Mapping[str, pd.Series], cd91: pd.Series,
                 policy: FoldPolicy) -> PeriodRun:
    """신호일 dates 구간을 현금 100%에서 새로 시작해 T7-80-RL·T7-80-C53·E = 1 목표 비중을 만든다.

    - 바구니 가격 = px_before(구간 앞) 뒤에 이 구간 E = 1 실행 NAV를 이은 것 (chain_basket, calendar = 거래일).
      결정일 특성은 그 가격과 raw(FRED 4계열)로 만든다. 내부 장부는 첫 결정일에 비중 E_C53에서 시작하고 현금은
      cd91(월평균, 빈 달은 직전 값).
    - snaps는 이 구간 신호로만 4주 평균 순위를 만든 것이어야 한다 (smooth_signals를 구간 신호에만 적용).
    - make_engine(ledger) = 새 C 엔진 (같은 비용·M9 일정). 홀드아웃(run_holdout)과 리허설이 같이 쓴다."""
    from src.portfolio.buffer_wf_report import replay

    risk = RiskLayerParams.from_prereg(cfg)
    buffer = t7_buffer(BufferRuleParams.from_prereg(cfg))
    rate, tax = cost_rates(cfg)
    dates = list(dates)
    n = len(dates)
    pos_e1 = build_positions(snaps, make_engine(False), buffer, risk, [1.0] * n)
    rep = replay(pos_e1, dates, make_engine(True))
    e1 = rep.nav.astype("float64").copy()
    e1.index = pd.DatetimeIndex(pd.to_datetime(e1.index)).as_unit("ns")
    cal = pd.DatetimeIndex(pd.to_datetime(list(calendar))).as_unit("ns")
    px = chain_basket(px_before, e1, cal)
    us = us_table(raw)
    reb = dates[0::EVERY]
    fm = feature_matrix(px, us, reb)
    cd, filled = extend_monthly(cd91, px.index[-1].to_period("M"))
    rc = cash_daily(px.index, cd)
    dec = exposure_path(px, fm, lambda d: policy, rc, rate, tax)
    pos_rl = build_positions(snaps, make_engine(False), buffer, risk, expand(dec["exposure"].tolist(), n))
    pos_c53 = build_positions(snaps, make_engine(False), buffer, risk, [E_C53] * n)
    record = {"model": MODEL_ID, "run_id": RUN_ID, "fold": policy.fold, "signal_dates": [dates[0], dates[-1], n],
              "rebalancing_dates": len(reb), "basket_px_before_end": str(px_before.index[-1].date()),
              "e1_first_fill": str(e1.index[0].date()), "us_last_date": str(us.index[-1].date()),
              "cd91_filled_months": filled, "mean_exposure": float(dec["exposure"].mean())}
    return PeriodRun(dates, dec, {"t7rl": pos_rl, "t7c53": pos_c53, "t7e1": pos_e1}, px, record)


def run_holdout(signals_run_id: str, m9_run_id: str | None, fred_2026: Mapping[str, pd.Series], cd91_2026: pd.Series, *,
                allow_holdout: bool, run_date: date, run_dir: Path = RUN_DIR) -> PeriodRun:
    """2026 홀드아웃 신호일(사전등록 splits.holdout_signal_range)의 T7-80-RL·T7-80-C53 목표 비중을 만든다.

    가드가 먼저다: 허용 플래그와 개봉일(2026-12-01, DEC-38) 이후 실행이 아니면 아무것도 읽지 않고 멈춘다 (4.7절, PM-14).
    signals_run_id = 사전등록 모델과 같은 A 최종 모델의 홀드아웃 ML 신호, m9_run_id = 같은 기간 PM M9 경보.
    fred_2026·cd91_2026 = FRED 최신 내려받기 (2025-12-31 이전 값은 고정 파일을 쓰고 그 뒤 값만 붙인다).
    지표는 C가 사전등록 모델과 같은 엔진·같은 구간으로 계산한다 (positions를 넘긴다).

    예 (C, 개봉 뒤): fred = {k: read_fred_csv(f"{k}.csv") for k in US_SERIES}
                     cd = cd91_from_fred_csv("IR3TIB01KRM156N.csv")
                     run = run_holdout("<A 홀드아웃 신호 run_id>", "<PM M9 run_id>", fred, cd, allow_holdout=True,
                                       run_date=<KST 오늘>)"""
    from src.portfolio.engine_adapter import build_engine, engine_inputs
    from src.portfolio.env_check_report import TABLES, feature_table, kospi200, members_by_date, signal_dates
    from src.portfolio.params import PREREG_PATH, load_prereg
    from src.portfolio.rl_data import build_snapshots
    from src.portfolio.rl_fold_report import ENGINE_TABLES, m9_schedule

    cfg = load_prereg(PREREG_PATH)
    h0, h1 = cfg["splits"]["holdout_signal_range"]
    assert_outcome_dates_allowed([h0, h1], policy=holdout_access_policy(cfg), allow_holdout=allow_holdout, run_date=run_date)
    risk = RiskLayerParams.from_prereg(cfg)
    px0, raw0, cd0 = frozen_inputs(run_dir)
    policy = load_policy(FINAL_FOLD, run_dir)
    tables = {n: io.read_table(n) for n in TABLES}
    bm = kospi200(tables["index"])
    dates = signal_dates(tables["universe"], bm.index, h0, h1)
    sig = io.read_table("signals", run_id=signals_run_id)
    sig["date"] = pd.to_datetime(sig["date"])
    sig = sig[sig["date"].isin(pd.to_datetime(dates))].reset_index(drop=True)
    if sorted(sig["date"].dt.strftime("%Y-%m-%d").unique()) != list(dates):
        raise ValueError("홀드아웃 신호일마다 ML 신호가 있어야 한다")
    feats = feature_table(tables["prices"], bm, members_by_date(tables["universe"], dates), risk)
    inputs = engine_inputs(*(tables[n] for n in ENGINE_TABLES))
    m9 = m9_schedule(m9_run_id, dates[0], dates[-1])
    snaps = build_snapshots(dates, smooth_signals(sig), feats, tables["universe"], tables["sector"], inputs.bm_close,
                            m9.cooldown_at, risk)

    def make_engine(ledger: bool):
        alerts = {"cooldowns": m9.cooldowns, "event_liquidations": m9.liquidations} if m9.cooldowns is not None else {}
        return build_engine(inputs, cfg, record_ledger=ledger, **alerts)

    raw = {k: pd.concat([raw0[k], fred_2026[k][fred_2026[k].index > FROZEN_END]]).sort_index() for k in US_SERIES}
    cd = pd.concat([cd0, cd91_2026[cd91_2026.index > cd0.index.max()]]).sort_index()
    out = build_period(dates, snaps, make_engine, inputs.bm_close.index, cfg, px0, raw, cd, policy)
    out.record.update({"signals_run_id": signals_run_id, "m9": m9.record, "run_date": run_date.isoformat()})
    return out
