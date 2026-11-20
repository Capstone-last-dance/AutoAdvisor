"""B-09 탐색 공용 도구 (브랜치 b09-explore). 공식 함수를 불러 쓴다. 실행: python -I explore/b09/run.py <스크립트> [인자]

- 입력: 9.3절 표(data/processed), v7 시험 신호, PM M9 경보, 부록 1 설정 — 공식 G5 비교 표와 같다.
- 체결: C 엔진(build_engine). 지표: buffer_wf_report.replay/summary (C M5 함수, 판정 기간 끝 처리 포함).
- 변형은 목표 비중·노출·차단 종목만 바꾼다. 불변식은 check_invariants로 매번 확인한다.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # 저장소 루트 (explore/b09/의 두 단계 위)
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from src.common import io
from src.portfolio.buffer_rule import buffer_rule_weights
from src.portfolio.buffer_wf_report import g5_judgment, g7_check, period_end, replay, summary
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.env_check_report import (FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE, TABLES, feature_table, kospi200,
                                            members_by_date, signal_dates)
from src.portfolio.params import PREREG_PATH, BufferRuleParams, RiskLayerParams, load_prereg
from src.portfolio.risk_layer import CASH, check_invariants, market_state_table, positions_frame, risk_layer_weights
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_fold_report import ENGINE_TABLES, m9_schedule

SIG_RUN = "20261117_A_wfnews"
M9_RUN = "20261112_PM_m9_b07_v1"
DIAG = REPO / "files/A_files/A06_07_wfnews_20261117/reports/wf_diagnostics.parquet"
SEARCH_YEARS = (2016, 2017, 2018, 2019, 2020)
CONFIRM_YEARS = (2021, 2022, 2023, 2024, 2025)
ALL_YEARS = SEARCH_YEARS + CONFIRM_YEARS


def iso(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


class Ctx:
    """한 번 읽어 여러 변형에 쓴다. years: 평가할 시험 연도 (선별은 SEARCH_YEARS)."""

    def __init__(self, years=SEARCH_YEARS):
        t0 = time.perf_counter()
        self.cfg = load_prereg(PREREG_PATH)
        self.risk = RiskLayerParams.from_prereg(self.cfg)
        self.buffer = BufferRuleParams.from_prereg(self.cfg)
        self.kpi = self.cfg["kpi"]
        self.tables = {n: io.read_table(n) for n in TABLES}
        self.bm = kospi200(self.tables["index"])
        self.all_dates = signal_dates(self.tables["universe"], self.bm.index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)
        self.years = tuple(years)
        self.dates = [d for d in self.all_dates if int(d[:4]) in self.years]
        sig = io.read_table("signals", run_id=SIG_RUN)
        sig["date"] = pd.to_datetime(sig["date"])
        self.signals = sig[sig["date"].isin(pd.to_datetime(self.dates))].reset_index(drop=True)
        self.members = members_by_date(self.tables["universe"], self.dates)
        self.features = feature_table(self.tables["prices"], self.bm, self.members, self.risk)
        self.inputs = engine_inputs(*(self.tables[n] for n in ENGINE_TABLES))
        self.m9 = m9_schedule(M9_RUN, self.all_dates[0], self.all_dates[-1])
        try:  # 10년·확인 구간: 마지막 신호일도 체결하고 그해 마지막 거래일까지 (판정 기간과 같다)
            self.last_period = period_end(self.inputs.bm_close.index, self.dates[-1], self.years[-1])
        except ValueError:  # 선별 구간(2020-12-30이 신호일이자 마지막 거래일): 마지막 신호일은 체결하지 않는다
            self.last_period = None
        self._snaps: dict = {}
        self._close = None
        self._alt: dict = {}
        self.load_seconds = time.perf_counter() - t0

    def alt_signals(self, sig: str | None) -> pd.DataFrame:
        """성능 트랙 신호 판 (sig_b1.py): None = v7 ML 시험 신호, b1 = A 베이스라인, b1ml = B1 rank + ML 급락."""
        if sig is None:
            return self.signals
        if sig not in self._alt:
            from sig_b1 import apply_sig, b1_table
            b1 = b1_table(self.tables["prices"], self.tables["universe"], self.dates)
            self._alt[sig] = apply_sig(sig, self.signals, b1)
        return self._alt[sig]

    def sig_snaps(self, sig: str | None):
        return self.snaps() if sig is None else self.snaps(signals=self.alt_signals(sig), key=f"sig-{sig}")

    # ---- 입력 -------------------------------------------------------------
    def snaps(self, signals: pd.DataFrame | None = None, risk: RiskLayerParams | None = None, key: str = "base"):
        k = (key, id(risk) if risk is not None else None)
        if k not in self._snaps:
            self._snaps[k] = build_snapshots(self.dates, self.signals if signals is None else signals, self.features,
                                             self.tables["universe"], self.tables["sector"], self.inputs.bm_close,
                                             self.m9.cooldown_at, risk or self.risk)
        return self._snaps[k]

    def engine(self, with_m9: bool, ledger: bool):
        alerts = {"cooldowns": self.m9.cooldowns, "event_liquidations": self.m9.liquidations} if with_m9 else {}
        return build_engine(self.inputs, self.cfg, record_ledger=ledger, **alerts)

    def close(self) -> pd.DataFrame:
        """수정 종가 (거래일 × 종목). U_t 합집합만."""
        if self._close is None:
            p = self.tables["prices"]
            tick = sorted({t for v in self.members.values() for t in v})
            p = p[p["ticker"].isin(tick)]
            self._close = p.pivot(index="date", columns="ticker", values="close").sort_index()
            self._close.index = pd.to_datetime(self._close.index)
        return self._close

    def diag(self) -> pd.DataFrame:
        d = pd.read_parquet(DIAG)
        d["date"] = pd.to_datetime(d["date"])
        return d[d["date"].isin(pd.to_datetime(self.dates))].reset_index(drop=True)

    # ---- 평가 -------------------------------------------------------------
    def evaluate(self, positions: pd.DataFrame, with_m9: bool) -> dict:
        rep = replay(positions, self.dates, self.engine(with_m9, ledger=True), self.last_period)
        s = summary(rep, self.inputs.bm_close, positions)
        s["_rep"] = rep
        return s


def top_fraction_flag(df: pd.DataFrame, col: str, fraction: float) -> pd.Series:
    """날짜별 col 상위 ceil(fraction × N_t)개에 1 (A flag_top과 같은 개수, 동점은 종목코드 순)."""
    out = pd.Series(0, index=df.index, dtype="int8")
    for _, g in df.groupby("date"):
        k = math.ceil(fraction * len(g))
        idx = g.sort_values([col, "ticker"], ascending=[False, True], kind="stable").index[:k]
        out.loc[idx] = 1
    return out


def rank_from_score(df: pd.DataFrame, col: str) -> pd.Series:
    """U_t 안 순위 (1이 최고, 동점은 종목코드 순)."""
    r = df.sort_values(["date", col, "ticker"], ascending=[True, False, True], kind="stable")
    return r.groupby("date").cumcount().add(1).reindex(df.index).astype("int32")


# ---- 버퍼 규칙 실행 (run_buffer_rule과 같은 흐름 + 변형 고리) -------------------------


def buffer_run(ctx: Ctx, snaps, buffer: BufferRuleParams | None = None, risk: RiskLayerParams | None = None,
               mode: str = "full", with_m9: bool | None = None, exposure_fn=None, block_fn=None,
               trade_every: int = 1, choose_fn=None, hold_scale: bool = False) -> pd.DataFrame:
    """positions를 만든다. exposure_fn(i, snap, E_t, nav_hist, chosen) → 새 E_t (mode full에서만),
    block_fn(i, snap) → 추가 Step 1 차단(bool 배열, mode off에서는 쓰지 않음),
    trade_every: k주마다 리밸런싱 (사이 주는 직전 실제 보유 비중 유지, E_t는 리밸런싱 주에만 곱한다),
    choose_fn(snap, holdings, buffer) → w_raw Series (기본 buffer_rule_weights),
    hold_scale: True면 사이 주에도 들고 있는 종목을 비례로 늘리거나 줄여 합을 그 주 E_t에 맞춘다 (종목은 바꾸지 않음).
    들고 있는 종목이 없는 사이 주는 그 주에 바구니를 고른다. 기본 False (11/20 T7 비중 에이전트 v5용)."""
    buffer, risk = buffer or ctx.buffer, risk or ctx.risk
    with_m9 = (mode != "off") if with_m9 is None else with_m9
    engine = ctx.engine(with_m9, ledger=exposure_fn is not None)
    state, holdings, obs = engine.initial_state(), set(), {}
    frames, nav_hist = [], []
    for i, snap in enumerate(snaps):
        cooldown = snap.cooldown if mode != "off" else np.zeros_like(snap.cooldown)
        hold_week = trade_every > 1 and i % trade_every != 0 and i > 0
        if hold_week and hold_scale and not obs:
            hold_week = False  # hold_scale: 현금 100%인 사이 주는 그 주에 바구니를 고른다
        if hold_week:
            w_raw = pd.Series(obs, dtype="float64")
            w_raw = w_raw[w_raw.index.isin(snap.tickers)]
            if hold_scale and mode == "full" and float(w_raw.sum()) > 0.0:
                w_raw = w_raw * (float(snap.exposure) / float(w_raw.sum()))  # 들고 있는 종목을 비례로 그 주 E_t에 맞춘다
        elif choose_fn is not None:
            w_raw = choose_fn(snap, holdings, buffer, cooldown)
        else:
            sig = pd.DataFrame({"ticker": snap.tickers, "rank": snap.rank, "crash_flag": snap.crash_flag.astype(np.int8)})
            w_raw = buffer_rule_weights(holdings, sig, snap.tickers, snap.tickers[cooldown], buffer)
        w = np.zeros(len(snap.tickers))
        if len(w_raw):
            w[np.array([snap.index[t] for t in w_raw.index], dtype=np.int64)] = w_raw.to_numpy()
        e = snap.exposure if mode == "full" else 1.0
        if exposure_fn is not None and mode == "full":
            if engine.ledger:
                led = pd.DataFrame(engine.ledger)
                nav_hist = led.groupby("date", sort=False)["nav_after"].last()
                nav_hist = nav_hist[pd.to_datetime(nav_hist.index) <= pd.Timestamp(snap.date)]
            e = float(exposure_fn(i, snap, e, nav_hist, list(w_raw.index)))
        blocked = snap.crash_flag | cooldown
        if block_fn is not None and mode != "off":
            blocked = blocked | block_fn(i, snap)
        # 사이 주의 w_raw는 이미 직전 E_t가 곱해진 실제 비중이라 E_t를 다시 곱하지 않는다 (Step 1~3만).
        # 11/20 고침: 전에는 mode full + trade_every > 1에서 두 번 곱했다 (6절 S-freq-2·S-b1+freq2가 영향을 받음, E = 1 실행은 같음)
        w = risk_layer_weights(w, blocked, snap.sector_code, 1.0 if hold_week else e, risk, mode)
        cash = check_invariants(w, snap.sector_code, risk, mode)
        targets = {str(snap.tickers[j]): float(w[j]) for j in np.flatnonzero(w > 0.0)}
        frames.append(positions_frame(snap.date, targets, cash, "buffer", e, snap.regime_mc))
        if i + 1 < len(snaps):
            res = engine.run_period(state, targets, snap.date, snaps[i + 1].date)
            state = res.state
            obs = {t: float(v) for t, v in res.weights_obs.items() if v > 0.0}
            holdings = set(obs)
    return pd.concat(frames, ignore_index=True)


METRIC_KEYS = ("ir", "cagr", "annual_excess", "mdd", "annual_turnover", "mean_cash")


def brief(s: dict) -> dict:
    return {k: (None if s.get(k) is None else float(s[k])) for k in METRIC_KEYS}


class Ledger:
    """원장: 실행 전에 계획 행, 실행 뒤에 결과 행을 덧붙인다 (지우지 않는다)."""

    def __init__(self, path: Path):
        self.path = path

    def _write(self, row: dict) -> None:
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")

    def plan(self, vid: str, track: str, change: str, why: str) -> None:
        self._write({"event": "plan", "id": vid, "track": track, "change": change, "why": why})

    def result(self, vid: str, **kw) -> None:
        self._write({"event": "result", "id": vid, **kw})
