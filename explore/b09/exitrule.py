r"""종목 이탈 장치: T7·T3 위에 ① 느린 추세 이탈 ② 매일 급락 손절 ③ 빠른 복귀를 더하면 MDD가 줄고 수익이 오르나 (11/19 사용자 결정).

사용자 결정(11/19): 오락가락(잦은 교체)·갑작스런 폭락·V자 반등 놓침을 모두 피해야 수익이 오르고 MDD가 준다.
첫 하락을 못 막아도 적용한다.

실행 (저장소 루트, PowerShell, 사용자가 돌린다, 5~15분, 창 1개):
  .\.venv\Scripts\python.exe -I explore\b09\run.py exitrule.py
  끊기면 같은 명령을 다시 돌린다 (끝난 실행은 건너뛴다).

기준 (turnover.py, docs 16절): T7 = 4주 평균 순위 + 청산 80 + 격주, T3 = 4주 평균 순위 + 청산 80 + 매주.
모두 노출 조절 끔(E = 1), crash_flag 차단, 종목 10%·업종 30% 한도, M9, C 엔진, 2016~2025 연속.

장치 (실행 전 고정 11/19). 가격은 수정종가(거래 없는 날은 직전 값). 판단은 그날 종가까지만 쓴다.
- ① 느린 추세 이탈: 리밸런싱 신호일(T7은 2주마다, T3은 매주)에 종가 < 자기 100일선 × 0.95이면 제외한다.
  ③이 없으면 제외된 종목은 리밸런싱 신호일 종가 > 100일선일 때 다시 후보가 된다 (완충 구간).
- ② 매일 급락 손절: 종가 < 최근 20거래일 종가 고점 × 0.80이 되는 첫날 d마다, 들고 있는 종목은 d 다음 거래일 종가에 판다.
  엔진의 이벤트 청산 경로(M9 청산과 같다)라 하한가(−29.5%)·거래정지 날에는 못 팔고 다음 날로 넘어간다.
  ③이 없으면 d부터 20거래일 안의 리밸런싱 신호일에는 다시 사지 않는다. 들고 있지 않은 종목에도 같이 계산한다 (최근 급락 종목은 새로 사지 않는다).
- ③ 빠른 복귀: ①과 함께면 '100일선 × 0.95 아래이면서 20일선 아래'일 때만 제외한다 (20일선 위로 올라오면 바로 후보).
  ②와 함께면 손절 뒤(손절일 다음 날부터) 리밸런싱 신호일 종가가 20일선 위이면 다시 후보가 된다.
- 제외는 crash_flag와 같게 다룬다: 들고 있으면 팔고, 새로 사지 않고, 빈자리는 다음 순위 종목으로 채운다.
  ①·③ 판단과 ② 재매수 금지는 리밸런싱 신호일에만 적용하고, ② 손절 매도는 매일이다.
  ②의 첫 판단일은 첫 신호일(2016-01-08)이고, 그보다 20거래일 넘게 앞선 손절은 무시한다.

구성 (기준마다 8개, 합 16개)
- S0 장치 없음 (기준. turnover.py의 T7·T3과 같아야 한다 — 다르면 멈춘다)
- S1 ①, S2 ②, S3 ①+③, S4 ①+②+③
- 민감도: S4-s15 손절 15%, S4-s25 손절 25%, S4-m50 추세선 50일 (나머지는 S4와 같다)

판정 (실행 전 고정 11/19), 같은 기준의 S0와 비교:
- (1) 10년 MDD < S0
- (2) 10년 CAGR ≥ S0 − 1%p
- (3) 2016~2020과 2021~2025 각각에서도 MDD < S0이고 CAGR ≥ S0 − 1%p (10년 연속 실행을 2020년 마지막 거래일에서 나눠 잰다)
- 넘음 = (1)·(2)·(3) 모두. 확인 대상 = 기준마다 넘은 구성 중 10년 CAGR이 가장 높은 것.
- 참고: 회전율, 비용 추정, 손절이 걸린 보유 종목 수, 그중 하한가·정지로 그날 못 판 수, 하락·상승 구간 수익,
  연도별, 리밸런싱 신호일마다 rank ≤ 40 후보 중 장치로 제외된 평균 수, 코스피 200.
- 2016~2025는 여러 번 본 구간이고 구성이 16개라, 넘어도 forward test(PM이 허락하면 2026 홀드아웃)로 확인해야 한다.
- 탐색 결과다 (R-15). 결과: exit_runs.csv, exit_judgment.json, runs/exit/, 원장 exit|*.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

from common import ALL_YEARS, HERE, Ctx, Ledger, buffer_run, replay, summary
import numpy as np
import pandas as pd
from src.portfolio.buffer_wf_report import benchmark_summary
from src.portfolio.engine_adapter import build_engine
from turnover import ledger_has, smooth_signals, split_cagr, yearly

OUT = HERE / "runs" / "exit"
TOV = HERE / "runs" / "tov"
SPLIT = pd.Timestamp("2020-12-31")
BAND, BAN, FAST, HIGH = 0.05, 20, 20, 20
BASES = {"T7": ("4주 평균 순위 + 청산 80 + 격주", 2), "T3": ("4주 평균 순위 + 청산 80, 매주", 1)}
VARIANTS = [
    ("S0", "장치 없음", {}),
    ("S1", "① 추세 이탈", {"trend": 100}),
    ("S2", "② 급락 손절", {"stop": 0.20}),
    ("S3", "①+③", {"trend": 100, "fast": True}),
    ("S4", "①+②+③", {"trend": 100, "stop": 0.20, "fast": True}),
    ("S4-s15", "①+②+③, 손절 15%", {"trend": 100, "stop": 0.15, "fast": True}),
    ("S4-s25", "①+②+③, 손절 25%", {"trend": 100, "stop": 0.25, "fast": True}),
    ("S4-m50", "①+②+③, 추세선 50일", {"trend": 50, "stop": 0.20, "fast": True}),
]
WINDOWS = [("2018 하락", "2018-01-26", "2018-10-31"), ("2020 폭락", "2020-01-17", "2020-03-19"),
           ("2020 반등", "2020-03-19", "2020-12-30"), ("2021~22 하락", "2021-06-25", "2022-09-30"),
           ("2023 SG 폭락", "2023-04-21", "2023-05-31"), ("2024 하반기", "2024-07-01", "2024-12-30"),
           ("2025 상승", "2025-03-31", "2025-12-30")]


class Prices:
    """후보 종목(U_t 합집합) 수정종가와 이동평균·20일 고점."""

    def __init__(self, ctx: Ctx) -> None:
        c = ctx.close().sort_index()
        c = c.where(c > 0).ffill()
        c.index = pd.DatetimeIndex(c.index).as_unit("ns")
        self.C = c
        self.days = c.index
        self.k = {d: i for i, d in enumerate(self.days)}
        self.ma = {n: c.rolling(n, min_periods=n).mean() for n in (FAST, 50, 100)}
        self.h20 = c.rolling(HIGH, min_periods=HIGH).max()
        self._trig: dict = {}

    def triggers(self, stop: float) -> pd.DataFrame:
        """종가 < 20일 고점 × (1 − stop)이 되는 첫날 (조건이 이어지는 동안은 첫날만)."""
        if stop not in self._trig:
            cond = self.C < (1.0 - stop) * self.h20  # NaN 비교는 False
            self._trig[stop] = cond & ~cond.shift(1, fill_value=False)
        return self._trig[stop]


def stop_events(px: Prices, trig: pd.DataFrame, eng_days, first_day: pd.Timestamp, known: set) -> dict:
    """손절 신호일 d → 엔진 이벤트 청산일(d 다음 거래일) → 종목 집합."""
    pos = {d: i for i, d in enumerate(eng_days)}
    ks, js = np.nonzero(trig.to_numpy())
    cols = trig.columns
    ev: dict[str, set] = {}
    for k, j in zip(ks, js):
        d = px.days[k]
        t = str(cols[j])
        if d < first_day or t not in known:
            continue
        i = pos.get(d.strftime("%Y-%m-%d"))
        if i is None or i + 1 >= len(eng_days):
            continue
        ev.setdefault(eng_days[i + 1], set()).add(t)
    return ev


def block_matrix(px: Prices, D: pd.DatetimeIndex, opt: dict) -> pd.DataFrame:
    """리밸런싱 신호일 × 종목: 장치로 제외하면 True."""
    C = px.C.reindex(D)
    out = np.zeros(C.shape, dtype=bool)
    trend, stop, fast = opt.get("trend"), opt.get("stop"), bool(opt.get("fast", False))
    m20 = px.ma[FAST].reindex(D)
    kd = np.array([px.k[d] for d in D])
    if trend:
        ma = px.ma[trend].reindex(D)
        below = (C < ma * (1.0 - BAND)).to_numpy()
        if fast:
            out |= below & (C < m20).to_numpy()
        else:  # 완충 구간: 100일선 위로 올라와야 다시 후보
            above = ((C > ma) | ma.isna()).to_numpy()
            state = np.zeros(C.shape[1], dtype=bool)
            for r in range(len(D)):
                state = np.where(state, ~above[r], below[r])
                out[r] |= state
    if stop:
        T = px.triggers(stop).to_numpy()
        last = np.where(T, np.arange(len(px.days))[:, None], -10 ** 9)
        last = np.maximum.accumulate(last, axis=0)[kd]  # 신호일까지 마지막 손절일 위치
        start = kd[0] - BAN  # 첫 신호일보다 20거래일 넘게 앞선 손절은 무시
        if not fast:
            out |= (last >= start) & ((kd[:, None] - last) <= BAN)
        else:
            above20 = (C > m20).to_numpy()
            state = np.zeros(C.shape[1], dtype=bool)
            seen = np.full(C.shape[1], start)
            for r in range(len(D)):
                state |= last[r] > seen
                seen = np.maximum(seen, last[r])
                state &= ~(above20[r] & (last[r] < kd[r]))
                out[r] |= state
    return pd.DataFrame(out, index=D, columns=C.columns)


def with_blocks(snaps, B: pd.DataFrame, trade_idx: set):
    """리밸런싱 신호일 스냅샷의 crash_flag에 장치 제외를 더한다. 반환: (스냅샷, rank ≤ 40 후보 중 새로 제외된 평균 수)."""
    out, extra = [], []
    for i, s in enumerate(snaps):
        if i in trade_idx:
            blk = B.loc[pd.Timestamp(s.date)].reindex(s.tickers, fill_value=False).to_numpy(dtype=bool)
            extra.append(int((blk & ~s.crash_flag & (s.rank <= 40)).sum()))
            out.append(replace(s, crash_flag=s.crash_flag | blk))
        else:
            out.append(s)
    return out, float(np.mean(extra)) if extra else 0.0


def run_eval(ctx: Ctx, snaps, b, every: int, extra: dict):
    """ctx.engine을 잠시 바꿔 M9 청산에 손절 이벤트를 더한다 (positions 만들기와 재생 모두 같은 엔진)."""
    m9_ev = ctx.m9.liquidations or {}
    merged = {d: set(v) for d, v in m9_ev.items()}
    for d, v in extra.items():
        merged.setdefault(d, set()).update(v)
    merged = {d: frozenset(v) for d, v in merged.items()}

    def eng(with_m9: bool, ledger: bool):
        if with_m9:
            return build_engine(ctx.inputs, ctx.cfg, record_ledger=ledger, cooldowns=ctx.m9.cooldowns, event_liquidations=merged)
        return build_engine(ctx.inputs, ctx.cfg, record_ledger=ledger, event_liquidations={d: frozenset(v) for d, v in extra.items()})

    ctx.engine = eng
    try:
        pos = buffer_run(ctx, snaps, buffer=b, mode="no_exposure", with_m9=True, trade_every=every)
        e = eng(True, True)
        rep = replay(pos, ctx.dates, e, ctx.last_period)
        s = summary(rep, ctx.inputs.bm_close, pos)
    finally:
        del ctx.engine
    return s, rep.nav, e.ledger


def stop_stats(ledger: list, extra: dict) -> tuple[int, int]:
    """손절 이벤트일에 들고 있던 종목 수, 그중 그날 체결이 거부된(하한가·정지) 수."""
    by_day: dict[str, dict] = {}
    order = []
    for row in ledger:
        d = row["date"]
        if d not in by_day:
            by_day[d] = {"rejected": set(), "end": {}}
            order.append(d)
        by_day[d]["rejected"] |= set(row["rejected"])
        by_day[d]["end"] = row["holdings"]
    hit = delayed = 0
    prev: dict = {}
    for d in order:
        ev = extra.get(d)
        if ev:
            h = ev & set(prev)
            hit += len(h)
            delayed += len(h & by_day[d]["rejected"])
        prev = by_day[d]["end"]
    return hit, delayed


def half_mdd(nav: pd.Series) -> tuple[float, float]:
    cut = nav[nav.index <= SPLIT].index[-1]
    a = pd.concat([pd.Series([1.0], index=[nav.index[0] - pd.Timedelta(days=1)]), nav[nav.index <= cut]])
    b = nav[nav.index >= cut]
    return float((1.0 - a / a.cummax()).max()), float((1.0 - b / b.cummax()).max())


def windows(nav: pd.Series) -> dict:
    out = {}
    for name, a, b in WINDOWS:
        s = nav.loc[a:b]
        out[name] = float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) > 1 else None
    return out


def save(name: str, rec: dict, nav: pd.Series) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": nav.index, "nav": nav.to_numpy()}).to_parquet(OUT / f"nav_{name}.parquet", index=False)
    (OUT / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, default=float), encoding="utf-8")


def load(name: str):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("plan", "exit|T7|S0"):  # 실행 전 계획 행 (처음 한 번)
        for base, (blabel, _) in BASES.items():
            for vid, label, opt in VARIANTS:
                ledger.plan(f"exit|{base}|{vid}", "종목 이탈 장치 (T7·T3 위)", f"{blabel} + {label} {json.dumps(opt, ensure_ascii=False)}",
                            "사용자 결정(11/19): 오락가락·갑작스런 폭락·V자 반등 놓침을 피해야 MDD가 줄고 수익이 오른다")
    t0 = time.time()
    ctx = Ctx(ALL_YEARS)
    cost_unit = 2 * (float(ctx.cfg["costs"]["commission_per_side"]) + float(ctx.cfg["costs"]["slippage_per_side"])) \
        + float(ctx.cfg["costs"]["sell_tax"])
    base_snaps = ctx.snaps(signals=smooth_signals(ctx.signals, 4), key="sm4")
    px = Prices(ctx)
    probe = ctx.engine(False, False)
    eng_days, known = probe.days, set(probe.col_index)
    first_day = pd.Timestamp(ctx.dates[0])
    b80 = replace(ctx.buffer, exit_rank_greater_than=80)
    dates = pd.DatetimeIndex(pd.to_datetime(ctx.dates))
    print(f"입력 읽음 ({time.time() - t0:.0f}초)", flush=True)
    ev_cache: dict = {}

    for base, (blabel, every) in BASES.items():
        trade_idx = [i for i in range(len(dates)) if i % every == 0]
        D = dates[trade_idx]
        for vid, label, opt in VARIANTS:
            name = f"{base}_{vid}"
            if load(name) is not None:
                continue
            t = time.time()
            if opt:
                snaps, n_block = with_blocks(base_snaps, block_matrix(px, D, opt), set(trade_idx))
            else:
                snaps, n_block = base_snaps, 0.0
            stop = opt.get("stop")
            if stop and stop not in ev_cache:
                ev_cache[stop] = stop_events(px, px.triggers(stop), eng_days, first_day, known)
            extra = ev_cache.get(stop, {}) if stop else {}
            s, nav, led = run_eval(ctx, snaps, b80, every, extra)
            m = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v)
                 for k, v in s.items()}
            c1, c2 = split_cagr(nav)
            d1, d2 = half_mdd(nav)
            hit, delayed = stop_stats(led, extra) if extra else (0, 0)
            rec = {**m, "base": base, "variant": vid, "label": label, "opt": opt, "cagr_2016_2020": c1, "cagr_2021_2025": c2,
                   "mdd_2016_2020": d1, "mdd_2021_2025": d2, "cost_est": m["annual_turnover"] * cost_unit,
                   "stops_hit": hit, "stops_delayed": delayed, "blocked_top40": n_block, "windows": windows(nav),
                   "yearly": yearly(nav)}
            save(name, rec, nav)
            print(f"  {base} {vid} {label}: CAGR {m['cagr']:+.2%} (전반 {c1:+.2%}, 후반 {c2:+.2%}), MDD {m['mdd']:.1%} "
                  f"(전반 {d1:.1%}, 후반 {d2:.1%}), 회전율 {m['annual_turnover']:.1f}, 손절 {hit}건(미뤄짐 {delayed}) "
                  f"({time.time() - t:.0f}초)", flush=True)
        ref = json.loads((TOV / f"{base}.json").read_text(encoding="utf-8"))
        d = abs(load(f"{base}_S0")["cagr"] - ref["cagr"])
        print(f"대조: {base} S0 CAGR {load(f'{base}_S0')['cagr']:+.6%} vs turnover.py {base} {ref['cagr']:+.6%} (차이 {d:.1e})")
        if d > 1e-12:
            raise RuntimeError(f"{base} S0가 turnover.py 결과와 다르다")

    days = pd.DatetimeIndex(pd.read_parquet(OUT / "nav_T7_S0.parquet")["date"])
    k = benchmark_summary(ctx.inputs.bm_close, days)
    judg, rows = {}, []
    for base in BASES:
        b0 = load(f"{base}_S0")
        for vid, label, _ in VARIANTS:
            r = load(f"{name_of(base, vid)}")
            ok1 = r["mdd"] < b0["mdd"]
            ok2 = r["cagr"] >= b0["cagr"] - 0.01
            ok3 = (r["mdd_2016_2020"] < b0["mdd_2016_2020"] and r["mdd_2021_2025"] < b0["mdd_2021_2025"]
                   and r["cagr_2016_2020"] >= b0["cagr_2016_2020"] - 0.01 and r["cagr_2021_2025"] >= b0["cagr_2021_2025"] - 0.01)
            j = {"(1) MDD < S0": ok1, "(2) CAGR ≥ S0 − 1%p": ok2, "(3) 두 구간 모두": ok3} if vid != "S0" else {}
            judg[f"{base}|{vid}"] = {"label": label, **j, "pass": bool(j and all(j.values())),
                                     **{c: r[c] for c in ("cagr", "mdd", "ir", "annual_turnover", "cost_est", "final_nav", "cagr_2016_2020",
                                                          "cagr_2021_2025", "mdd_2016_2020", "mdd_2021_2025", "stops_hit",
                                                          "stops_delayed", "blocked_top40")}}
            rows.append({"base": base, "variant": vid, "label": label, **{c: r[c] for c in (
                "cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "mdd_2016_2020", "mdd_2021_2025", "ir", "annual_turnover",
                "cost_est", "final_nav", "stops_hit", "stops_delayed", "blocked_top40")}, "pass": judg[f"{base}|{vid}"]["pass"],
                **{f"w:{n}": v for n, v in r["windows"].items()}, **{f"y{y}": v for y, v in r["yearly"].items()}})
    rd = pd.DataFrame(rows)
    rd.to_csv(HERE / "exit_runs.csv", index=False)
    picks = {}
    for base in BASES:
        cands = [(v, judg[f"{base}|{v}"]["cagr"]) for v, *_ in VARIANTS if judg[f"{base}|{v}"]["pass"]]
        picks[base] = max(cands, key=lambda z: z[1])[0] if cands else None
    (HERE / "exit_judgment.json").write_text(json.dumps({
        "period": [str(days[0].date()), str(days[-1].date())], "kospi200": {"cagr": float(k["cagr"]), "mdd": float(k["mdd"])},
        "rules": {"band": BAND, "ban_days": BAN, "fast_ma": FAST, "high_window": HIGH}, "variants": judg, "confirm_candidates": picks},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not ledger_has("result", "exit|판정"):
        for key, v in judg.items():
            ledger.result(f"exit|{key}", **v)
        ledger.result("exit|판정", confirm_candidates=picks)

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    show = rd.copy()
    pct = ["cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "mdd_2016_2020", "mdd_2021_2025", "cost_est"] + \
        [c for c in show.columns if c.startswith(("w:", "y")) and (c.startswith("w:") or c[1:].isdigit())]
    for c in pct:
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(1)
    for c in ("ir", "annual_turnover", "final_nav", "blocked_top40"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    print(f"\n[2016-01-11 ~ {days[-1].date()}, 노출 조절 끔, M9, C 엔진] 코스피 200 CAGR {k['cagr']:+.1%}, MDD {k['mdd']:.1%}")
    print(show[["base", "variant", "label", "cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "mdd_2016_2020", "mdd_2021_2025",
                "annual_turnover", "cost_est", "final_nav", "stops_hit", "stops_delayed", "blocked_top40", "pass"]].to_string(index=False))
    print("\n[하락·상승 구간 수익률 %]")
    print(show[["base", "variant"] + [c for c in show.columns if c.startswith("w:")]].to_string(index=False))
    print("\n[연도별 수익률 %]")
    print(show[["base", "variant"] + [c for c in show.columns if c.startswith("y") and c[1:].isdigit()]].to_string(index=False))
    print(f"\n[판정] (실행 전 고정) 확인 대상: T7 기준 {picks['T7'] or '없음'}, T3 기준 {picks['T3'] or '없음'}")
    for key, v in judg.items():
        if key.endswith("|S0"):
            continue
        marks = ", ".join(f"{q} {'만족' if v[q] else '미달'}" for q in v if q.startswith("("))
        print(f"  {key} {v['label']}: {marks}")
    print("done")


def name_of(base: str, vid: str) -> str:
    return f"{base}_{vid}"


if __name__ == "__main__":
    main()
