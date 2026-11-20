r"""여러 이탈선: T7·T3 위에서 종목마다 20·50·100·200일선 중 몇 개 위에 있는지로 비중을 줄이고 늘린다 (11/19 사용자 결정).

사용자 결정(11/19): 이탈선은 여러 개 써도 된다. exitrule.py(17절)는 선 하나로 전부 팔고 전부 사서,
2020년 바닥에서 후보가 100% 제외(전부 현금)됐고 반등 중 눌림에 다시 95%가 제외돼 V자 반등을 놓쳤다.
여러 선을 쓰면 빠른 선(20·50일)으로 먼저 조금 돌아오고, 느린 선(100·200일)까지 깨져야 다 판다.

실행 (저장소 루트, PowerShell, 사용자가 돌린다, 5~10분, 창 1개):
  .\.venv\Scripts\python.exe -I explore\b09\run.py exitmulti.py
  끊기면 같은 명령을 다시 돌린다 (끝난 실행은 건너뛴다).

기준 (16절): T7 = 4주 평균 순위 + 청산 80 + 격주, T3 = 4주 평균 순위 + 청산 80 + 매주.
모두 노출 조절 끔(E = 1), crash_flag 차단, 종목 10%·업종 30% 한도, M9, C 엔진, 2016~2025 연속.

선 점수 (실행 전 고정 11/19): 리밸런싱 신호일(T7은 2주마다, T3은 매주) 종가가 자기 20·50·100·200일선 중 몇 개 위에 있나.
점수 = 위에 있는 선 수 ÷ 계산되는 선 수 (상장이 짧아 선이 없으면 있는 선만, 하나도 없으면 1). 수정종가, 거래 없는 날은 직전 값.

구성 (기준마다 5개, 합 10개)
- M0 장치 없음 (기준. exitrule.py S0 = turnover.py T7·T3과 같아야 한다 — 다르면 멈춘다)
- M1 다수결 이탈: 점수 ≤ 1/4 (4개 선 중 3개 이상 아래)이면 제외한다 (crash_flag처럼: 들고 있으면 팔고, 빈자리는 다음 순위로).
- M2 단계 비중: 점수 0(모든 선 아래)이면 제외하고, 나머지는 그 종목 비중 × 점수로 줄인다 (줄인 몫은 현금).
  예: 20일선 위·나머지 아래면 1/4만, 20·50일선 위면 1/2만 담는다. 비중은 리밸런싱 신호일에만 바꾼다.
- M3 = M2 + 급락 손절: 종가 < 최근 20거래일 고점 × 0.80인 첫날 d → d 다음 거래일 종가 매도 (엔진 이벤트 청산, 하한가·정지면 다음 날로).
  손절 뒤 리밸런싱 신호일 종가가 20일선 위이면 다시 후보 (exitrule.py ②+③과 같다).
- M4 = M1 + 급락 손절

판정 (실행 전 고정 11/19, exitrule.py와 같다), 같은 기준의 M0과 비교:
- (1) 10년 MDD < M0  (2) 10년 CAGR ≥ M0 − 1%p  (3) 2016~2020과 2021~2025 각각에서도 (1)·(2)
- 넘음 = (1)·(2)·(3) 모두. 확인 대상 = 기준마다 넘은 구성 중 10년 CAGR이 가장 높은 것.
- 참고: 코스피 종합(18절, A 참고용 파일)·코스피 200 대비 연 초과수익과 IR, 평균 투자 비중, 회전율, 비용 추정, 손절 건수,
  하락·상승 구간 수익, 연도별. 구간·연도는 exitrule.py와 같다.
- 2016~2025는 여러 번 본 구간이라, 넘어도 forward test(PM이 허락하면 2026 홀드아웃)로 확인해야 한다.
- 탐색 결과다 (R-15). 결과: exitm_runs.csv, exitm_judgment.json, runs/exitm/, 원장 exitm|*.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

from common import ALL_YEARS, HERE, Ctx, Ledger, buffer_run, replay, summary
import exitrule as ex
from bench_kospi import KOSPI_PATH, stats as kospi_stats
import numpy as np
import pandas as pd
from src.portfolio.buffer_rule import buffer_rule_weights
from src.portfolio.engine_adapter import build_engine
from turnover import ledger_has, smooth_signals, split_cagr, yearly

OUT = HERE / "runs" / "exitm"
LINES = (20, 50, 100, 200)
STOP = 0.20
VARIANTS = [
    ("M0", "장치 없음", {}),
    ("M1", "다수결 이탈 (4개 선 중 3개 이상 아래면 제외)", {"vote_exit": True}),
    ("M2", "단계 비중 (위에 있는 선 수 ÷ 4, 0개면 제외)", {"graded": True}),
    ("M3", "단계 비중 + 급락 손절", {"graded": True, "stop": STOP}),
    ("M4", "다수결 이탈 + 급락 손절", {"vote_exit": True, "stop": STOP}),
]


def vote_frac(px: ex.Prices, D: pd.DatetimeIndex) -> pd.DataFrame:
    """리밸런싱 신호일 × 종목: 종가가 위에 있는 선 수 ÷ 계산되는 선 수."""
    C = px.C.reindex(D)
    c = C.to_numpy()
    above = np.zeros(C.shape)
    avail = np.zeros(C.shape)
    for n in LINES:
        ma = px.ma[n].reindex(D).to_numpy()
        ok = np.isfinite(ma) & np.isfinite(c)
        avail += ok
        above += ok & (c > ma)
    frac = np.where(avail > 0, above / np.maximum(avail, 1.0), 1.0)
    return pd.DataFrame(frac, index=D, columns=C.columns)


def make_choose(scale: dict):
    """버퍼 규칙(단계 1~4)으로 고른 뒤 종목 비중에 선 점수를 곱한다 (리밸런싱 주에만 불린다)."""
    def choose(snap, holdings, buffer, cooldown):
        sig = pd.DataFrame({"ticker": snap.tickers, "rank": snap.rank, "crash_flag": snap.crash_flag.astype(np.int8)})
        w = buffer_rule_weights(holdings, sig, snap.tickers, snap.tickers[cooldown], buffer)
        sc = scale.get(snap.date)
        if sc is not None and len(w):
            w = w * sc.reindex(w.index).fillna(1.0).to_numpy()
        return w
    return choose


def run_eval(ctx: Ctx, snaps, b, every: int, extra: dict, choose_fn=None):
    """exitrule.run_eval과 같다 (M9 청산 + 손절 이벤트). choose_fn을 넘길 수 있다."""
    merged = {d: set(v) for d, v in (ctx.m9.liquidations or {}).items()}
    for d, v in extra.items():
        merged.setdefault(d, set()).update(v)
    merged = {d: frozenset(v) for d, v in merged.items()}

    def eng(with_m9: bool, ledger: bool):
        if with_m9:
            return build_engine(ctx.inputs, ctx.cfg, record_ledger=ledger, cooldowns=ctx.m9.cooldowns, event_liquidations=merged)
        return build_engine(ctx.inputs, ctx.cfg, record_ledger=ledger, event_liquidations={d: frozenset(v) for d, v in extra.items()})

    ctx.engine = eng
    try:
        pos = buffer_run(ctx, snaps, buffer=b, mode="no_exposure", with_m9=True, trade_every=every, choose_fn=choose_fn)
        e = eng(True, True)
        rep = replay(pos, ctx.dates, e, ctx.last_period)
        s = summary(rep, ctx.inputs.bm_close, pos)
    finally:
        del ctx.engine
    return s, rep.nav, e.ledger


def save(name: str, rec: dict, nav: pd.Series) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": nav.index, "nav": nav.to_numpy()}).to_parquet(OUT / f"nav_{name}.parquet", index=False)
    (OUT / f"{name}.json").write_text(json.dumps(rec, ensure_ascii=False, default=float), encoding="utf-8")


def load(name: str):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    ledger = Ledger(HERE / "ledger.jsonl")
    if not ledger_has("plan", "exitm|T7|M0"):  # 실행 전 계획 행 (처음 한 번)
        for base, (blabel, _) in ex.BASES.items():
            for vid, label, opt in VARIANTS:
                ledger.plan(f"exitm|{base}|{vid}", "여러 이탈선 (T7·T3 위)", f"{blabel} + {label} {json.dumps(opt, ensure_ascii=False)}",
                            "사용자 결정(11/19): 이탈선은 여러 개 써도 된다. 선 하나(17절)는 바닥에서 전부 현금, 반등 중 다시 제외돼 V자 반등을 놓쳤다")
    t0 = time.time()
    ctx = Ctx(ALL_YEARS)
    cost_unit = 2 * (float(ctx.cfg["costs"]["commission_per_side"]) + float(ctx.cfg["costs"]["slippage_per_side"])) \
        + float(ctx.cfg["costs"]["sell_tax"])
    base_snaps = ctx.snaps(signals=smooth_signals(ctx.signals, 4), key="sm4")
    px = ex.Prices(ctx)
    px.ma[200] = px.C.rolling(200, min_periods=200).mean()
    probe = ctx.engine(False, False)
    eng_days, known = probe.days, set(probe.col_index)
    first_day = pd.Timestamp(ctx.dates[0])
    b80 = replace(ctx.buffer, exit_rank_greater_than=80)
    dates = pd.DatetimeIndex(pd.to_datetime(ctx.dates))
    stop_ev = ex.stop_events(px, px.triggers(STOP), eng_days, first_day, known)
    k = pd.read_parquet(KOSPI_PATH)
    kospi = pd.Series(k["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(k["date"])).as_unit("ns"))
    k200 = ctx.inputs.bm_close.copy()
    k200.index = pd.DatetimeIndex(pd.to_datetime(k200.index)).as_unit("ns")
    cal = k200.index[k200.index <= pd.Timestamp("2025-12-30")]
    fills = pd.DatetimeIndex([cal[cal.searchsorted(d, side="right")] for d in dates])
    print(f"입력 읽음 ({time.time() - t0:.0f}초)", flush=True)

    for base, (blabel, every) in ex.BASES.items():
        trade_idx = [i for i in range(len(dates)) if i % every == 0]
        D = dates[trade_idx]
        F = vote_frac(px, D)
        S = ex.block_matrix(px, D, {"stop": STOP, "fast": True})
        for vid, label, opt in VARIANTS:
            name = f"{base}_{vid}"
            if load(name) is not None:
                continue
            t = time.time()
            B = pd.DataFrame(False, index=D, columns=F.columns)
            if opt.get("vote_exit"):
                B |= F <= 0.25 + 1e-9
            if opt.get("graded"):
                B |= F <= 1e-12
            if opt.get("stop"):
                B |= S
            snaps, n_block = ex.with_blocks(base_snaps, B, set(trade_idx)) if opt else (base_snaps, 0.0)
            choose = make_choose({d.strftime("%Y-%m-%d"): F.loc[d] for d in D}) if opt.get("graded") else None
            extra = stop_ev if opt.get("stop") else {}
            s, nav, led = run_eval(ctx, snaps, b80, every, extra, choose)
            m = {kk: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else v)
                 for kk, v in s.items()}
            c1, c2 = split_cagr(nav)
            d1, d2 = ex.half_mdd(nav)
            hit, delayed = ex.stop_stats(led, extra) if extra else (0, 0)
            ks = kospi_stats(nav, {"kospi": kospi, "k200": k200}, fills)
            rec = {**m, "base": base, "variant": vid, "label": label, "opt": opt, "cagr_2016_2020": c1, "cagr_2021_2025": c2,
                   "mdd_2016_2020": d1, "mdd_2021_2025": d2, "cost_est": m["annual_turnover"] * cost_unit,
                   "invested": 1.0 - m["mean_cash"], "stops_hit": hit, "stops_delayed": delayed, "blocked_top40": n_block,
                   "excess_kospi": ks["excess_kospi"], "ir_kospi": ks["ir_kospi"], "excess_k200": ks["excess_k200"],
                   "ir_k200": ks["ir_k200"], "windows": ex.windows(nav), "yearly": yearly(nav)}
            save(name, rec, nav)
            print(f"  {base} {vid} {label}: CAGR {m['cagr']:+.2%} (전반 {c1:+.2%}, 후반 {c2:+.2%}), MDD {m['mdd']:.1%} "
                  f"(전반 {d1:.1%}, 후반 {d2:.1%}), 투자 {1 - m['mean_cash']:.0%}, 회전율 {m['annual_turnover']:.1f}, "
                  f"코스피 대비 {ks['excess_kospi']:+.1%}·IR {ks['ir_kospi']:+.2f} ({time.time() - t:.0f}초)", flush=True)
        ref = ex.load(f"{base}_S0")
        d = abs(load(f"{base}_M0")["cagr"] - ref["cagr"])
        print(f"대조: {base} M0 CAGR {load(f'{base}_M0')['cagr']:+.6%} vs exitrule.py {base} S0 {ref['cagr']:+.6%} (차이 {d:.1e})")
        if d > 1e-12:
            raise RuntimeError(f"{base} M0가 exitrule.py S0와 다르다")

    judg, rows = {}, []
    for base in ex.BASES:
        b0 = load(f"{base}_M0")
        for vid, label, _ in VARIANTS:
            r = load(f"{base}_{vid}")
            j = {} if vid == "M0" else {
                "(1) MDD < M0": r["mdd"] < b0["mdd"],
                "(2) CAGR ≥ M0 − 1%p": r["cagr"] >= b0["cagr"] - 0.01,
                "(3) 두 구간 모두": bool(r["mdd_2016_2020"] < b0["mdd_2016_2020"] and r["mdd_2021_2025"] < b0["mdd_2021_2025"]
                                    and r["cagr_2016_2020"] >= b0["cagr_2016_2020"] - 0.01
                                    and r["cagr_2021_2025"] >= b0["cagr_2021_2025"] - 0.01)}
            cols = ("cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "mdd_2016_2020", "mdd_2021_2025", "ir", "annual_turnover",
                    "cost_est", "invested", "final_nav", "stops_hit", "stops_delayed", "blocked_top40", "excess_kospi", "ir_kospi",
                    "excess_k200", "ir_k200")
            judg[f"{base}|{vid}"] = {"label": label, **j, "pass": bool(j and all(j.values())), **{c: r[c] for c in cols}}
            rows.append({"base": base, "variant": vid, "label": label, **{c: r[c] for c in cols},
                         "pass": judg[f"{base}|{vid}"]["pass"], **{f"w:{n}": v for n, v in r["windows"].items()},
                         **{f"y{y}": v for y, v in r["yearly"].items()}})
    rd = pd.DataFrame(rows)
    rd.to_csv(HERE / "exitm_runs.csv", index=False)
    picks = {}
    for base in ex.BASES:
        cands = [(v, judg[f"{base}|{v}"]["cagr"]) for v, *_ in VARIANTS if judg[f"{base}|{v}"]["pass"]]
        picks[base] = max(cands, key=lambda z: z[1])[0] if cands else None
    (HERE / "exitm_judgment.json").write_text(json.dumps({"lines": LINES, "stop": STOP, "variants": judg, "confirm_candidates": picks},
                                                         ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not ledger_has("result", "exitm|판정"):
        for key, v in judg.items():
            ledger.result(f"exitm|{key}", **v)
        ledger.result("exitm|판정", confirm_candidates=picks)

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    show = rd.copy()
    pct = ["cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "mdd_2016_2020", "mdd_2021_2025", "cost_est", "invested",
           "excess_kospi", "excess_k200"] + [c for c in show.columns if c.startswith("w:") or (c.startswith("y") and c[1:].isdigit())]
    for c in pct:
        show[c] = (pd.to_numeric(show[c], errors="coerce") * 100).round(1)
    for c in ("ir", "ir_kospi", "ir_k200", "annual_turnover", "final_nav", "blocked_top40"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    print("\n[2016-01-11 ~ 2025-12-30, 노출 조절 끔, M9, C 엔진]")
    print(show[["base", "variant", "cagr", "cagr_2016_2020", "cagr_2021_2025", "mdd", "mdd_2016_2020", "mdd_2021_2025", "invested",
                "annual_turnover", "cost_est", "final_nav", "stops_hit", "excess_kospi", "ir_kospi", "pass"]].to_string(index=False))
    print("\n[하락·상승 구간 수익률 %]")
    print(show[["base", "variant"] + [c for c in show.columns if c.startswith("w:")]].to_string(index=False))
    print("\n[연도별 수익률 %]")
    print(show[["base", "variant"] + [c for c in show.columns if c.startswith("y") and c[1:].isdigit()]].to_string(index=False))
    print(f"\n[판정] (실행 전 고정) 확인 대상: T7 기준 {picks['T7'] or '없음'}, T3 기준 {picks['T3'] or '없음'}")
    for key, v in judg.items():
        if key.endswith("|M0"):
            continue
        marks = ", ".join(f"{q} {'만족' if v[q] else '미달'}" for q in v if q.startswith("("))
        print(f"  {key} {v['label']}: {marks}")
    print("done")


if __name__ == "__main__":
    main()
