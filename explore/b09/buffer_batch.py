"""G7 트랙(M8 변형)과 구성 트랙(신호·버퍼 변형) 선별. 선별 연도(2016~2020)만 본다.

G7 비교 대상: 사전등록 버퍼 규칙의 M8·M9 미적용 실행 (M8 변형) / 같은 구성의 미적용 실행 (구성 변형).
"""
import json
import math
import sys
import time
from dataclasses import replace

from common import HERE, SEARCH_YEARS, Ctx, Ledger, brief, buffer_run, g7_check, rank_from_score, top_fraction_flag
import numpy as np
import pandas as pd

BATCH = sys.argv[1] if len(sys.argv) > 1 else "b1"
ledger = Ledger(HERE / "ledger.jsonl")
ctx = Ctx(SEARCH_YEARS)
risk, buf, kpi = ctx.risk, ctx.buffer, ctx.kpi
bm = ctx.inputs.bm_close
ma200 = bm.rolling(200, min_periods=200).mean()
diag = ctx.diag()
close = ctx.close()
ret = close.pct_change(fill_method=None)

base = ctx.snaps()
base_off = brief(ctx.evaluate(buffer_run(ctx, base, mode="off"), with_m9=False))
base_full = brief(ctx.evaluate(buffer_run(ctx, base, mode="full"), with_m9=True))


# ---- 변형 정의 -----------------------------------------------------------------
def snaps_for_risk(r, key):
    return ctx.snaps(risk=r, key=key)


def below_ma(snap):
    t = pd.Timestamp(snap.date)
    return bool(bm[t] < ma200[t])


def exp_trend(kind):
    def f(i, snap, e, nav, chosen):
        if not below_ma(snap):
            return e if kind != "c" else 1.0 if snap.regime_mc == 0 else min(1.0, risk.momentum_crash_regime.exposure_cap)
        return {"a": e * 0.5, "b": min(e, 0.3), "c": 0.3}[kind]
    return f


def exp_dd(th, factor=0.5):
    def f(i, snap, e, nav, chosen):
        if len(nav) == 0:
            return e
        dd = 1.0 - float(nav.iloc[-1]) / float(nav.max())
        return e * factor if dd > th else e
    return f


def compose(*fs):
    def f(i, snap, e, nav, chosen):
        for g in fs:
            e = g(i, snap, e, nav, chosen)
        return e
    return f


def any_block(*fs):
    return lambda i, snap: np.logical_or.reduce([g(i, snap) for g in fs])


def exp_pv(target):
    def f(i, snap, e, nav, chosen):
        if not chosen:
            return e
        t = pd.Timestamp(snap.date)
        win = ret.loc[:t].iloc[-60:][[c for c in chosen if c in ret.columns]]
        sig = float(win.mean(axis=1).std(ddof=1) * math.sqrt(252))
        new = min(risk.exposure_max, max(risk.exposure_min, target / sig)) if sig > 0 else risk.exposure_max
        return min(new, risk.momentum_crash_regime.exposure_cap) if snap.regime_mc == 1 else new
    return f


v60 = diag.set_index(["date", "ticker"])["base_flag"]


def block_v60(i, snap):
    t = pd.Timestamp(snap.date)
    return v60.reindex(pd.MultiIndex.from_product([[t], snap.tickers])).fillna(0).to_numpy().astype(bool)


def block_v20(q):
    return lambda i, snap: snap.vol_20 > q


def sig_with(**cols):
    s = ctx.signals.copy()
    for k, v in cols.items():
        s[k] = v
    return s


sig = ctx.signals
dmerge = sig[["date", "ticker"]].merge(diag, on=["date", "ticker"], how="left")
assert dmerge["base_flag"].notna().all()


def smooth_score(weeks):
    s = sig[["date", "ticker", "rank_score"]].sort_values(["ticker", "date"])
    s["sm"] = s.groupby("ticker")["rank_score"].transform(lambda x: x.rolling(weeks, min_periods=1).mean())
    return sig[["date", "ticker"]].merge(s[["date", "ticker", "sm"]], on=["date", "ticker"], how="left")["sm"].to_numpy()


VARIANTS = []  # (id, track, change, why, runner) ; runner() -> (full_summary, off_summary or None)


def add(vid, track, change, why, runner):
    VARIANTS.append((vid, track, change, why, runner))


def m8(snaps=None, r=None, **kw):
    def run():
        full = brief(ctx.evaluate(buffer_run(ctx, snaps or base, risk=r, mode="full", **kw), with_m9=True))
        return full, base_off
    return run


def construction(signals=None, b=None, key=None, **kw):
    def run():
        sn = base if signals is None else ctx.snaps(signals=signals, key=key)
        full = brief(ctx.evaluate(buffer_run(ctx, sn, buffer=b, mode="full", **kw), with_m9=True))
        off = brief(ctx.evaluate(buffer_run(ctx, sn, buffer=b, mode="off", **kw), with_m9=False))
        return full, off
    return run


if BATCH == "b1":
    add("C0", "control", "없음", "기준", lambda: (base_full, base_off))
    for f in (0.75, 1.25, 1.5):
        r = replace(risk, vol_target_annual=risk.vol_target_annual * f)
        add(f"M-vt-{f}", "G7", f"vol_target_annual × {f}", "노출 수준", m8(snaps_for_risk(r, f"vt{f}"), r))
    for em in (0.0, 0.5):
        r = replace(risk, exposure_min=em)
        add(f"M-emin-{em}", "G7", f"exposure_min {em}", "노출 하한", m8(snaps_for_risk(r, f"em{em}"), r))
    for cap in (0.5, 1.0):
        r = replace(risk, momentum_crash_regime=replace(risk.momentum_crash_regime, exposure_cap=cap))
        add(f"M-rg-{cap}", "G7", f"국면 상한 {cap}", "반등 국면 대응", m8(snaps_for_risk(r, f"rg{cap}"), r))
    r = replace(risk, vol_lookback_days=60)
    add("M-lb-60", "G7", "변동성 창 60일", "느린 노출 변화", m8(snaps_for_risk(r, "lb60"), r))
    for k in ("a", "b", "c"):
        add(f"M-tr-{k}", "G7", {"a": "200일선 아래 E×0.5", "b": "200일선 아래 E≤0.3", "c": "200일선 위 1, 아래 0.3 (변동성 목표 대신)"}[k],
            "추세 필터", m8(exposure_fn=exp_trend(k)))
    for th in (0.15, 0.20):
        add(f"M-dd-{th}", "G7", f"NAV 낙폭 > {th}면 E×0.5", "포트폴리오 낙폭 제어", m8(exposure_fn=exp_dd(th)))
    for tg in (0.157263, 0.25):
        add(f"M-pv-{tg}", "G7", f"포트폴리오 60일 변동성 목표 {tg}", "보유 종목 변동성 기준", m8(exposure_fn=exp_pv(tg)))
    add("M-tag-v60", "G7", "M8 Step 1에 vol_60 상위 10% 차단", "위험 태그", m8(block_fn=block_v60))
    for q in (0.9, 0.8):
        add(f"M-tag-v20-{q}", "G7", f"M8 Step 1에 vol_20 백분위 > {q} 차단", "위험 태그", m8(block_fn=block_v20(q)))
    add("M-noexp", "G7", "노출 조절 끔 (G4 진단과 같음)", "참고", m8(snaps=base, r=None, exposure_fn=lambda i, s, e, n, c: 1.0))

    add("S-flag-v60", "구성", "crash_flag = vol_60 기준선", "E1", construction(sig_with(crash_flag=dmerge["base_flag"].to_numpy()), key="v60"))
    add("S-flag-union", "구성", "crash_flag = LightGBM ∪ vol_60", "두 플래그", construction(
        sig_with(crash_flag=(sig["crash_flag"].to_numpy() | dmerge["base_flag"].to_numpy()).astype("int8")), key="union"))
    add("S-flag-lgb20", "구성", "crash_flag = LightGBM 상위 20%", "플래그 비율", construction(
        sig_with(crash_flag=top_fraction_flag(sig, "crash_prob", 0.20).to_numpy()), key="lgb20"))
    add("S-flag-none", "구성", "crash_flag 없음", "참고", construction(sig_with(crash_flag=np.zeros(len(sig), dtype="int8")), key="none"))
    for col in ("b1_score", "b3_score"):
        s2 = sig_with(rank_score=dmerge[col].to_numpy())
        s2["rank"] = rank_from_score(s2, "rank_score").to_numpy()
        add(f"S-rank-{col[:2]}", "구성", f"rank 출처 = {col}", "베이스라인 순위", construction(s2, key=col))
    s2 = sig_with(rank_score=smooth_score(4))
    s2["rank"] = rank_from_score(s2, "rank_score").to_numpy()
    add("S-rank-sm4", "구성", "rank_score 4주 평균", "순위 안정 → 회전율", construction(s2, key="sm4"))
    for name, b in (("x60", replace(buf, exit_rank_greater_than=60)), ("x80", replace(buf, exit_rank_greater_than=80)),
                    ("e10", replace(buf, enter_rank_max=10)),
                    ("h10", replace(buf, enter_rank_max=10, exit_rank_greater_than=20, max_holdings=10)),
                    ("h30", replace(buf, enter_rank_max=30, exit_rank_greater_than=60, max_holdings=30))):
        add(f"S-buf-{name}", "구성", f"버퍼 {b.enter_rank_max}/{b.exit_rank_greater_than}/{b.max_holdings}", "회전율·집중도", construction(b=b))
    add("S-freq-2", "구성", "격주 리밸런싱", "회전율", construction(trade_every=2))

if BATCH == "b2":
    for th in (0.10, 0.125):
        add(f"M-dd-{th}", "G7", f"NAV 낙폭 > {th}면 E×0.5", "낙폭 제어 임계값", m8(exposure_fn=exp_dd(th)))
    for fac in (0.3, 0.7):
        add(f"M-dd-0.15-f{fac}", "G7", f"NAV 낙폭 > 0.15면 E×{fac}", "낙폭 제어 강도", m8(exposure_fn=exp_dd(0.15, fac)))
    add("M-dd+tr", "G7", "M-dd-0.15 + M-tr-b", "조합", m8(exposure_fn=compose(exp_trend("b"), exp_dd(0.15))))
    add("M-dd+tag", "G7", "M-dd-0.15 + M-tag-v20-0.8", "조합", m8(exposure_fn=exp_dd(0.15), block_fn=block_v20(0.8)))
    add("M-tr+tag", "G7", "M-tr-b + M-tag-v20-0.8", "조합", m8(exposure_fn=exp_trend("b"), block_fn=block_v20(0.8)))
    add("M-dd+tr+tag", "G7", "M-dd-0.15 + M-tr-b + M-tag-v20-0.8", "조합", m8(exposure_fn=compose(exp_trend("b"), exp_dd(0.15)), block_fn=block_v20(0.8)))
    sb1 = sig_with(rank_score=dmerge["b1_score"].to_numpy()); sb1["rank"] = rank_from_score(sb1, "rank_score").to_numpy()
    sb2 = sig_with(rank_score=dmerge["b2_score"].to_numpy()); sb2["rank"] = rank_from_score(sb2, "rank_score").to_numpy()
    ssm = sig_with(rank_score=smooth_score(4)); ssm["rank"] = rank_from_score(ssm, "rank_score").to_numpy()
    mix = sig_with(rank_score=(sig.groupby("date")["rank_score"].rank(pct=True).to_numpy()
                               + pd.Series(dmerge["b1_score"].to_numpy()).groupby(sig["date"].to_numpy()).rank(pct=True).to_numpy()) / 2)
    mix["rank"] = rank_from_score(mix, "rank_score").to_numpy()
    x80 = replace(buf, exit_rank_greater_than=80)
    add("S-rank-b2", "구성", "rank 출처 = b2_score (−rev_1m)", "베이스라인 순위", construction(sb2, key="b2"))
    add("S-rank-mix", "구성", "rank = (ML 백분위 + B1 백분위) / 2", "ML과 모멘텀 결합", construction(mix, key="mix"))
    add("S-b1+x80", "구성", "B1 순위 + 청산 80", "조합", construction(sb1, b=x80, key="b1"))
    add("S-b1+freq2", "구성", "B1 순위 + 격주", "조합", construction(sb1, key="b1", trade_every=2))
    add("S-sm4+x80", "구성", "4주 평균 순위 + 청산 80", "조합", construction(ssm, b=x80, key="sm4"))

for vid, track, change, why, _ in VARIANTS:
    ledger.plan(vid, track, change, why)

rows = []
for vid, track, change, why, runner in VARIANTS:
    t0 = time.perf_counter()
    try:
        full, off = runner()
        g7 = g7_check(full, off, kpi)
        row = {"id": vid, "track": track, "change": change, **{f"full_{k}": v for k, v in full.items()},
               **{f"off_{k}": v for k, v in off.items()}, "g7_mdd_red": g7["mdd_reduction_rel"],
               "g7_cagr_gap": full["cagr"] - off["cagr"], "g7_pass": g7["pass"], "sec": round(time.perf_counter() - t0, 1)}
        ledger.result(vid, years="2016-2020", **{k: v for k, v in row.items() if k not in ("id",)})
    except Exception as e:  # 실패도 원장에 남긴다
        row = {"id": vid, "track": track, "change": change, "error": repr(e)}
        ledger.result(vid, years="2016-2020", error=repr(e))
    rows.append(row)
    print(vid, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in
                ("full_cagr", "full_mdd", "full_ir", "full_annual_turnover", "full_mean_cash", "off_cagr", "off_mdd", "g7_mdd_red", "g7_cagr_gap", "g7_pass", "error", "sec")}, flush=True)

pd.DataFrame(rows).to_csv(HERE / f"{BATCH}_search.csv", index=False)
print("done", len(rows))
