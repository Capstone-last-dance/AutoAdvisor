"""국면 성적표 + 여러 속도 설계 (11/19 사용자 결정 '둘 다 진행').

  python -I explore/b09/run.py safe_phases.py

질문: 같은 해 안에도 상승장과 하락장이 섞인다. 각 모델은 국면이 바뀔 때 제때 줄이고 제때 돌아왔나.

1) 국면 나누기 (사후 구분, 평가 전용. 매매 신호에 쓰지 않는다)
- 코스피 200 종가(2011년 첫 체결일 ~ 2025-12-30)에서 고점·저점을 번갈아 찾는다 (지그재그, θ = 10%).
  고점에서 10% 넘게 빠지면 그 고점을 확정하고, 저점에서 10% 넘게 오르면 그 저점을 확정한다.
- 고점 → 저점 = 하락 구간, 저점 → 고점 = 상승 구간. 처음과 마지막 구간은 확정되지 않은 부분 구간이라 합계에서 뺀다.

2) 모델 (모두 같은 15년 연속 실행, 현금 수익 0, ETF 비용)
- 지수 그대로, 고정 30%·50%·70%, 사전등록 M8, M-tr-b, M-tr-c, M-dd-0.15-f0.3, M-dd+tr, M-tr-0 (safe_dynamic 정의)
- 낙폭 한도 5%·15%·30% (A0 배수 4 · 고점 바닥 · 매주, safe_drawdown 정의)
- 새 설계 2개 (이 파일에서 처음 정의, 실행 전 고정 11/19. 국면 성적표 결과를 보기 전에 정했다)
  · D1 여러 속도 추세: 신호일 t에 지수의 지난 21·63·126·252거래일 수익률(약 1·3·6·12개월)이 0보다 크면 한 표씩.
    E_t = 표 수 / 4 (0%·25%·50%·75%·100%). 이력이 모자란 기간은 있는 기간만으로 평균한다.
  · D2 기본 + 조절: E_t = 0.5 + 0.2 × (2 × 표 비율 − 1) (30%·40%·50%·60%·70%). 기본 50%에서 ±20%p만 움직인다.
  · 같은 평균 비중의 고정 비중 (비교 기준): D1·D2의 15년 연속 실행 평균 주식 비중과 같은 w로 고정 비중을 돌린다.

3) 모델마다 재는 것 (확정된 구간만)
- 하락 구간 합산: 모든 하락 구간 수익률을 이어 곱한 값 (모델 vs 지수). '하락 포착률' = 모델 합산 손실 ÷ 지수 합산 손실.
- 상승 구간 합산: 같은 방식. '상승 포착률' = 모델 합산 이익 ÷ 지수 합산 이익.
- 타이밍 차 = 상승 구간 평균 주식 비중 − 하락 구간 평균 주식 비중 (일 가중). 0보다 크면 오를 때 더 들고 있었다.
- 줄인 날: 하락 구간 시작(고점) 뒤 비중이 고점 날보다 20%p 넘게 줄어든 첫날까지 거래일 수 (중앙값, 줄인 구간 수).
- 돌아온 날: 상승 구간 시작(저점) 뒤 비중이 저점 날보다 20%p 넘게 늘어난 첫날까지 거래일 수 (중앙값, 늘린 구간 수).

4) 새 설계 참고 판정 (실행 전 고정, 2011~2025는 이미 본 구간이라 '참고'다)
- D1·D2 각각: 15년 연속 실행에서 같은 평균 비중의 고정 비중보다 CAGR이 높고 최대 낙폭이 작으며, 타이밍 차 > 0.
- 17개 실행(safe_drawdown의 RUNS) 요약도 같이 낸다.
- 진짜 판정은 처음 보는 데이터로 한다: 설계를 커밋으로 고정한 뒤 forward test 기록, 그리고 PM이 허락하면 2026 홀드아웃 참고 행.
- 결과: safe_phases.csv (구간 × 모델), safe_phases_summary.csv (모델 요약), safe_phases_runs.csv (17개 실행), safe_phases.json, 원장.
"""
import json

from common import HERE, Ledger
import numpy as np
import pandas as pd
import safe_drawdown as sdd
import safe_dynamic as sdy

THETA = 0.10
LAGS = (21, 63, 126, 252)
STEP = 0.20  # 줄인 날·돌아온 날의 비중 변화 기준
DAYS, P = sdd.DAYS, sdd.P
_px = P.to_numpy(dtype=float)
VOTE = {}
for _i, _d in enumerate(DAYS):
    _v = [_px[_i] / _px[_i - k] - 1.0 > 0.0 for k in LAGS if _i - k >= 0]
    VOTE[_d] = (float(np.mean(_v)) if _v else 0.5, len(_v))


def d1(t, st):
    return VOTE[t][0]


def d2(t, st):
    return 0.5 + 0.2 * (2.0 * VOTE[t][0] - 1.0)


def fixed(w):
    return lambda t, st: w


RULES = ["사전등록 M8", "M-tr-b", "M-tr-c", "M-dd-0.15-f0.3", "M-dd+tr", "M-tr-0"]


def run_models(dates, end_day):
    """모델 이름 → simulate 결과 (일별 _nav, _w 포함)."""
    out = {"지수 그대로": sdy.simulate("지수 그대로", dates, end_day, series=True)}
    for w in (0.3, 0.5, 0.7):
        out[f"고정 {w:.0%}"] = sdy.simulate(f"고정 {w:.0%}", dates, end_day, series=True)
    for r in RULES:
        out[r] = sdy.simulate(r, dates, end_day, series=True)
    for x in (0.05, 0.15, 0.30):
        out[f"낙폭 한도 {x:.0%}"] = sdd.simulate("A0 고정 배수", "고점", "매주", dates, end_day, x=x, series=True)
    out["D1 여러 속도 추세"] = sdy.simulate("D1", dates, end_day, fn=d1, series=True)
    out["D2 기본 + 조절"] = sdy.simulate("D2", dates, end_day, fn=d2, series=True)
    return out


def zigzag(s, theta):
    """θ 이상 움직인 고점·저점을 번갈아 찾는다 (사후 구분, 평가 전용). [(날짜, 'peak'|'trough'), ...]"""
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


def phases(s, theta):
    piv = zigzag(s, theta)
    bounds = [(s.index[0], None)] + piv + [(s.index[-1], None)]
    segs = []
    for k in range(len(bounds) - 1):
        (a, _), (b, kind_b) = bounds[k], bounds[k + 1]
        if a == b:
            continue
        partial = k == 0 or k == len(bounds) - 2
        if kind_b == "peak":
            kind = "상승"
        elif kind_b == "trough":
            kind = "하락"
        else:  # 마지막 부분 구간: 직전 확정점의 반대 방향
            kind = "하락" if bounds[k][1] == "peak" else "상승"
        segs.append({"start": a, "end": b, "kind": kind, "partial": partial})
    return segs


def first_cross(w, a, b, base, up):
    seg = w[(w.index > a) & (w.index <= b)]
    hit = seg[seg >= base + STEP - 1e-12] if up else seg[seg <= base - STEP + 1e-12]
    if hit.empty:
        return None
    return int(((w.index > a) & (w.index <= hit.index[0])).sum())


def main():
    ledger = Ledger(HERE / "ledger.jsonl")
    ledger.plan("safeph|D1 여러 속도 추세", "국면 대응 설계", "1·3·6·12개월 지수 수익률 부호 투표, 비중 = 표 수 / 4",
                "사용자 질문(11/19): 같은 해 안의 상승·하락장 대응. 국면 성적표 결과를 보기 전에 고정")
    ledger.plan("safeph|D2 기본 + 조절", "국면 대응 설계", "기본 50% ± 20%p (D1과 같은 투표)",
                "사용자 질문(11/19): 같은 해 안의 상승·하락장 대응. 국면 성적표 결과를 보기 전에 고정")
    dates, end_day = sdd.window(2011, 2025)
    first = run_models(dates, end_day)
    w_eq = {}
    for d in ("D1 여러 속도 추세", "D2 기본 + 조절"):
        r = first[d]
        rep = r["_w"][r["_w"].index >= r["_nav"].index[1]]
        w_eq[d] = round(float(rep.mean()), 4)
    eq_names = {d: f"고정 {w:.1%} (={d.split()[0]} 평균)" for d, w in w_eq.items()}
    extra = [(eq_names[d], w) for d, w in w_eq.items()]
    models = dict(first)
    for name, w in extra:  # 같은 평균 비중의 고정 비중
        models[name] = sdy.simulate(name, dates, end_day, fn=fixed(w), series=True)
    nav0 = models["지수 그대로"]["_nav"]
    start = nav0.index[1]  # 첫 체결일
    s = P[(P.index >= start) & (P.index <= end_day)]
    segs = phases(s, THETA)

    rows = []
    for seg in segs:
        a, b = seg["start"], seg["end"]
        base = {"start": a.date().isoformat(), "end": b.date().isoformat(), "kind": seg["kind"], "partial": seg["partial"],
                "days": int(((s.index > a) & (s.index <= b)).sum()), "index_ret": float(s[b] / s[a] - 1.0)}
        for name, r in models.items():
            nav, w = r["_nav"], r["_w"]
            win = w[(w.index > a) & (w.index <= b)]
            rows.append({**base, "model": name, "ret": float(nav[b] / nav[a] - 1.0), "w_start": float(w[a]),
                         "w_mean": float(win.mean()), "w_end": float(w[b]),
                         "lag": first_cross(w, a, b, float(w[a]), up=seg["kind"] == "상승")})
    ph = pd.DataFrame(rows)
    ph.to_csv(HERE / "safe_phases.csv", index=False)

    full = ph[~ph.partial]
    summ = []
    for name, r in models.items():
        g = full[full.model == name]
        dn, upp = g[g.kind == "하락"], g[g.kind == "상승"]
        dn_m, dn_i = float(np.prod(1 + dn.ret) - 1), float(np.prod(1 + dn.index_ret) - 1)
        up_m, up_i = float(np.prod(1 + upp.ret) - 1), float(np.prod(1 + upp.index_ret) - 1)
        wd = float(np.average(dn.w_mean, weights=dn.days)) if len(dn) else np.nan
        wu = float(np.average(upp.w_mean, weights=upp.days)) if len(upp) else np.nan
        summ.append({
            "model": name, "cagr15": r["cagr"], "mdd15": r["mdd"], "final15": r["final_nav"],
            "mean_w": float(r["_w"][r["_w"].index >= start].mean()),
            "down_total": dn_m, "down_capture": dn_m / dn_i, "up_total": up_m, "up_capture": up_m / up_i,
            "w_up": wu, "w_down": wd, "timing_spread": wu - wd,
            "cut_lag_median": dn.lag.dropna().median() if dn.lag.notna().any() else None,
            "cut_count": f"{int(dn.lag.notna().sum())}/{len(dn)}",
            "back_lag_median": upp.lag.dropna().median() if upp.lag.notna().any() else None,
            "back_count": f"{int(upp.lag.notna().sum())}/{len(upp)}",
        })
    sm = pd.DataFrame(summ)
    sm.to_csv(HERE / "safe_phases_summary.csv", index=False)

    runs, run_rows = {}, []
    names = ["D1 여러 속도 추세", "D2 기본 + 조절", *eq_names.values(), "고정 50%", "지수 그대로"]
    for wname, (y0, y1) in sdd.RUNS.items():
        dts, ed = sdd.window(y0, y1)
        for name in names:
            if name == "D1 여러 속도 추세":
                res = sdy.simulate("D1", dts, ed, fn=d1)
            elif name == "D2 기본 + 조절":
                res = sdy.simulate("D2", dts, ed, fn=d2)
            elif name in eq_names.values():
                res = sdy.simulate(name, dts, ed, fn=fixed(dict(extra)[name]))
            else:
                res = sdy.simulate(name, dts, ed)
            runs[(wname, name)] = res
            run_rows.append({"window": wname, "model": name, **{k: v for k, v in res.items() if k != "yearly"},
                             **{f"y{y}": v for y, v in res["yearly"].items()}})
    rd = pd.DataFrame(run_rows)
    rd.to_csv(HERE / "safe_phases_runs.csv", index=False)
    (HERE / "safe_phases.json").write_text(json.dumps({
        "theta": THETA, "phases": [{k: (v.date().isoformat() if hasattr(v, "date") else v) for k, v in sg.items()} for sg in segs],
        "w_equal": w_eq, "summary": summ,
        "runs": {f"{w} | {m}": {k: v for k, v in r.items()} for (w, m), r in runs.items()},
    }, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(f"\n[국면] 지그재그 θ = {THETA:.0%}: 구간 {len(segs)}개 (확정 {int((~pd.Series([g['partial'] for g in segs])).sum())}개)")
    seg_t = ph[ph.model == "지수 그대로"][["start", "end", "kind", "partial", "days", "index_ret"]].copy()
    seg_t["index_ret"] = (seg_t.index_ret * 100).round(1)
    print(seg_t.to_string(index=False))

    show = sm.copy()
    for c in ("cagr15", "mdd15", "mean_w", "down_total", "down_capture", "up_total", "up_capture", "w_up", "w_down", "timing_spread"):
        show[c] = (show[c] * 100).round(1)
    show["final15"] = show["final15"].round(3)
    print("\n[국면 성적표] % 단위. down_capture = 하락 구간 합산 손실 ÷ 지수, up_capture = 상승 구간 합산 이익 ÷ 지수")
    print(show.to_string(index=False))

    key = ["지수 그대로", "고정 50%", "사전등록 M8", "M-tr-c", "낙폭 한도 15%", "D1 여러 속도 추세", "D2 기본 + 조절"]
    pv = ph[ph.model.isin(key)].pivot_table(index=["start", "end", "kind"], columns="model", values="ret", sort=False)[key]
    print("\n[구간별 수익률 %] 주요 모델")
    print((pv * 100).round(1).to_string())

    print("\n[17개 실행 요약] D1·D2와 같은 평균 비중의 고정 비중")
    cols = ["window", "model", "cagr", "mdd", "final_nav", "mean_exposure"]
    t = rd[rd.window.isin(list(sdd.MAIN))][cols].copy()
    for c in ("cagr", "mdd", "mean_exposure"):
        t[c] = (t[c] * 100).round(2)
    t["final_nav"] = t["final_nav"].round(3)
    print(t.to_string(index=False))
    for name in names:
        three = [runs[(w, name)] for w in sdd.STARTS]
        print(f"  {name}: 3년 실행 원금 이상 {sum(r['final_nav'] >= 1 for r in three)}/{len(three)}, "
              f"가장 나쁜 3년 끝 잔고 {min(r['final_nav'] for r in three):.3f}")

    print("\n[새 설계 참고 판정] (2011~2025는 이미 본 구간 → 참고. 진짜 판정은 forward test·홀드아웃)")
    for d in ("D1 여러 속도 추세", "D2 기본 + 조절"):
        me, eq = sm[sm.model == d].iloc[0], sm[sm.model == eq_names[d]].iloc[0]
        checks = {"CAGR > 같은 평균 비중 고정": bool(me.cagr15 > eq.cagr15), "최대 낙폭 < 같은 평균 비중 고정": bool(me.mdd15 < eq.mdd15),
                  "타이밍 차 > 0": bool(me.timing_spread > 0)}
        print(f"  {d} (평균 비중 {w_eq[d]:.1%}): " + ", ".join(f"{k} {'만족' if v else '미달'}" for k, v in checks.items()))
        ledger.result(f"safeph|{d}", w_equal=w_eq[d], checks=checks, reference_pass=all(checks.values()),
                      **{k: v for k, v in me.items() if k != "model"})
    print("done")


if __name__ == "__main__":
    main()
