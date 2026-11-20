r"""코스피 종합지수 기준 비교 (11/19 사용자 결정: 명세 4.5절의 코스피 200 대신 코스피 종합지수로 비교한다).

결과 읽기다. 엔진을 다시 돌리지 않고, 이미 저장된 일별 NAV를 지수와 비교한다 (몇 초).
  .\.venv\Scripts\python.exe -I explore\b09\run.py bench_kospi.py

- 지수: files/A_files/A_kospi_plain_20261119 (코스피 종합, 순수지수, 배당 제외, A가 기존 원문에서 추출한 참고용 파일),
  data/processed/index.parquet 코스피 200 (공식). 두 지수의 거래일은 같다.
- CAGR·연 초과수익: 공식 summary와 같은 정의 (시작 NAV 1, 일간 수익률 수 ÷ 252로 연율화). MDD: 일별 NAV.
- IR: 체결일(신호일 다음 거래일) 사이 주간 초과수익 a_k의 mean / std(ddof=1) × √52. 마지막 신호일 뒤 구간은 뺀다 (공식과 같다).
  저장된 NAV는 체결 후 값이라 공식(체결 직전 NAV)과 주 첫 체결 비용이 한 주씩 밀린다. 그래서 코스피 200 기준 IR을
  저장된 공식 IR과 대조해 차이를 같이 적는다.
- 보조 기준: U_t 동일가중 (명세 4.5절 EW, 비용 없음, 매 신호일 다음 거래일 종가에 같은 비중으로 맞춘다).
- 결과: bench_kospi.csv, bench_kospi.json.
"""
from __future__ import annotations

import json

from common import HERE, REPO
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KOSPI_PATH = REPO / "files" / "A_files" / "A_kospi_plain_20261119" / "index_kospi_plain_20261119.parquet"
OFFICIAL = REPO / "reports" / "20261118_B_buffer_wf"
END = pd.Timestamp("2025-12-30")


def nav_parquet(p):
    f = pd.read_parquet(p)
    return pd.Series(f["nav"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(f["date"])).as_unit("ns"))


def runs() -> list[tuple[str, str, pd.Series, float | None]]:
    """(묶음, 이름, 일별 NAV, 저장된 공식 IR(코스피 200))."""
    out = []
    d = pd.read_csv(OFFICIAL / "daily_nav.csv", parse_dates=["date"]).set_index("date")
    d.index = pd.DatetimeIndex(d.index).as_unit("ns")
    g5 = pd.read_csv(OFFICIAL / "g5_table.csv")
    m = json.loads((OFFICIAL / "metrics.json").read_text(encoding="utf-8"))
    out.append(("공식", "B + 사전등록 M8 + M9 (공식)", d["buffer_full"], float(m["g5"]["buffer"]["ir"])))
    out.append(("공식", "B 100% (노출 조절 끔) = T0", d["buffer_no_exposure"], float(m["g4_values"]["buffer_no_exposure"]["ir"])))
    for s in range(5):
        ir = float(g5[(g5.portfolio == "rl") & (g5.seed == str(s))]["ir"].iloc[0]) if "seed" in g5 else None
        out.append(("공식 PPO", f"공식 PPO 시드 {s}", d[f"rl_s{s}"], ir))
    for n, label in [("T3", "T3 4주 평균 순위 + 청산 80"), ("T6", "T6 8주 평균 순위 + 청산 80"),
                     ("T7", "T7 4주 평균 순위 + 청산 80 + 격주"), ("T3+agent", "T3 + 바구니 에이전트 비중")]:
        j = json.loads((HERE / "runs" / "tov" / f"{n}.json").read_text(encoding="utf-8"))
        out.append(("교체 줄이기", label, nav_parquet(HERE / "runs" / "tov" / f"nav_{n}.parquet"), j["ir"]))
    for n, label in [("T7_S4", "T7 + 이탈 장치 S4"), ("T7_S4-m50", "T7 + 이탈 장치 S4-m50"), ("T7_S2", "T7 + 손절만 S2")]:
        j = json.loads((HERE / "runs" / "exit" / f"{n}.json").read_text(encoding="utf-8"))
        out.append(("이탈 장치", label, nav_parquet(HERE / "runs" / "exit" / f"nav_{n}.parquet"), j["ir"]))
    for n, label in [("bkt_ens", "B + 바구니 에이전트 비중 (15절)")]:
        j = json.loads((HERE / "runs" / "bt" / "eval" / f"{n}.json").read_text(encoding="utf-8"))
        out.append(("바구니 타이밍", label, nav_parquet(HERE / "runs" / "bt" / "eval" / f"nav_{n}.parquet"), j["ir"]))
    return out


def ew_nav(days: pd.DatetimeIndex, sig_dates: pd.DatetimeIndex) -> pd.Series:
    u = pd.read_parquet(REPO / "data" / "processed" / "universe.parquet")
    u = u[u["in_universe"].astype(bool)]
    u = u.assign(date=pd.to_datetime(u["date"]), ticker=u["ticker"].astype(str))
    mem = {pd.Timestamp(k): g["ticker"].tolist() for k, g in u[u["date"].isin(sig_dates)].groupby("date")}
    tick = sorted({t for v in mem.values() for t in v})
    px = pq.read_table(REPO / "data" / "processed" / "prices.parquet", columns=["date", "ticker", "close"],
                       filters=[("ticker", "in", tick), ("date", ">=", pd.Timestamp("2015-12-01")),
                                ("date", "<=", END)]).to_pandas()
    px = px.assign(date=pd.to_datetime(px["date"]), ticker=px["ticker"].astype(str))
    C = px.pivot(index="date", columns="ticker", values="close").reindex(days).astype(float)
    C = C.where(C > 0).ffill()
    fills = [days[days.searchsorted(d, side="right")] for d in sig_dates]
    nav, out = 1.0, {}
    for k, d in enumerate(sig_dates):
        a, b = fills[k], (fills[k + 1] if k + 1 < len(sig_dates) else days[days <= END][-1])
        seg = C.loc[a:b, [t for t in mem[d] if t in C.columns]]
        rel = seg.div(seg.iloc[0]).mean(axis=1, skipna=True)
        for day, v in rel.items():
            out[day] = nav * v
        nav = out[b]
    return pd.Series(out).sort_index()


def stats(nav: pd.Series, bms: dict, fills: pd.DatetimeIndex) -> dict:
    nav = nav[(nav.index >= fills[0]) & (nav.index <= END)]
    n = len(nav) - 1
    v = np.r_[1.0, nav.to_numpy()[1:]]
    cagr = v[-1] ** (252 / n) - 1
    out = {"cagr": float(cagr), "mdd": float((1 - nav / nav.cummax()).max()), "final_nav": float(nav.iloc[-1])}
    f = [d for d in fills[:-1] if d in nav.index] + ([fills[-1]] if fills[-1] in nav.index else [])
    pa = nav.reindex(f).to_numpy()
    for k, bm in bms.items():
        b = bm.reindex(nav.index).to_numpy()
        out[f"excess_{k}"] = float(cagr - ((b[-1] / b[0]) ** (252 / n) - 1))
        bb = bm.reindex(f).to_numpy()
        a = np.diff(pa) / pa[:-1] - np.diff(bb) / bb[:-1]
        out[f"ir_{k}"] = float(a.mean() / a.std(ddof=1) * np.sqrt(52))
    return out


def yearly(s: pd.Series) -> dict:
    l = s.groupby(s.index.year).last()
    first = s.iloc[0]
    return {int(y): float(v) for y, v in (l / l.shift(1).fillna(first) - 1).items()}


def main() -> None:
    k = pd.read_parquet(KOSPI_PATH)
    kospi = pd.Series(k["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(k["date"])).as_unit("ns"))
    i = pd.read_parquet(REPO / "data" / "processed" / "index.parquet")
    i = i[i["index_code"] == "KOSPI200"]
    k200 = pd.Series(i["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(i["date"])).as_unit("ns"))
    days = k200.index[k200.index <= END]
    if not days.isin(kospi.index).all():
        raise RuntimeError("코스피 종합지수에 없는 거래일이 있다")
    dates = pd.DatetimeIndex(pd.to_datetime(json.loads((HERE / "runs" / "bt" / "eval" / "bkt_ens.json")
                                                       .read_text(encoding="utf-8"))["dates"])).as_unit("ns")
    fills = pd.DatetimeIndex([days[days.searchsorted(d, side="right")] for d in dates])
    bms = {"kospi": kospi, "k200": k200}
    ew = ew_nav(days, dates)
    rows = []
    for group, name, nav, ir_official in runs() + [("기준", "U_t 동일가중 (EW, 비용 없음)", ew, None)]:
        s = stats(nav, bms, fills)
        rows.append({"group": group, "name": name, **s, "ir_k200_official": ir_official,
                     "ir_check": None if ir_official is None else s["ir_k200"] - ir_official,
                     **{f"y{y}": r for y, r in yearly(nav[(nav.index >= fills[0]) & (nav.index <= END)]).items()}})
    rd = pd.DataFrame(rows)
    rl = rd[rd.group == "공식 PPO"]
    mean_row = {"group": "공식 PPO", "name": "공식 PPO 5시드 평균",
                **{c: float(rl[c].mean()) for c in rd.columns if c not in ("group", "name") and rl[c].notna().any()}}
    rd = pd.concat([rd[rd.group != "공식 PPO"], pd.DataFrame([mean_row])], ignore_index=True)
    rd.to_csv(HERE / "bench_kospi.csv", index=False)
    idx = {}
    for kname, bm in (("코스피 종합", kospi), ("코스피 200", k200)):
        b = bm[(bm.index >= fills[0]) & (bm.index <= END)]
        n = len(b) - 1
        idx[kname] = {"cagr": float((b.iloc[-1] / b.iloc[0]) ** (252 / n) - 1), "mdd": float((1 - b / b.cummax()).max()),
                      "yearly": yearly(b)}
    (HERE / "bench_kospi.json").write_text(json.dumps({"period": [str(fills[0].date()), str(END.date())], "index": idx,
                                                       "ir_check_max_abs": float(rd["ir_check"].abs().max())},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print(f"[{fills[0].date()} ~ {END.date()}] 코스피 종합 CAGR {idx['코스피 종합']['cagr']:+.2%}, MDD {idx['코스피 종합']['mdd']:.1%} / "
          f"코스피 200 CAGR {idx['코스피 200']['cagr']:+.2%}, MDD {idx['코스피 200']['mdd']:.1%}")
    show = rd[["name", "cagr", "mdd", "excess_kospi", "ir_kospi", "excess_k200", "ir_k200", "ir_k200_official", "final_nav"]].copy()
    for c in ("cagr", "mdd", "excess_kospi", "excess_k200"):
        show[c] = (show[c] * 100).round(1)
    for c in ("ir_kospi", "ir_k200", "ir_k200_official", "final_nav"):
        show[c] = pd.to_numeric(show[c], errors="coerce").round(3)
    print(show.to_string(index=False))
    print(f"IR 재현 확인 (코스피 200 기준, 저장된 공식 값과의 차이 최대): {rd['ir_check'].abs().max():.4f}")
    yc = [c for c in rd.columns if c.startswith("y") and c[1:].isdigit()]
    yt = pd.DataFrame({**{kname: {f"y{y}": r for y, r in v["yearly"].items()} for kname, v in idx.items()}}).T
    print("\n[연도별 수익률 %]")
    print(pd.concat([yt[yc] * 100, rd.set_index("name").loc[["U_t 동일가중 (EW, 비용 없음)", "B 100% (노출 조절 끔) = T0",
                                                              "T7 4주 평균 순위 + 청산 80 + 격주", "T7 + 이탈 장치 S4-m50"], yc] * 100])
          .round(1).to_string())
    print("done")


if __name__ == "__main__":
    main()
