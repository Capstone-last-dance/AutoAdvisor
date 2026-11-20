r"""T7-80-RL 고정 (docs 35절, 11/20 사용자 결정: "수치가 제일 좋았던 RL 모델로 판정 받아보자").

탐색 결과(v12 = t7_agent12.py, runs/t7l)에서 λ 0.25 판을 main 보고 폴더 reports/20261120_B_t7rl로 옮긴다.
- models/y{2016..2026}/seed_{0..4}/best.zip (+ done.json, eval_log.csv), models/y{Y}/stats.json (폴드 표준화 값)
- inputs/basket_px.csv (2011-01-07 ~ 2025-12-30 바구니 가격), inputs/fred_us.csv (FRED 4계열 2008-01-01 ~ 2025-12-31),
  inputs/cd91_monthly.csv (~2025-12)
- reference/: 탐색에서 잰 값 (main 재현 대조용) — E = 1·C53 일별 NAV, walk-forward 특성·E 경로·NAV, 폴드 2026 검증 구간 E 경로
- manifest.json: 파일마다 SHA-256
2026년 자료는 읽지 않는다 (FRED·CD 91일물은 2025-12-31까지 자른다).

실행 (저장소 루트): .\.venv\Scripts\python.exe -I -X utf8 explore\b09\run.py t7rl_freeze.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import t7_agent12 as m  # 경로·특성·행동 11단계를 v12 것으로 바꾼다
import t7_agent8 as base

LAM = 0.25
OUT = m.REPO / "reports" / "20261120_B_t7rl"
FRED_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else None  # FRED 내려받기 폴더 (VIXCLS 등 CSV)
END = pd.Timestamp("2025-12-31")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    if FRED_DIR is None or not (FRED_DIR / "VIXCLS.csv").exists():
        raise SystemExit("FRED 내려받기 폴더를 인자로 준다 (VIXCLS.csv·BAA10Y.csv·DEXKOUS.csv·NASDAQCOM.csv)")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)
    data = base.Data()

    # 1) 정책 (λ 0.25, 폴드 2016~2026, 시드 0~4)
    for y in m.FOLDS:
        st0 = None
        for s in m.SEEDS:
            src = m.run_dir(LAM, y, s)
            st = json.loads((src / "stats.json").read_text(encoding="utf-8"))
            if st0 is None:
                st0 = st
            elif st["mu"] != st0["mu"] or st["sd"] != st0["sd"] or st["cols"] != st0["cols"]:
                raise RuntimeError(f"폴드 {y}: 시드마다 표준화 값이 다르다")
            dst = OUT / "models" / f"y{y}" / f"seed_{s}"
            dst.mkdir(parents=True)
            for n in ("best.zip", "done.json", "eval_log.csv"):
                shutil.copy2(src / n, dst / n)
        (OUT / "models" / f"y{y}" / "stats.json").write_text(json.dumps(st0, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"정책: 폴드 {len(m.FOLDS)}개 × 시드 {len(m.SEEDS)}개", flush=True)

    # 2) 고정 입력
    (OUT / "inputs").mkdir()
    pd.DataFrame({"date": data.days.strftime("%Y-%m-%d"), "px": data.px}).to_csv(OUT / "inputs" / "basket_px.csv", index=False)
    us = {}
    for k in ("VIXCLS", "BAA10Y", "DEXKOUS", "NASDAQCOM"):
        f = pd.read_csv(FRED_DIR / f"{k}.csv")
        s = pd.Series(pd.to_numeric(f[f.columns[1]], errors="coerce").to_numpy(), index=pd.to_datetime(f[f.columns[0]]))
        us[k] = s[(s.index >= "2008-01-01") & (s.index <= END)].dropna()
    u = pd.DataFrame(us).sort_index()
    u.index.name = "date"
    u.reset_index().assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_csv(OUT / "inputs" / "fred_us.csv", index=False)
    cd = pd.read_csv(m.HERE / "cd91_monthly.csv")
    cd[cd["month"] <= "2025-12"].to_csv(OUT / "inputs" / "cd91_monthly.csv", index=False)
    print(f"입력: 바구니 가격 {len(data.px)}일 ({data.days[0].date()}~{data.days[-1].date()}), FRED {len(u)}행 "
          f"(~{u.index.max().date()}), CD 91일물 ~{cd[cd['month'] <= '2025-12']['month'].max()}", flush=True)

    # 3) 대조값 (탐색에서 잰 값)
    ref = OUT / "reference"
    ref.mkdir()
    for name, src in (("e1_nav", base.SAMECASH / "nav_e1.parquet"), ("c53_nav", base.SAMECASH / "nav_cash.parquet")):
        f = pd.read_parquet(src)
        pd.DataFrame({"date": pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d"), "nav": f["nav"].to_numpy(float)}).to_csv(
            ref / f"{name}.csv", index=False)
    lv = data.lv
    wf = json.loads((m.EVAL / "wf_m0.25.json").read_text(encoding="utf-8"))
    if len(wf["es"]) != len(lv):
        raise RuntimeError("walk-forward E 경로 길이가 결정일 수와 다르다")
    pd.DataFrame({"date": lv["date"].dt.strftime("%Y-%m-%d"), **{c: lv[c].to_numpy(float) for c in m.FEATS}}).to_csv(
        ref / "wf_features.csv", index=False)
    pd.DataFrame({"date": lv["date"].dt.strftime("%Y-%m-%d"), "exposure": wf["es"]}).to_csv(ref / "wf_decisions.csv", index=False)
    nv = pd.read_parquet(m.EVAL / "nav_wf_m0.25.parquet")
    nv["date"] = pd.to_datetime(nv["date"]).dt.strftime("%Y-%m-%d")
    nv.to_csv(ref / "wf_nav.csv", index=False)
    # 폴드 2026 정책의 검증 구간(2024~2025) 경로 — 탐색 val_metrics와 같은 함수 (검증 자료만)
    models, mu, sd = m.load_fold(LAM, m.FINAL_FOLD)
    rec = []
    last = int(np.flatnonzero(data.days.year <= 2025)[-1])
    es, _ = base.path_from(data, models, mu, sd, 2024, 2025, finish=last, rec=rec)
    rows = data.f[(data.f["year"] >= 2024) & (data.f["year"] <= 2025)]
    pd.DataFrame({"date": rows["date"].dt.strftime("%Y-%m-%d"), "exposure": es}).to_csv(ref / "val2026_decisions.csv", index=False)
    v = m.val_metrics(data, models, mu, sd, m.FINAL_FOLD)
    print(f"대조값: walk-forward 결정 {len(lv)}개, 폴드 2026 검증 {len(es)}개 (검증 2024~2025 {v['cagr']:+.2%}·{v['mdd']:.1%}, "
          f"평균 E {v['mean_e']:.2f})", flush=True)

    # 4) 지문
    files = {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    man = {"model": "T7-80-RL", "run_id": OUT.name, "frozen": "2026-11-20",
           "source": {"branch": "b09-explore", "trainer": "explore/b09/t7_agent12.py (v12, runs/t7l)", "lambda": LAM,
                      "note": "docs/b09_explore.md 34·35절"},
           "val2026": v, "files": files}
    (OUT / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    size = sum((OUT / k).stat().st_size for k in files)
    print(f"고정: 파일 {len(files)}개, {size / 1e6:.1f}MB → {OUT}", flush=True)


if __name__ == "__main__":
    main()
