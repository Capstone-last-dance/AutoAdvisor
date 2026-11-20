r"""walk-forward 실험실 (11/20 자율 모드, docs 33절): v9 폴드 정책(runs/t7i) 위에서 엔진 밖 장부로 여러 설계를 빠르게 잰다.

- 장부: t7_agent8.Book (현금 = CD 91일물, 비용 = 사전등록 수수료·슬리피지·매도세). 2016-01-08 첫 리밸런싱일 E 0.466107에서 시작.
- 폴드 Y = 2016~2025 (학습 2011~Y−3, 검증 Y−2~Y−1, 테스트 Y). 설계 값(λ, 오버레이 θ)은 폴드마다 그 폴드 검증 구간 점수
  CAGR ÷ max(MDD, 2%)로만 고른다. 테스트 해의 결과는 고르는 데 쓰지 않는다.
- 이 파일은 2026년 자료를 읽지 않는다 (바구니·특성·M8 상태는 2025-12-30까지).

명령 (저장소 루트): .\.venv\Scripts\python.exe -I -X utf8 explore\b09\run.py wf_lab.py check     # 엔진과 장부 비교
                    .\.venv\Scripts\python.exe -I -X utf8 explore\b09\run.py wf_lab.py overlays  # 실험 A
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import t7_agent9 as v9  # runs/t7i 경로로 바꿔 둔다
import t7_agent8 as base
from t7_agent8 import E_C53, FEATS, LEVELS, MDD_FLOOR, Book, normalize, obs_vec

HERE = v9.HERE
M8_PATH = v9.REPO / "reports" / "20261104_B_m8_states" / "figures" / "e_t_regime_mc.csv"  # 사전등록 M8 (B-04, 11/04)
PERIODS = {"all": (2016, 2025), "y16_20": (2016, 2020), "y21_25": (2021, 2025), "y24_25": (2024, 2025)}


class Lab:
    def __init__(self, load_models: bool = True) -> None:
        t0 = time.time()
        self.data = base.Data()
        f = self.data.f
        self.f = f
        self.X = f[FEATS].to_numpy(dtype=float)
        self.dec = f["k"].to_numpy(dtype=np.int64)
        self.year = f["year"].to_numpy()
        self.ma200 = f["ma200gap"].to_numpy(dtype=float)
        self.vol20 = f["vol20"].to_numpy(dtype=float)
        m8 = pd.read_csv(M8_PATH, parse_dates=["date"]).set_index("date")
        if m8.index.max() > pd.Timestamp("2025-12-31"):
            raise RuntimeError("M8 상태 파일에 2026년이 있다")
        self.m8 = m8["exposure"].reindex(f["date"]).to_numpy(dtype=float)
        self.regime = m8["regime_mc"].reindex(f["date"]).to_numpy(dtype=float)
        self.models = {}
        if load_models:
            for y in v9.FOLDS:
                for lam in v9.LAMBDAS:
                    ms, mu, sd = v9.load_fold(lam, y)
                    self.models[(y, lam)] = (ms, normalize(self.X, mu, sd))
        self.first = {y: int(np.flatnonzero(self.year == y)[0]) for y in range(2011, 2026)}
        print(f"실험실 준비 {time.time() - t0:.0f}초 (정책 {sum(len(v[0]) for v in self.models.values())}개)", flush=True)

    # ------------------------------------------------------------------
    def rows(self, y0: int, y1: int) -> np.ndarray:
        return np.flatnonzero((self.year >= y0) & (self.year <= y1))

    def run(self, rows: np.ndarray, choose, finish: int | None = None):
        """rows(리밸런싱일)마다 choose(k, book) → E, 다음 리밸런싱일까지 장부를 돌린다. 반환: (E 배열, 일별 잔고)."""
        px, rc = self.data.px, self.data.rc
        book, rec, es = Book(px, int(self.dec[rows[0]]), E_C53, rc=rc), [], []
        for n, k in enumerate(rows):
            e = float(np.clip(choose(k, book), 0.0, 1.0))
            es.append(e)
            if n + 1 < len(rows):
                j = int(self.dec[rows[n + 1]])
            else:
                j = finish if finish is not None else (int(self.dec[k + 1]) if k + 1 < len(self.dec) else len(px) - 1)
            if j > book.i:
                book.run(j, e, rec)
        nav = pd.Series([v for _, v in rec], index=self.data.days[[d for d, _ in rec]])
        return np.array(es), nav

    def agent(self, y: int, lam: float, seeds=None):
        ms, Z = self.models[(y, lam)]
        ms = ms if seeds is None else [ms[s] for s in seeds]

        def e_of(k, book):
            o = obs_vec(Z, k, book)
            return float(np.mean([LEVELS[int(m.predict(o, deterministic=True)[0])] for m in ms]))
        return e_of

    def overlay(self, theta, k: int, e: float, book) -> float:
        kind = theta[0]
        if kind == "none":
            return e
        if kind == "trend":  # 코스피 200이 200일선 아래면 비중 상한
            return min(e, theta[1]) if self.ma200[k] < 0 else e
        if kind == "m8":  # 사전등록 M8 노출(변동성 목표·급락 국면)을 곱한다
            return e * self.m8[k]
        if kind == "vol":  # 코스피 200 20일 변동성 목표
            return e * min(1.0, theta[1] / self.vol20[k])
        if kind == "dd":  # 자기 장부가 고점 대비 theta[1]보다 빠지면 비중 상한 theta[2]
            return min(e, theta[2]) if (1.0 - book.nav / book.peak) > theta[1] else e
        if kind == "band":  # 200일선 위면 하한 theta[1], 아래면 상한 theta[2] (그 안에서는 RL이 정한다)
            return max(e, theta[1]) if self.ma200[k] >= 0 else min(e, theta[2])
        raise ValueError(theta)

    # ------------------------------------------------------------------
    def stats(self, nav: pd.Series, y0: int, y1: int) -> tuple[float, float]:
        start = self.data.days[int(self.dec[self.first[y0]]) + 1]
        s = nav[(nav.index >= start) & (nav.index.year <= y1)]
        s = s / s.iloc[0]
        n = len(s) - 1
        return float(s.iloc[-1] ** (252 / n) - 1), float((1 - s / s.cummax()).max())

    def val_score(self, y: int, choose) -> dict:
        last = int(np.flatnonzero(self.data.days.year <= y - 1)[-1])
        es, nav = self.run(self.rows(y - 2, y - 1), choose, finish=last)
        n = len(nav) - 1
        cagr, mdd = float(nav.iloc[-1] ** (252 / n) - 1), float((1 - nav / nav.cummax()).max())
        return {"cagr": cagr, "mdd": mdd, "score": cagr / max(mdd, MDD_FLOOR), "mean_e": float(es.mean())}

    def test_path(self, pick: dict, seeds=None):
        """pick[y] = (λ, θ). 2016~2025 리밸런싱일마다 그 해 폴드 정책 + 오버레이 (장부는 이어진다)."""
        cache = {y: self.agent(y, lam, seeds) for y, (lam, _) in pick.items()}

        def choose(k, book):
            y = int(self.year[k])
            return self.overlay(pick[y][1], k, cache[y](k, book), book)
        return self.run(self.rows(2016, 2025), choose)

    def metrics(self, es, nav) -> dict:
        r = {"mean_e": float(np.mean(es))}
        for p, (y0, y1) in PERIODS.items():
            r[f"cagr_{p}"], r[f"mdd_{p}"] = self.stats(nav, y0, y1)
        r["yearly"] = {int(y): float(g.iloc[-1] / g.iloc[0] - 1) for y, g in nav.groupby(nav.index.year)}
        return r

    def select(self, grid: list, folds=None) -> dict:
        """폴드마다 grid의 (λ, θ) 중 검증 점수가 가장 높은 것."""
        out = {}
        for y in folds or v9.TEST_YEARS:
            sc = {(lam, th): self.val_score(y, self._comb(y, lam, th)) for lam, th in grid}
            best = max(sc, key=lambda c: sc[c]["score"])
            out[y] = {"pick": best, "score": sc[best]}
        return out

    def _comb(self, y, lam, th):
        a = self.agent(y, lam)
        return lambda k, book: self.overlay(th, k, a(k, book), book)


def fmt(r: dict) -> str:
    return " | ".join(f"{p} {r[f'cagr_{p}']:+.1%}·{r[f'mdd_{p}']:.1%}" for p in PERIODS) + f" | E {r['mean_e']:.2f}"


def check() -> None:
    lab = Lab()
    j = json.loads((HERE / "t7i_judgment.json").read_text(encoding="utf-8"))
    pick = {int(y): (float(v), ("none",)) for y, v in j["judgment"]["폴드별 고른 λ (그 폴드 검증 점수 최고)"].items() if int(y) <= 2025}
    es, nav = lab.test_path(pick)
    r = lab.metrics(es, nav)
    eng = j["results"]["wf_ens"]
    es_eng = np.array(json.loads((v9.EVAL / "wf_ens.json").read_text(encoding="utf-8"))["es"])
    print("v9 앙상블  장부:", fmt(r))
    print("v9 앙상블  엔진:", " | ".join(f"{p} {eng[f'cagr_{p}']:+.1%}·{eng[f'mdd_{p}']:.1%}" for p in PERIODS))
    print(f"  E 경로 엔진 평가 때와 최대 차이 {np.max(np.abs(es - es_eng)):.1e}")
    es, nav = lab.run(lab.rows(2016, 2025), lambda k, book: E_C53)
    c = j["results"]["c53"]
    print("C53 장부:", fmt(lab.metrics(es, nav)))
    print("C53 엔진:", " | ".join(f"{p} {c[f'cagr_{p}']:+.1%}·{c[f'mdd_{p}']:.1%}" for p in PERIODS))


GRID_A = [("none",), ("trend", 0.5), ("trend", 0.25), ("trend", 0.0), ("m8",), ("vol", 0.15), ("vol", 0.20),
          ("dd", 0.10, 0.25), ("dd", 0.15, 0.0)]


def overlays() -> None:
    """실험 A (docs 33절, 실행 전 고정): v9 폴드 정책 + 오버레이. 폴드마다 λ × θ (3 × 9 = 27가지)를 검증 점수로 고른다."""
    lab = Lab()
    grid = [(lam, th) for lam in v9.LAMBDAS for th in GRID_A]
    t0 = time.time()
    sel = lab.select(grid)
    print(f"폴드별 고르기 {time.time() - t0:.0f}초")
    for y, v in sel.items():
        print(f"  폴드 {y}: λ {v['pick'][0]:g}, θ {v['pick'][1]} (검증 {v['score']['cagr']:+.1%}·{v['score']['mdd']:.1%}, E {v['score']['mean_e']:.2f})")
    pick = {y: v["pick"] for y, v in sel.items()}
    res = {}
    es, nav = lab.test_path(pick)
    res["A_select"] = lab.metrics(es, nav)
    es_c, nav_c = lab.run(lab.rows(2016, 2025), lambda k, book: E_C53)
    res["c53"] = lab.metrics(es_c, nav_c)
    res["A_fixed"] = lab.metrics(*lab.run(lab.rows(2016, 2025), lambda k, book, m=float(np.mean(es)): m))
    seeds = []
    for s in v9.SEEDS:
        es_s, nav_s = lab.test_path(pick, seeds=[s])
        m_s = lab.metrics(es_s, nav_s)
        fx = lab.metrics(*lab.run(lab.rows(2016, 2025), lambda k, book, m=float(np.mean(es_s)): m))
        seeds.append((m_s, fx))
    for th in GRID_A:  # 참고: θ 하나를 모든 폴드에 (λ는 v9가 고른 값)
        j = json.loads((HERE / "t7i_judgment.json").read_text(encoding="utf-8"))
        lp = {int(y): float(v) for y, v in j["judgment"]["폴드별 고른 λ (그 폴드 검증 점수 최고)"].items() if int(y) <= 2025}
        res["ref_" + "_".join(map(str, th))] = lab.metrics(*lab.test_path({y: (lp[y], th) for y in v9.TEST_YEARS}))
    print("\n[엔진 밖 장부, 현금 = CD 91일물]")
    for n, r in res.items():
        print(f"  {n:22s} {fmt(r)}")
    for s, (m_s, fx) in zip(v9.SEEDS, seeds):
        print(f"  시드 {s}: {fmt(m_s)}  /  자기 고정 {fmt(fx)}")
    a, c, fx = res["A_select"], res["c53"], res["A_fixed"]
    g1 = a["cagr_all"] > c["cagr_all"] and a["mdd_all"] < c["mdd_all"]
    g2 = all(a[f"cagr_{p}"] > c[f"cagr_{p}"] and a[f"mdd_{p}"] < c[f"mdd_{p}"] for p in ("y16_20", "y21_25"))
    t1 = a["cagr_all"] > fx["cagr_all"] and a["mdd_all"] < fx["mdd_all"] and sum(m["cagr_all"] > f["cagr_all"] for m, f in seeds) >= 4
    print(f"\n목표 G1 (2016~2025 C53보다 CAGR↑·MDD↓): {'만족' if g1 else '미달'} / G2 (두 반기 모두): {'만족' if g2 else '미달'} / "
          f"타이밍 실력 (같은 평균 고정 대비, 시드 4/5): {'만족' if t1 else '미달'}")
    out = {"grid": [list(map(str, g)) for g in grid], "selection": {str(y): {"lam": v["pick"][0], "theta": list(v["pick"][1]), **v["score"]}
                                                                    for y, v in sel.items()},
           "results": res, "seeds": [{"agent": m, "fixed": f} for m, f in seeds], "G1": g1, "G2": g2, "timing": t1}
    (HERE / "wf_lab_A.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def judge(lab: Lab, pick: dict, name: str) -> dict:
    """pick으로 이어 붙인 앙상블·시드 판과 같은 평균 고정, C53을 장부로 재고 목표를 본다."""
    es, nav = lab.test_path(pick)
    a = lab.metrics(es, nav)
    c = lab.metrics(*lab.run(lab.rows(2016, 2025), lambda k, book: E_C53))
    fx = lab.metrics(*lab.run(lab.rows(2016, 2025), lambda k, book, m=float(np.mean(es)): m))
    seeds = []
    for s in v9.SEEDS:
        es_s, nav_s = lab.test_path(pick, seeds=[s])
        seeds.append((lab.metrics(es_s, nav_s), lab.metrics(*lab.run(lab.rows(2016, 2025), lambda k, book, m=float(np.mean(es_s)): m))))
    g1 = a["cagr_all"] > c["cagr_all"] and a["mdd_all"] < c["mdd_all"]
    g2 = all(a[f"cagr_{p}"] > c[f"cagr_{p}"] and a[f"mdd_{p}"] < c[f"mdd_{p}"] for p in ("y16_20", "y21_25"))
    nb = sum(m["cagr_all"] > f["cagr_all"] for m, f in seeds)
    t1 = a["cagr_all"] > fx["cagr_all"] and a["mdd_all"] < fx["mdd_all"] and nb >= 4
    sc = sum(m["cagr_all"] > c["cagr_all"] and m["mdd_all"] < c["mdd_all"] for m, _ in seeds)
    print(f"[{name}] 엔진 밖 장부, 현금 = CD 91일물")
    print(f"  앙상블        {fmt(a)}")
    print(f"  C53           {fmt(c)}")
    print(f"  같은 평균 고정 {fmt(fx)}")
    for s, (m, f) in zip(v9.SEEDS, seeds):
        print(f"  시드 {s}        {fmt(m)}")
    print("  연도 %: " + " ".join(f"{y % 100:02d}:{v * 100:+.0f}" for y, v in a["yearly"].items()))
    print(f"  G1 {'만족' if g1 else '미달'} / G2 {'만족' if g2 else '미달'} / 타이밍 실력 {'만족' if t1 else '미달'} "
          f"(자기 고정보다 CAGR 높은 시드 {nb}/5, C53보다 둘 다 나은 시드 {sc}/5)")
    return {"ensemble": a, "c53": c, "fixed": fx, "seeds": [{"agent": m, "fixed": f} for m, f in seeds],
            "G1": g1, "G2": g2, "timing": t1, "seeds_beat_fixed": nb, "seeds_beat_c53": sc}


def exp_b() -> None:
    """실험 B (docs 33절, 실행 전 고정): v9 폴드 정책(λ는 v9가 고른 값) + 추세 밴드 (위 하한 0.466107, 아래 상한 0.233054)."""
    lab = Lab()
    j = json.loads((HERE / "t7i_judgment.json").read_text(encoding="utf-8"))
    lp = {int(y): float(v) for y, v in j["judgment"]["폴드별 고른 λ (그 폴드 검증 점수 최고)"].items() if int(y) <= 2025}
    out = judge(lab, {y: (lp[y], ("band", E_C53, E_C53 / 2)) for y in v9.TEST_YEARS}, "실험 B: v9 + 추세 밴드")
    (HERE / "wf_lab_B.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    {"check": check, "overlays": overlays, "b": exp_b}[sys.argv[1]]()
