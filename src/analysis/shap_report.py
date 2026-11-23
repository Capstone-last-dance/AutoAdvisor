"""M12(C-06): A의 M3a/M3b 테스트 폴드 모델을 해석한다.

각 테스트 연도에서 시드 42로 최대 5,000행을 뽑는다. 게이트는 양성(급락)
출력의 SHAP 값, 순위 모델은 회귀 출력의 SHAP 값을 사용한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.models.gate import load_booster

MODEL_TYPES = ("gate", "ranker")
SAMPLE_SIZE = 5_000
SEED = 42


def sample_test_year(features: pd.DataFrame, year: int, *, size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """행 순서에 영향받지 않도록 키 정렬 후 연도별 결정적 표본을 뽑는다."""
    if not {"date", "ticker"} <= set(features):
        raise ValueError("features에 date, ticker가 필요하다")
    if size < 1:
        raise ValueError("표본 크기는 양수여야 한다")
    rows = features.loc[pd.to_datetime(features["date"]).dt.year.eq(year)].copy()
    if rows.empty:
        raise ValueError(f"{year}년 시험 피처가 없다")
    if rows[["date", "ticker"]].isna().any().any() or rows.duplicated(["date", "ticker"]).any():
        raise ValueError(f"{year}년 피처 키가 결측이거나 중복됐다")
    rows = rows.sort_values(["date", "ticker"]).reset_index(drop=True)
    chosen = np.random.default_rng(SEED).choice(len(rows), size=min(size, len(rows)), replace=False)
    return rows.iloc[np.sort(chosen)].reset_index(drop=True)


def _values(explainer, frame: pd.DataFrame, *, binary: bool) -> np.ndarray:
    """SHAP 버전에 따른 이진 분류 출력 표현을 양성 클래스 2차원 배열로 통일한다."""
    values = explainer.shap_values(frame)
    if isinstance(values, list):
        values = values[-1] if binary else values[0]
    values = np.asarray(values, dtype=float)
    if binary and values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, :, 1]
    if values.shape != frame.shape or not np.isfinite(values).all():
        raise ValueError(f"SHAP 배열 형식/유한값 오류: {values.shape}, 예상 {frame.shape}")
    return values


def _model_path(models: Path, year: int, kind: str, h: int) -> Path:
    name = f"gate_{year}.txt" if kind == "gate" else f"ranker_{year}_h{h}.txt"
    return models / f"fold_{year}" / kind / name


def _write_dependence(frame: pd.DataFrame, values: np.ndarray, feature: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    column = frame.columns.get_loc(feature)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(frame.iloc[:, column], values[:, column], s=5, alpha=0.25, rasterized=True)
    ax.set(xlabel=feature, ylabel="SHAP value", title=f"Dependence: {feature}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(features_path: Path, models: Path, output: Path, *, years=range(2016, 2026), h: int = 20) -> dict:
    """폴드별 모델·피처로 C-06 표/그림과 R-10 실행 기록을 생성한다."""
    features = pd.read_parquet(features_path)
    years = tuple(int(y) for y in years)
    if not years or len(set(years)) != len(years) or any(y < 2016 or y > 2025 for y in years) or h < 1:
        raise ValueError("시험 연도는 중복 없는 2016~2025 목록, h는 양수여야 한다")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"기존 결과를 덮어쓰지 않는다: {output}")
    missing = [_model_path(models, year, kind, h) for year in years for kind in MODEL_TYPES
               if not _model_path(models, year, kind, h).is_file()]
    if missing:
        raise FileNotFoundError(f"폴드 모델 파일이 없다: {missing[0]}")
    import shap

    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir()
    summaries = []
    dependence = {kind: [] for kind in MODEL_TYPES}
    model_hashes = {}
    samples = {}
    for year in years:
        rows = sample_test_year(features, year)
        samples[str(year)] = len(rows)
        for kind in MODEL_TYPES:
            path = _model_path(models, year, kind, h)
            model_hashes[str(path.relative_to(models))] = hashlib.sha256(path.read_bytes()).hexdigest()
            booster = load_booster(path)
            names = list(booster.feature_name())
            if not names or len(names) != len(set(names)) or set(names) - set(rows):
                raise ValueError(f"{path}: 모델 피처 이름과 입력이 맞지 않는다")
            matrix = rows[names]
            if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy(dtype=float)).all():
                raise ValueError(f"{year} {kind}: 모델 입력에 결측/무한값이 있다")
            values = _values(shap.TreeExplainer(booster), matrix, binary=(kind == "gate"))
            means = np.abs(values).mean(axis=0)
            summaries.extend({"model": kind, "year": year, "feature": name, "mean_abs_shap": float(value)}
                             for name, value in zip(names, means))
            dependence[kind].append((matrix, values))
    table = pd.DataFrame(summaries)
    table.sort_values(["model", "year", "mean_abs_shap", "feature"],
                      ascending=[True, True, False, True]).to_csv(output / "feature_importance_all.csv", index=False)
    top10 = table.sort_values(["model", "year", "mean_abs_shap", "feature"],
                              ascending=[True, True, False, True]).groupby(["model", "year"]).head(10)
    top10.to_csv(output / "top10_by_year.csv", index=False)
    stability = []
    for kind in MODEL_TYPES:
        wide = table.loc[table.model == kind].pivot(index="feature", columns="year", values="mean_abs_shap")
        if wide.isna().any().any():
            raise ValueError(f"{kind}: 연도별 피처 집합이 달라 순위 안정성을 비교할 수 없다")
        for i, left in enumerate(years):
            for right in years[i + 1:]:
                coefficient = float(spearmanr(wide[left], wide[right]).statistic)
                stability.append({"model": kind, "year_a": left, "year_b": right,
                                  "spearman_rho": coefficient})
        # 모든 연도의 표본 행을 합쳐 평균 |SHAP| 상위 3개를 고른다.
        matrices, values = zip(*dependence[kind])
        if any(list(m.columns) != list(matrices[0].columns) for m in matrices):
            raise ValueError(f"{kind}: 연도별 피처 순서가 다르다")
        matrix = pd.concat(matrices, ignore_index=True)
        joined = np.concatenate(values, axis=0)
        ranked = sorted(zip(matrix.columns, np.abs(joined).mean(axis=0)), key=lambda x: (-x[1], x[0]))
        for name, _ in ranked[:3]:
            _write_dependence(matrix, joined, name, output / "figures" / f"{kind}_{name}.png")
    pd.DataFrame(stability).to_csv(output / "rank_stability.csv", index=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (output / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    config = Path(__file__).resolve().parents[2] / "configs" / "prereg.yaml"
    (output / "config.yaml").write_bytes(config.read_bytes())
    record = {"years": years, "sample_size_per_year": SAMPLE_SIZE, "seed": SEED, "h": h,
              "sample_rows": samples, "input_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
              "model_sha256": model_hashes, "git_commit": commit}
    (output / "metrics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="C-06 폴드별 SHAP 보고서")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True, help="A walk-forward reports/{run_id}/models")
    parser.add_argument("--output", type=Path, required=True, help="빈 reports/{run_id} 폴더")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2016, 2026)))
    parser.add_argument("--h", type=int, default=20)
    args = parser.parse_args(argv)
    run(args.features, args.models, args.output, years=args.years, h=args.h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
