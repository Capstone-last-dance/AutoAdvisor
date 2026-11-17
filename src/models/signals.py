"""A-08 최종 모델과 주간 신호 생성 함수 `generate_signals(date)` (4.7절).

최종 모델: 학습 2011~2023년, 검증 2024~2025년 신호일 (splits.final_split, purge 포함). M3a와 M3b를 각각 적합해 한 묶음(FinalModels)으로 저장한다.
홀드아웃(2026-01-01~)은 학습·검증·튜닝에 쓰지 않는다 (R-04). 신호 생성(추론)은 신호일 t의 피처 행만 쓴다 (R-03).
generate_signals는 모델 파일만으로 돌아간다. 매주 실행(D의 run_weekly)은 load_final로 읽은 모델에 t의 피처를 넘기면 된다.
model_version 값은 명세에 정의가 없어 호출자가 정한다 (baseline은 "baseline").
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.data.schemas import validate
from src.models import gate, ranker
from src.models.splits import final_split


@dataclass
class FinalModels:
    gate_booster: lgb.Booster
    gate_features: list
    calibrator: IsotonicRegression
    ranker_booster: lgb.Booster
    ranker_features: list
    h: int
    meta: dict
    gate_iteration: int | None = None      # 학습 직후에는 best_iteration, 파일에서 읽으면 None (저장 때 이미 잘랐다)
    ranker_iteration: int | None = None


def train_final(features: pd.DataFrame, labels: pd.DataFrame, signal_dates, trading_days, h: int = 20,
                actual_end_dates=None, **kw) -> FinalModels:
    """features·labels는 2025-12-31 이하 신호일만 쓰인다(그 이후 행이 있어도 final_split이 구간에 넣지 않는다).
    actual_end_dates를 넘기면 학습↔검증·검증↔홀드아웃 경계를 C의 실제 라벨 종료일로 purge한다 (C 리뷰 #35).
    kw는 grid, fixed, seed, n_jobs를 gate·ranker 양쪽에 넘긴다."""
    fold = final_split(signal_dates, trading_days, h=h, actual_end_dates=actual_end_dates)
    g = gate.fit_fold(features, labels, fold, require_test=False, **kw)
    r = ranker.fit_fold(features, labels, fold, h=h, require_test=False, **kw)
    meta = {"h": h, "actual_end_dates_used": actual_end_dates is not None,
            "train_signal_dates": int(len(fold.train)), "val_signal_dates": int(len(fold.val)),
            "train_dates": [f"{fold.train.min():%Y-%m-%d}", f"{fold.train.max():%Y-%m-%d}"],
            "val_dates": [f"{fold.val.min():%Y-%m-%d}", f"{fold.val.max():%Y-%m-%d}"],
            "gate_params": g.params, "gate_val_auc": g.val_auc, "ranker_params": r.params, "ranker_val_mean_rank_ic": r.val_ic}
    return FinalModels(g.model.booster_, g.features, g.calibrator, r.model.booster_, r.features, h, meta,
                       int(g.model.best_iteration_), int(r.model.best_iteration_))


def save_final(models: FinalModels, directory: Path):
    """모델 파일과 meta를 쓰고, 가중치 파일의 SHA-256을 final_sha256.json에 남긴다.

    가중치는 저장소에 커밋하지 않고 팀 드라이브 run_id 폴더에 두기로 했으므로(PM 2026-11-10),
    해시 파일이 "어느 가중치로 만든 신호인가"를 가리키는 유일한 커밋 가능한 기록이다."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    gate.save_booster(models.gate_booster, d / "final_gate.txt", models.gate_iteration)
    gate.save_booster(models.ranker_booster, d / "final_ranker.txt", models.ranker_iteration)
    joblib.dump({"calibrator": models.calibrator, "gate_features": models.gate_features,
                 "ranker_features": models.ranker_features, "h": models.h}, d / "final_aux.joblib")
    (d / "final_meta.json").write_text(json.dumps(models.meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "final_sha256.json").write_text(json.dumps(final_sha256(d), ensure_ascii=False, indent=2), encoding="utf-8")


def final_sha256(directory: Path) -> dict:
    """가중치 파일의 SHA-256 (해시 파일 자신과 meta는 뺀다)."""
    skip = {"final_sha256.json", "final_meta.json"}
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(directory).iterdir()) if p.is_file() and p.name not in skip}


def load_final(directory: Path) -> FinalModels:
    d = Path(directory)
    aux = joblib.load(d / "final_aux.joblib")
    return FinalModels(gate.load_booster(d / "final_gate.txt"), aux["gate_features"], aux["calibrator"],
                       gate.load_booster(d / "final_ranker.txt"), aux["ranker_features"], aux["h"],
                       json.loads((d / "final_meta.json").read_text(encoding="utf-8")))


def generate_signals(date, features: pd.DataFrame, models: FinalModels, model_version: str) -> pd.DataFrame:
    """신호일 date의 U_t 종목 전체에 대해 signals(9.3절) 행을 만든다. features는 compute_feature_matrix 형식이고, date가 아닌 행은 쓰지 않는다."""
    t = pd.Timestamp(date)
    x = features[features["date"] == t].reset_index(drop=True)
    if x.empty:
        raise ValueError(f"{t:%Y-%m-%d} 신호일의 피처 행이 없다")
    need = set(models.gate_features) | set(models.ranker_features)
    if need - set(x.columns):
        raise ValueError(f"모델이 쓰는 피처가 입력에 없다: {sorted(need - set(x.columns))}")

    out = x[["date", "ticker"]].copy()
    raw = models.gate_booster.predict(x[models.gate_features], num_iteration=models.gate_iteration)
    out["raw_prob"] = raw
    out["crash_prob"] = models.calibrator.predict(raw).astype("float64")
    out["crash_flag"] = gate.flag_top(out, "crash_prob", "raw_prob")
    out["rank_score"] = models.ranker_booster.predict(x[models.ranker_features], num_iteration=models.ranker_iteration).astype("float64")
    out["rank"] = ranker.rank_in_universe(out, "rank_score")
    out["model_version"] = model_version
    out = out[["date", "ticker", "rank_score", "rank", "crash_prob", "crash_flag", "model_version"]]
    out = out.sort_values("rank").reset_index(drop=True)
    validate("signals", out)   # io.py(D-02)가 생기면 io.validate로 바꾼다 (R-18). 지금은 로컬 스키마 검증
    return out


# ---------------------------------------------------------------------------
# run_weekly 3단계 (5.2절 M3a·M3b). D가 STEPS에 "src.models.signals:m3_step"으로 등록한다.
# ---------------------------------------------------------------------------

FINAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "final"   # save_final이 쓴 폴더 (A 소유)
BASELINE_INPUTS = ("universe", "prices")    # 베이스라인이 읽는 표
ML_INPUTS = ("universe", "features")        # 최종 모델이 읽는 표


def ml_signals_from() -> pd.Timestamp:
    """사전등록 `splits.forward_test.ml_signals_from`(2026-11-27). 그 전 신호일은 베이스라인이다 (6.8절)."""
    from src.portfolio.params import load_prereg   # 설정만 읽는다. 함수 안에서 읽어 수집 환경(portfolio 없음)에서도 이 모듈을 쓴다

    return pd.Timestamp(load_prereg()["splits"]["forward_test"]["ml_signals_from"])


def m3_step(ctx) -> None:
    """5.2절 3단계. 신호일 t의 `signals/{t}.parquet`을 쓴다.

    - `ml_signals_from` 전 신호일은 베이스라인(6.3절 플래그 + 6.4절 B1, `model_version = "baseline"`)이다.
      11/20 기록은 여기까지만 있으면 된다 (`forward_test.expected_input`).
    - 그 날부터는 A-08 최종 모델(`FINAL_MODEL_DIR`)로 만든다. 모델이 없으면 단계가 실패한다 — 스텁으로 대신하지 않는다.
    - 앞 단계가 스텁으로 쓴 표를 입력으로 받으면 계산하지 않고 0행 `signals`를 스텁으로 쓴다 (6.12절).
      7단계가 그 실행을 forward 커밋에서 뺀다.
    - 베이스라인은 U_t로 행을 만들고, ML 경로는 피처 행렬에서 만들기 때문에 쓰기 전에 U_t와 같은 종목 집합인지 본다.
      다르면 D의 `forward_inputs`가 멈춘다 (B `weekly.py` 계약).
    """
    t = pd.Timestamp(ctx.date)
    use_ml = t >= ml_signals_from()
    stubbed = [name for name in (ML_INPUTS if use_ml else BASELINE_INPUTS) if name in getattr(ctx, "stub_tables", ())]
    if stubbed:
        from src.pipeline.run_weekly import empty_frame   # D의 스텁 형식 (9.3절 열, 0행)

        ctx.write_table("signals", empty_frame("signals"), date=ctx.date)
        ctx.log.warning("3단계 A: 입력 표가 스텁이라(%s) 0행 signals를 스텁으로 쓴다", ", ".join(stubbed))
        return

    if not use_ml:
        from src.models import baseline

        baseline.run_step(ctx)
        ctx.log.info("3단계 A: %s는 ML 신호 시작일 전이라 베이스라인으로 만든다 (B1 mom_12_1, vol_60 상위 %.0f%%)",
                     ctx.date, 100 * baseline.FLAG_FRACTION)
        return

    models = load_final(FINAL_MODEL_DIR)   # 없으면 FileNotFoundError로 단계가 멈춘다
    version = str(models.meta.get("model_version") or FINAL_MODEL_DIR.name)
    if version == "baseline":
        raise ValueError(f"{FINAL_MODEL_DIR}의 model_version이 'baseline'이다. "
                         "D의 forward_test가 신호 종류를 이 값으로 가르므로 ML 모델에는 다른 이름을 써야 한다")
    signals = generate_signals(ctx.date, ctx.read_table("features", date=ctx.date), models, version)
    _check_universe(signals, _universe_tickers(ctx, t), t)
    ctx.write_table("signals", signals, date=ctx.date)
    ctx.log.info("3단계 A: %s %d종목, 플래그 %d개, 모델 %s (학습 %s~%s)", ctx.date, len(signals),
                 int(signals["crash_flag"].sum()), version, *models.meta.get("train_dates", ("?", "?")))


def _universe_tickers(ctx, t: pd.Timestamp) -> list:
    u = ctx.read_table("universe")
    tickers = u.loc[(u["date"] == t) & u["in_universe"], "ticker"].tolist()
    if not tickers:
        raise ValueError(f"{t:%Y-%m-%d}: universe에 그 신호일이 없거나 U_t가 비어 있다")
    return tickers


def _check_universe(signals: pd.DataFrame, u_t: list, t: pd.Timestamp) -> None:
    missing = sorted(set(u_t) - set(signals["ticker"]))
    extra = sorted(set(signals["ticker"]) - set(u_t))
    if missing or extra:
        raise ValueError(f"{t:%Y-%m-%d}: signals 종목이 U_t와 다르다 (빠진 {len(missing)}개 {missing[:5]}, "
                         f"U_t 밖 {len(extra)}개 {extra[:5]}). 이대로 쓰면 D의 forward_inputs가 멈춘다")
