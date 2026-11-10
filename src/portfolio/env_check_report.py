"""작업 B-05 결과 리포트: 실제 A 베이스라인 신호로 완료 기준 (1)·(2)를 확인해 reports/{run_id}/에 남긴다 (R-10).

    python -m src.portfolio.env_check_report --run-id 20261109_B_env_check [--config configs/prereg.yaml]

- 표는 io.py로만 읽는다 (R-18): prices, status, index, delisted, delisted_reused, universe, sector.
- 신호일: universe의 2011-01-01 ~ 2025-12-31 날짜다. calendar.signal_day(주마다 마지막 거래일)와 같은지 확인한다.
  2026년 홀드아웃 신호일은 쓰지 않는다 (4.7절).
- 신호: A의 `src.models.baseline.baseline_signals` (A-04, model_version = "baseline")를 신호일마다 U_t로 부른다.
- 피처 (B가 읽는 4개):
  - vol_20: A의 `compute_stock_features`를 쓰고, `rank_and_fill`로 U_t 안 백분위로 바꾼다 (결측은 0.5).
  - mkt_vol_20, mkt_ret_20: A의 `compute_market_features`를 쓴다.
  - regime_mc: B의 `market_state_table`로 만든다. A가 피처를 만들 때 부르는 B 함수와 같은 값이다.
- M9 쿨다운과 이벤트 청산은 비어 있다 (M9 백테스트 경보 전). metrics.json에 그렇게 적는다.
- (1) run_full_episode: 행동 0과 무작위 행동(시드 고정)으로 첫 신호일부터 마지막 신호일까지 돈다.
- (2) compare_buffer_and_env: 버퍼 규칙 실행과, 버퍼 비중을 넣은 RL 환경 실행의 체결·비용·NAV를 비교한다.
- 입력 파일의 SHA-256을 남긴다. A 11/8 공지대로 수정종가가 소급해서 바뀔 수 있기 때문이다.

reports/{run_id}/에 config.yaml(사본), git_commit.txt, metrics.json을 쓴다 (9.3절).
같은 run_id 폴더가 있으면 덮어쓰지 않고 멈춘다 (R-11).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.common import io
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.env_check import compare_buffer_and_env, run_full_episode
from src.portfolio.params import PREREG_PATH, BufferRuleParams, RiskLayerParams, RLParams, load_prereg
from src.portfolio.risk_layer import market_state_table
from src.portfolio.rl_data import build_snapshots
from src.portfolio.rl_env import ACTION_BOUND
from src.portfolio.vol_target_report import ROOT, RUN_ID_RE, _git_commit
from src.portfolio.weekly import signal_dates_between

TABLES = ("prices", "status", "index", "delisted", "delisted_reused", "universe", "sector")
FIRST_SIGNAL_DATE = "2011-01-01"  # 4.7절 첫 학습 신호일
LAST_SIGNAL_DATE = "2025-12-31"  # walk-forward 끝. 홀드아웃(2026-01-01~)은 쓰지 않는다
TOLERANCE = 1e-12  # 완료 기준 (2)의 '같다'. 실제 차이는 metrics.json에 그대로 남긴다
CHUNK = 300  # vol_20을 계산할 때 한 번에 넣는 종목 수 (메모리). 종목별 시계열 계산이라 결과는 같다

# (close 피벗, U_t 종목 목록, 신호일) → 9.3절 signals 행
SignalFunction = Callable[[pd.DataFrame, list, pd.Timestamp], pd.DataFrame]


# ---------------------------------------------------------------------------
# 입력
# ---------------------------------------------------------------------------


def signal_dates(universe: pd.DataFrame, trading_days: Sequence[Any], start: str, end: str) -> list[str]:
    """universe의 start ~ end 신호일. 주마다 마지막 거래일(calendar.signal_day)과 다르면 멈춘다."""
    days = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(trading_days)))))
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    expected = [d for d in signal_dates_between(days, lo, days[-1] + pd.Timedelta(days=1)) if d <= hi.strftime("%Y-%m-%d")]
    u = pd.to_datetime(universe["date"])
    have = sorted({d.strftime("%Y-%m-%d") for d in u[(u >= lo) & (u <= hi)]})
    if have != expected:
        extra, missing = sorted(set(have) - set(expected)), sorted(set(expected) - set(have))
        raise ValueError(f"universe 날짜가 신호일 달력과 다르다. 신호일이 아닌 날짜: {extra[:5]}, 빠진 신호일: {missing[:5]}")
    return have


def kospi200(index: pd.DataFrame) -> pd.Series:
    k = index.loc[index["index_code"] == "KOSPI200"]
    return pd.Series(k["close"].to_numpy(dtype="float64"), index=pd.to_datetime(k["date"])).sort_index()


def members_by_date(universe: pd.DataFrame, dates: Sequence[str]) -> dict[pd.Timestamp, list[str]]:
    u = universe.loc[universe["in_universe"].astype(bool)]
    by = {pd.Timestamp(d): sorted(g["ticker"].astype(str)) for d, g in u.groupby(pd.to_datetime(u["date"]))}
    return {pd.Timestamp(d): by[pd.Timestamp(d)] for d in dates}


def _pivot(prices: pd.DataFrame, column: str, tickers: Sequence[str]) -> pd.DataFrame:
    p = prices.loc[prices["ticker"].isin(set(tickers))]
    return p.pivot(index="date", columns="ticker", values=column).sort_index()


def baseline_signal_table(close: pd.DataFrame, members: Mapping[pd.Timestamp, list], signal_fn: SignalFunction) -> pd.DataFrame:
    return pd.concat([signal_fn(close, tickers, t) for t, tickers in members.items()], ignore_index=True)


def feature_table(
    prices: pd.DataFrame, bm_close: pd.Series, members: Mapping[pd.Timestamp, list], risk: RiskLayerParams
) -> pd.DataFrame:
    """B가 읽는 features 4개 (9.3절 열 이름). 정의는 A의 함수를 그대로 쓴다 (R-14)."""
    from src.features.price_features import compute_market_features, compute_stock_features, rank_and_fill

    union = sorted({tk for tickers in members.values() for tk in tickers})
    vol_parts = []
    for i in range(0, len(union), CHUNK):
        chunk = union[i : i + CHUNK]
        inputs = [_pivot(prices, c, chunk) for c in ("close", "close_raw", "value", "mcap")]
        vol_parts.append(compute_stock_features(*inputs)["vol_20"])
    vol_20 = pd.concat(vol_parts, axis=1)

    regime = market_state_table(bm_close, risk, mode="no_exposure")["regime_mc"]
    market = compute_market_features(bm_close, regime, pd.Series(np.nan, index=bm_close.index))
    rows = []
    for t, tickers in members.items():
        ranked = rank_and_fill(pd.DataFrame({"vol_20": vol_20.loc[t].reindex(tickers)}), rank_cols=["vol_20"])
        rows.append(pd.DataFrame({
            "date": t, "ticker": tickers, "vol_20": ranked["vol_20"].to_numpy(),
            "mkt_vol_20": market.at[t, "mkt_vol_20"], "mkt_ret_20": market.at[t, "mkt_ret_20"],
            "regime_mc": int(market.at[t, "regime_mc"]),
        }))
    return pd.concat(rows, ignore_index=True)


def _a_signal_function() -> SignalFunction:
    try:
        from src.models.baseline import baseline_signals
        import src.features.price_features  # noqa: F401  (feature_table이 쓴다)
    except ImportError as e:
        raise SystemExit(f"A의 베이스라인 신호·피처 코드가 없다 (A PR #35 병합 전): {e}") from e
    return baseline_signals


def input_fingerprints(tables: Mapping[str, pd.DataFrame], root: Path) -> dict[str, dict[str, Any]]:
    """9.3절 data/processed/{name}.parquet 파일의 SHA-256과 행 수."""
    out = {}
    for name, df in tables.items():
        path = root / "data" / "processed" / f"{name}.parquet"
        out[name] = {"path": path.relative_to(root).as_posix(), "rows": len(df),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return out


def content_sha256(df: pd.DataFrame) -> str:
    """생성한 표(신호·피처)의 내용 지문. 같은 입력과 코드로 다시 만들면 같다."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 완료 기준
# ---------------------------------------------------------------------------


def env_check_metrics(
    tables: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    dates: Sequence[str],
    signals: pd.DataFrame,
    features: pd.DataFrame,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """완료 기준 (1)·(2)의 결과. M9 쿨다운·이벤트 청산은 비운다."""
    rl, risk, buffer = RLParams.from_prereg(cfg), RiskLayerParams.from_prereg(cfg), BufferRuleParams.from_prereg(cfg)
    inputs = engine_inputs(*(tables[n] for n in ("prices", "status", "index", "delisted", "delisted_reused")))
    start = perf_counter()
    snaps = build_snapshots(dates, signals, features, tables["universe"], tables["sector"], inputs.bm_close,
                            lambda d: (), risk)
    snapshot_seconds = perf_counter() - start

    rng = np.random.default_rng(seed)
    episodes = {
        "zero_action": run_full_episode(snaps, build_engine(inputs, cfg), rl, risk),
        "random_action": run_full_episode(
            snaps, build_engine(inputs, cfg), rl, risk,
            policy=lambda obs: rng.uniform(-ACTION_BOUND, ACTION_BOUND, rl.slots + 1).astype(np.float32),
        ),
    }
    compare = compare_buffer_and_env(snaps, lambda: build_engine(inputs, cfg, record_ledger=True), rl, risk, buffer)

    ran_through = all(
        e["n_steps"] == len(dates) and e["first_date"] == dates[0] and e["last_date"] == dates[-1]
        for e in episodes.values()
    )
    diffs = {k: v for k, v in compare.items() if k.startswith("max_")}
    same = not compare["mismatched_days"] and all(v <= TOLERANCE for v in diffs.values())
    return {
        "criteria": {
            "1_episode_runs_to_the_end": ran_through,
            "2_same_weights_same_fills_and_costs": same,
            "tolerance": TOLERANCE,
        },
        "signal_dates": {"n": len(dates), "first": dates[0], "last": dates[-1]},
        "universe_size": {
            "mean": float(np.mean([len(s.tickers) for s in snaps])),
            "min": int(min(len(s.tickers) for s in snaps)), "max": int(max(len(s.tickers) for s in snaps)),
        },
        "m9": "쿨다운·이벤트 청산 없음 (M9 백테스트 경보 전)",
        "random_action_seed": seed,
        "episodes": episodes,
        "buffer_vs_env": compare,
        "seconds": {"snapshots": snapshot_seconds},
    }


def main(argv: Sequence[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="B-05 완료 기준 리포트 (실제 A 베이스라인 신호)")
    ap.add_argument("--run-id", required=True, help="{YYYYMMDD}_B_{짧은설명}")
    ap.add_argument("--config", default=str(PREREG_PATH), help="부록 1 설정 (기본 configs/prereg.yaml)")
    ap.add_argument("--seed", type=int, default=0, help="무작위 행동 시드")
    args = ap.parse_args(argv)
    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit(f"run_id는 '{{YYYYMMDD}}_B_{{짧은설명}}' 형식이어야 한다: {args.run_id!r}")
    out = ROOT / "reports" / args.run_id
    if out.exists():
        raise SystemExit(f"{out}가 이미 있다. 결과를 덮어쓰지 않는다 (R-11)")

    signal_fn = _a_signal_function()
    cfg_path = Path(args.config)
    cfg = load_prereg(cfg_path)
    tables = {name: io.read_table(name) for name in TABLES}
    dates = signal_dates(tables["universe"], kospi200(tables["index"]).index, FIRST_SIGNAL_DATE, LAST_SIGNAL_DATE)
    members = members_by_date(tables["universe"], dates)

    start = perf_counter()
    union = sorted({tk for tickers in members.values() for tk in tickers})
    signals = baseline_signal_table(_pivot(tables["prices"], "close", union), members, signal_fn)
    signal_seconds = perf_counter() - start
    features = feature_table(tables["prices"], kospi200(tables["index"]), members, RiskLayerParams.from_prereg(cfg))

    metrics = {"run_id": args.run_id, "task": "B-05", **env_check_metrics(tables, cfg, dates, signals, features, seed=args.seed)}
    metrics["seconds"]["baseline_signals"] = signal_seconds
    metrics["inputs"] = input_fingerprints(tables, io.ROOT)
    metrics["generated"] = {
        "signals": {"source": "src.models.baseline.baseline_signals", "rows": len(signals),
                    "model_version": sorted(signals["model_version"].unique()), "sha256": content_sha256(signals)},
        "features": {"source": "src.features.price_features (vol_20, mkt_vol_20, mkt_ret_20) + B market_state_table "
                               "(regime_mc)", "rows": len(features), "sha256": content_sha256(features)},
    }

    out.mkdir(parents=True)
    shutil.copyfile(cfg_path, out / "config.yaml")
    (out / "git_commit.txt").write_text(_git_commit(), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    c = metrics["criteria"]
    print(f"B-05 (1) 끝까지 실행: {c['1_episode_runs_to_the_end']}, (2) 버퍼 규칙과 같음: "
          f"{c['2_same_weights_same_fills_and_costs']} → {out}")
    return out


if __name__ == "__main__":
    main()
