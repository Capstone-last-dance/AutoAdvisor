"""사전등록 G2: A의 고정 walk-forward 신호로 개인 행동 프록시를 재생한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.retail_proxy import cohort_effect_hac, evaluate_g2, select_retail_cohort, simulate_retail_proxy
from src.common.guards import DataAccessPolicy
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.params import load_prereg


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_handoff(market: Path, signals_path: Path, handoff_path: Path) -> dict[str, str]:
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))["handoff"]
    paths = {name: market / f"{name}.parquet" for name in
             ("prices", "status", "index", "universe", "delisted", "delisted_reused")}
    paths["signals"] = signals_path
    expected = {name: item["sha256"] for name, item in handoff["inputs"].items() if name in paths}
    expected["signals"] = handoff["test_signals"]["sha256"]
    measured = {name: _sha256(path) for name, path in paths.items()}
    for name, digest in measured.items():
        if digest != expected[name]:
            raise ValueError(f"{name} 지문이 B 인계와 다릅니다: {digest} != {expected[name]}")
    return measured


def _cohorts(market: Path, signals: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, dict[str, int]]:
    first, last = signals["date"].min(), signals["date"].max()
    universe = pd.read_parquet(market / "universe.parquet", columns=["date", "ticker", "in_universe"],
                               filters=[("date", ">=", first), ("date", "<=", last)])
    eligible = universe.loc[universe["in_universe"], ["date", "ticker"]]
    signal_keys = signals[["date", "ticker"]]
    membership = signal_keys.merge(eligible, on=["date", "ticker"], how="outer", indicator=True,
                                   validate="one_to_one")
    if not membership["_merge"].eq("both").all():
        raise ValueError(f"A 신호와 U_t가 다릅니다: {membership['_merge'].value_counts().to_dict()}")
    # 첫 시험 신호일(2016-01-08)의 t-64까지 포함한다. 미래 연도는 읽지 않는다.
    raw = pd.read_parquet(market / "prices.parquet", columns=["date", "ticker", "value"],
                          filters=[("date", ">=", pd.Timestamp("2015-09-01")),
                                   ("date", "<=", last)])
    values = raw.pivot(index="date", columns="ticker", values="value").sort_index()
    results = []
    counts = {"missing_history": 0, "zero_baseline": 0, "cohorts_short": 0}
    for day, group in eligible.groupby("date", sort=True):
        selected = select_retail_cohort(values, group["ticker"], day.strftime("%Y-%m-%d"), top_n=top_n)
        results.append(selected.cohort)
        counts["missing_history"] += selected.missing_history
        counts["zero_baseline"] += selected.zero_baseline
        counts["cohorts_short"] += len(selected.cohort) < top_n
    return pd.concat(results, ignore_index=True), counts


def _engine(market: Path, tickers: set[str], cfg: dict):
    begin, end = pd.Timestamp("2015-08-01"), pd.Timestamp("2025-12-30")
    dates = [("date", ">=", begin), ("date", "<=", end)]
    names = [("ticker", "in", sorted(tickers))]
    prices = pd.read_parquet(market / "prices.parquet", columns=["date", "ticker", "close", "close_raw"],
                             filters=dates + names)
    status = pd.read_parquet(market / "status.parquet", columns=["date", "ticker", "is_halted"],
                             filters=dates + names)
    index = pd.read_parquet(market / "index.parquet", filters=dates)
    delist_filters = [("delist_date", ">", begin), ("delist_date", "<=", end)] + names
    delisted = pd.read_parquet(market / "delisted.parquet", filters=delist_filters)
    reused = pd.read_parquet(market / "delisted_reused.parquet", filters=delist_filters)
    return build_engine(engine_inputs(prices, status, index, delisted, reused), cfg)


def run(market: Path, signals_path: Path, labels_path: Path, handoff_path: Path,
        cfg_path: Path, out: Path, *, run_date: date) -> dict:
    if out.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {out}")
    cfg = load_prereg(cfg_path)
    fingerprints = _check_handoff(market, signals_path, handoff_path)
    fingerprints["labels"] = _sha256(labels_path)
    fingerprints["prereg"] = _sha256(cfg_path)
    signals = pd.read_parquet(signals_path)
    signals["date"] = pd.to_datetime(signals["date"])
    if signals.duplicated(["date", "ticker"]).any() or signals["date"].dt.year.min() != 2016 \
            or signals["date"].dt.year.max() != 2025 or signals["date"].nunique() != 520:
        raise ValueError("G2 판정 신호는 중복 없는 2016~2025년 520개 신호일이어야 합니다")
    proxy_cfg = cfg["retail_proxy"]
    cohorts, quality = _cohorts(market, signals, int(proxy_cfg["top_n"]))
    labels = pd.read_parquet(labels_path, columns=["date", "ticker", "h", "er"])
    policy = DataAccessPolicy(
        date(2025, 12, 31),
        *(date.fromisoformat(x) for x in cfg["splits"]["holdout_signal_range"]),
        date.fromisoformat(cfg["splits"]["holdout_open_date"]),
    )
    guard = {"policy": policy, "run_date": run_date, "allow_holdout": False}
    effect = cohort_effect_hac(cohorts, signals, labels,
                               horizon=int(cfg["labels"]["main_horizon_days"]),
                               maxlags=int(cfg["kpi"]["G2"]["nw_maxlags"]), **guard)
    engine = _engine(market, set(cohorts["ticker"]), cfg)
    kwargs = {"end_date": engine.days[-1], "holding_days": int(proxy_cfg["holding_days"]),
              "cohort_fraction": float(proxy_cfg["cohort_fraction_of_nav"]), **guard}
    baseline = simulate_retail_proxy(engine, cohorts, signals, gate=False, **kwargs)
    guarded = simulate_retail_proxy(engine, cohorts, signals, gate=True, **kwargs)
    kpi = cfg["kpi"]["G2"]
    decision = evaluate_g2(baseline, guarded, effect,
                           min_mdd_reduction_rel=float(kpi["min_mdd_reduction_rel"]),
                           max_t_stat=float(kpi["max_t_stat"]))
    metrics = {
        "classification": "preregistered G2 evaluation; 2016-2025 walk-forward",
        "inputs_sha256": fingerprints,
        "signal_days": int(signals["date"].nunique()), "cohort_rows": len(cohorts),
        "cohort_quality": quality,
        "baseline": {"mdd": baseline.mdd, "cagr": baseline.cagr},
        "gate": {"mdd": guarded.mdd, "cagr": guarded.cagr},
        "effect": {"mean_d": effect.mean_d, "hac_t": effect.hac_t_statistic,
                   "signal_days_used": len(effect.daily_d), "signal_days_undefined": effect.signal_days_undefined,
                   "missing_pairs": effect.missing_pairs},
        "G2": {"mdd_reduction_relative": decision.mdd_reduction_relative,
               "cagr_gate_not_lower": decision.cagr_gate_not_lower,
               "hac_t": decision.hac_t_statistic, "passed": decision.passed},
    }
    out.mkdir(parents=True)
    shutil.copyfile(cfg_path, out / "config.yaml")
    (out / "git_commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cohorts.to_parquet(out / "cohorts.parquet", index=False)
    pd.DataFrame({"baseline": baseline.daily_nav, "gate": guarded.daily_nav}).rename_axis("date").to_csv(
        out / "daily_nav.csv")
    effect.daily_d.rename_axis("date").to_csv(out / "daily_d.csv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, default=Path("configs/prereg.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.market, args.signals, args.labels, args.handoff, args.prereg, args.out,
                 run_date=datetime.now(ZoneInfo("Asia/Seoul")).date())
    print(json.dumps(result["G2"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
