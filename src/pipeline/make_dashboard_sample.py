"""대시보드 확인용 샘플 데이터 생성 (D-05). 실제 폴더에 쓰지 않는다.

    python -m src.pipeline.make_dashboard_sample --out _sample
    DASHBOARD_DATA_ROOT=_sample streamlit run app/Home.py

종목코드와 수치는 모두 가짜다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import io
from src.pipeline.run_weekly import empty_frame

KST = "Asia/Seoul"
SIGNAL_DATES = ("2026-11-06", "2026-11-13")
TICKERS = [f"{i:06d}" for i in range(10, 410, 10)]
REPO = Path(__file__).resolve().parents[2]
SAMPLE_MARKER = ".dashboard_sample"  # 이 생성기가 만든 폴더 표시


def write_sample(root: Path) -> None:
    """root에 샘플을 쓴다. 저장소 루트와 샘플 폴더가 아닌 기존 폴더는 ValueError로 거부한다."""
    root = root.resolve()
    _check_output(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / SAMPLE_MARKER).touch()
    saved = io.ROOT
    io.ROOT = root
    try:
        _write_all()
    finally:
        io.ROOT = saved


def _check_output(root: Path) -> None:
    if root == REPO or root in REPO.parents:
        raise ValueError(f"저장소 루트에는 샘플을 쓸 수 없다: {root}")
    if root.exists() and (not root.is_dir() or (any(root.iterdir()) and not (root / SAMPLE_MARKER).exists())):
        raise ValueError(f"샘플 폴더가 아닌 기존 경로에는 쓸 수 없다: {root}")


def _write_all() -> None:
    n = len(TICKERS)
    days = pd.bdate_range("2026-08-03", SIGNAL_DATES[-1])
    io.write_table("index", pd.DataFrame({
        "date": days,
        "index_code": ["KOSPI200"] * len(days),
        "close": np.full(len(days), 400.0),
    }))

    for shift, date in enumerate(SIGNAL_DATES):
        day = pd.Timestamp(date)
        order = np.roll(np.arange(n), shift * 3)
        io.write_table("signals", pd.DataFrame({
            "date": [day] * n,
            "ticker": [TICKERS[i] for i in order],
            "rank_score": np.linspace(0.99, 0.01, n),
            "rank": np.arange(1, n + 1, dtype="int32"),
            "crash_prob": np.linspace(0.05, 0.60, n),
            "crash_flag": np.array([0] * (n - 4) + [1] * 4, dtype="int8"),
            "model_version": ["baseline"] * n,
        }), date=date)

    date = SIGNAL_DATES[-1]
    holdings = TICKERS[:5]
    io.write_table("positions", pd.DataFrame({
        "date": [pd.Timestamp(date)] * 6,
        "ticker": [*holdings, "CASH"],
        "target_weight": [0.1] * 5 + [0.5],
        "constructor": ["buffer"] * 6,
        "exposure": [0.5] * 6,
        "regime_mc": np.zeros(6, dtype="int8"),
    }), date=date)

    # 경보가 없는 날도 0행 파일을 둔다. 파일 없음은 미수집이다 (D-06)
    alert_day = "2026-11-11"
    for day in pd.date_range(days[0], SIGNAL_DATES[-1], freq="D").strftime("%Y-%m-%d"):
        if day != alert_day:
            io.write_table("alerts", empty_frame("alerts"), date=day)
    rcept =pd.to_datetime([f"{alert_day} 16:20:00", pd.NaT]).tz_localize(KST)
    io.write_table("alerts", pd.DataFrame({
        "date": [pd.Timestamp(alert_day)] * 2,
        "ticker": [TICKERS[6], TICKERS[7]],
        "trigger_type": ["T1", "T5"],
        "rcept_no": ["20261111000001", "KIND:20261111000002"],
        "report_nm": ["샘플 공시 A", "샘플 공시 B"],
        "rcept_dt": rcept,
        "available_at": pd.to_datetime([f"{alert_day} 16:20:00", f"{alert_day} 23:59:59"]).tz_localize(KST),
        "generated_at": pd.to_datetime([f"{alert_day} 16:20:00", f"{alert_day} 23:59:59"]).tz_localize(KST),
    }), date=alert_day)

    io.write_evidence(date, [{
        "date": date,
        "ticker": TICKERS[-1],
        "signal_type": "crash",
        "summary": "샘플 요약입니다. 실제 공시나 뉴스가 아닙니다.",
        "sources": [{"n": 1, "type": "disclosure", "id": "sample", "title": "샘플 출처",
                     "url": "https://example.com/sample", "timestamp": f"{date}T09:00:00+09:00"}],
        "disclaimer": io.DISCLAIMER,
    }], )

    io.write_table("forward_log", pd.DataFrame({
        "signal_date": [pd.Timestamp(SIGNAL_DATES[0])],
        "commit_hash": ["0000000sample"],
        "signal_source": ["baseline"],
        "has_positions": np.zeros(1, dtype="int8"),
        "label_ready_date": [pd.Timestamp("2026-12-07")],
        "mean_er_top20": [np.nan],
        "mean_er_flagged": [np.nan],
        "portfolio_return": [np.nan],
        "bm_return": [np.nan],
    }))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="대시보드 샘플 데이터 생성")
    parser.add_argument("--out", default="_sample", help="출력 폴더")
    args = parser.parse_args(argv)
    try:
        write_sample(Path(args.out))
    except ValueError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    print(f"샘플 데이터를 {args.out}에 만들었다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
