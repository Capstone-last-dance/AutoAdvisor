"""9.3절 표 → M5 체결 엔진 입력 (작업 B-05, B 소유).

A의 prices, status, index, delisted 표와 부록 1의 costs, execution 값으로 C의 ExecutionEngine을 만든다.
엔진 자체(체결, 비용, NAV, 정지·폐지 처리)는 C 소유이고 여기서는 입력 형식만 맞춘다 (R-17).

- closes: prices.close (수정종가). 상장 전·폐지 후 칸은 결측으로 둔다. 정지일 평가 가격은 엔진이
  마지막 거래일 수정종가로 바꾼다 (DEC-25)
- halted: 일별 status.is_halted (DEC-24). v1은 volume == 0이다 (DEC-26). status 행이 없는 칸은 결측
- bm_close: index에서 index_code = KOSPI200 (DEC-27)
- delistings: delisted.delist_date는 마지막 가격일의 다음 거래일이므로 엔진에는 전 거래일로 옮겨 넘긴다.
  last_price(원종가)는 그날의 close ÷ close_raw로 수정종가 단위로 바꾼다
- delisted_reused: 재사용 종목코드의 옛 회사 폐지 (DEC-28). 공용 스키마 밖 파일이다
- M9 쿨다운·이벤트 청산은 PM 함수가 나오기 전까지 호출자가 넘기지 않으면 비어 있다
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.params import _expect

BENCHMARK = "KOSPI200"
REUSED_DELISTED_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "delisted_reused.parquet"


@dataclass(frozen=True)
class EngineInputs:
    """ExecutionEngine 생성자 입력. 표 변환은 비싸므로 한 번 만들고 실행마다 새 엔진을 만든다."""

    closes: pd.DataFrame  # 거래일 × 종목코드, 수정종가
    halted: pd.DataFrame  # 같은 모양, boolean (결측 = 알 수 없음)
    bm_close: pd.Series  # 거래일, 코스피 200 종가
    delistings: dict[str, tuple[str, float | None]]  # 종목코드 → (마지막 가격일, 수정종가 단위 청산가)


def _delistings(
    delisted: pd.DataFrame, closes: pd.DataFrame, close_raw: pd.DataFrame
) -> dict[str, tuple[str, float | None]]:
    days = closes.index
    out: dict[str, tuple[str, float | None]] = {}
    for ticker, delist_date, last_price in delisted[["ticker", "delist_date", "last_price"]].itertuples(index=False):
        ticker, delist_date = str(ticker), pd.Timestamp(delist_date)
        if ticker in out:
            raise ValueError(f"{ticker}: 폐지 기록이 두 개다. 같은 코드의 상장 구간 구분은 v2에서 정한다 (DEC-28)")
        if ticker not in closes.columns:
            raise ValueError(f"{ticker}: 폐지 종목인데 가격 표에 없다")
        i = days.searchsorted(delist_date)
        if i == 0 or i >= len(days) or days[i] != delist_date:
            raise ValueError(f"{ticker}: delist_date {delist_date:%Y-%m-%d}의 전 거래일이 가격 표에 없다")
        last_day = days[i - 1]
        close, raw = closes.at[last_day, ticker], close_raw.at[last_day, ticker]
        if not (np.isfinite(close) and np.isfinite(raw) and raw > 0):
            raise ValueError(f"{ticker}: 마지막 가격일 {last_day:%Y-%m-%d}에 종가가 없다")
        if np.isfinite(closes.at[delist_date, ticker]):
            raise ValueError(f"{ticker}: delist_date {delist_date:%Y-%m-%d}에도 종가가 있다")
        price = None if pd.isna(last_price) else float(last_price) * float(close) / float(raw)
        out[ticker] = (last_day.strftime("%Y-%m-%d"), price)
    return out


def engine_inputs(
    prices: pd.DataFrame,
    status: pd.DataFrame,
    index: pd.DataFrame,
    delisted: pd.DataFrame,
    delisted_reused: pd.DataFrame | None = None,
) -> EngineInputs:
    """9.3절 형식의 표로 엔진 입력을 만든다. 거래일은 prices의 날짜이고 index도 같은 날짜를 모두 가져야 한다."""
    p = prices.assign(date=pd.to_datetime(prices["date"]), ticker=prices["ticker"].astype(str))
    closes = p.pivot(index="date", columns="ticker", values="close").sort_index().astype("float64")
    close_raw = p.pivot(index="date", columns="ticker", values="close_raw").reindex_like(closes).astype("float64")
    closes.index.name = closes.columns.name = None

    s = status.assign(date=pd.to_datetime(status["date"]), ticker=status["ticker"].astype(str))
    halted = s.pivot(index="date", columns="ticker", values="is_halted").reindex_like(closes).astype("boolean")
    if (halted.isna() & closes.notna()).to_numpy().any():
        raise ValueError("종가가 있는데 status 행이 없는 칸이 있다 (DEC-24는 모든 거래일의 행을 요구한다)")

    k = index[index["index_code"] == BENCHMARK]
    bm = pd.Series(k["close"].to_numpy(dtype="float64"), index=pd.to_datetime(k["date"])).sort_index()
    if not closes.index.isin(bm.index).all():
        raise ValueError(f"{BENCHMARK} 종가가 없는 거래일이 있다")
    bm = bm.reindex(closes.index)

    rows = [delisted] if delisted_reused is None or delisted_reused.empty else [delisted, delisted_reused]
    delistings = _delistings(pd.concat(rows, ignore_index=True), closes, close_raw)
    return EngineInputs(closes, halted, bm, delistings)


def load_engine_inputs() -> EngineInputs:
    """io.py로 9.3절 표를 읽어 엔진 입력을 만든다 (R-18). delisted_reused는 공용 스키마 밖 파일이라
    있을 때만 직접 읽는다 (DEC-28)."""
    from src.common import io

    reused = pd.read_parquet(REUSED_DELISTED_PATH) if REUSED_DELISTED_PATH.exists() else None
    return engine_inputs(
        io.read_table("prices"), io.read_table("status"), io.read_table("index"), io.read_table("delisted"), reused
    )


def cost_rates(cfg: Mapping[str, Any], *, stress: bool = False):
    """부록 1 costs (4.8절). 기본 실행은 배율 1이고, stress=True이면 stress_multiplier를 쓴다."""
    from src.backtest.accounting import CostRates

    c = cfg["costs"]
    _expect(float(c["cash_return"]), 0.0, "costs.cash_return")  # M5는 현금에 수익을 주지 않는다
    return CostRates(
        sell_tax=float(c["sell_tax"]),
        commission_per_side=float(c["commission_per_side"]),
        slippage_per_side=float(c["slippage_per_side"]),
        stress_multiplier=float(c["stress_multiplier"]) if stress else 1.0,
    )


def build_engine(
    inputs: EngineInputs,
    cfg: Mapping[str, Any],
    *,
    stress: bool = False,
    cooldowns: Mapping[str, frozenset[str]] | None = None,
    event_liquidations: Mapping[str, frozenset[str]] | None = None,
    record_ledger: bool = False,
):
    """C의 ExecutionEngine을 새로 만든다. 체결 횟수 집계가 인스턴스에 쌓이므로 실행마다 새로 만든다.

    cooldowns, event_liquidations: 적용할 거래일(YYYY-MM-DD) → 종목코드 집합 (M9, PM). 없으면 빈 값이다.
    """
    from src.backtest.engine import ExecutionEngine

    _expect(cfg["calendar"]["execution"], "next_trading_day_close", "calendar.execution")
    e = cfg["execution"]
    _expect(float(e["delisted_without_price_value"]), 0.0, "execution.delisted_without_price_value")
    engine = ExecutionEngine(
        inputs.closes,
        inputs.bm_close,
        cost_rates(cfg, stress=stress),
        halted=inputs.halted,
        event_liquidations=event_liquidations,
        cooldowns=cooldowns,
        delistings=inputs.delistings,
        limit_move_abs_return=float(e["limit_move_abs_return"]),
        record_ledger=record_ledger,
    )
    # DEC-28: 엔진은 폐지 뒤 가격 공백 다음에 다시 나타나는 가격을 새 회사로 본다. 그렇게 인식되지 않은
    # 재사용 코드가 있으면 새 회사를 살 수 없으므로 멈춘다
    for ticker, (day, _) in inputs.delistings.items():
        after = inputs.closes[ticker].loc[pd.Timestamp(day) :].iloc[1:]
        if after.notna().any() and ticker not in engine.relisting_index:
            raise ValueError(f"{ticker}: 폐지 뒤 가격이 다시 있는데 엔진이 새 상장으로 인식하지 않는다 (DEC-28)")
    return engine
