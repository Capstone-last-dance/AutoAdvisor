"""4.6절의 종목 라벨을 메모리에서 계산한다 (C-02).

실제 파일 저장은 D 소유 ``src.common.io``가 연결된 뒤 수행한다. 이 모듈은
라벨 계산에 필요한 KRX 거래일과 사전등록 값을 인자로 받아 시점을 숨기지 않는다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.common.guards import DataAccessPolicy, assert_outcome_dates_allowed

LABEL_COLUMNS = ("date", "ticker", "h", "ret", "er", "y_rank", "y_crash")


@dataclass(frozen=True)
class LabelRules:
    horizons_days: tuple[int, ...]
    main_horizon_days: int
    crash_excess_threshold: float

    def __post_init__(self) -> None:
        if (
            not self.horizons_days
            or len(set(self.horizons_days)) != len(self.horizons_days)
            or any(type(h) is not int or h < 1 for h in self.horizons_days)
        ):
            raise ValueError("예측 기간은 중복 없는 양의 거래일 수여야 합니다")
        if self.main_horizon_days not in self.horizons_days:
            raise ValueError("주 예측 기간이 horizons_days에 없습니다")
        if not isfinite(self.crash_excess_threshold):
            raise ValueError("급락 초과수익 기준은 유한해야 합니다")


@dataclass(frozen=True)
class LabelBuildResult:
    labels: pd.DataFrame
    missing_counts: dict[str, int]
    # 실제 종료일은 재개 지연·상장폐지 후 purge 검토용이며 공개 스키마에 넣지 않는다.
    end_dates: pd.DataFrame


def publish_labels(
    result: LabelBuildResult, *, policy: DataAccessPolicy, allow_holdout: bool,
    run_date: date, stub: bool = False,
) -> Path:
    """C 라벨을 홀드아웃 가드와 D 공용 I/O를 거쳐 저장한다.

    실제 라벨의 생성·공개에 쓰는 유일한 파일 경로다. 빈 표도 공용
    스키마를 검증하되, 입력 데이터가 없는 상황에서 덮어쓰지 않는다.
    """

    from src.common import io

    if result.labels.empty:
        raise ValueError("빈 라벨은 공개 저장하지 않습니다")
    assert_outcome_dates_allowed(
        result.labels["date"].dt.date,
        policy=policy, allow_holdout=allow_holdout, run_date=run_date,
    )
    return io.write_table("labels", result.labels, stub=stub)


def _day(value: object) -> date:
    if type(value) is date:
        return value
    if isinstance(value, str):
        if len(value) != 10 or value[4] != "-" or value[7] != "-":
            raise ValueError("날짜는 YYYY-MM-DD 형식이어야 합니다")
        return date.fromisoformat(value)
    if isinstance(value, pd.Timestamp) and not pd.isna(value):
        if value.tzinfo is not None:
            raise ValueError("KRX 날짜에 시간대가 있는 시각을 쓸 수 없습니다")
        return value.date()
    raise TypeError("KRX 거래일은 date, YYYY-MM-DD 또는 Timestamp여야 합니다")


def _frame(frame: pd.DataFrame, columns: set[str], name: str) -> pd.DataFrame:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name}의 필수 열이 없습니다: {sorted(missing)}")
    copy = frame.copy()
    if "date" in columns:
        copy["date"] = copy["date"].map(_day)
    if "delist_date" in columns:
        copy["delist_date"] = copy["delist_date"].map(
            lambda value: None if pd.isna(value) else _day(value)
        )
    if "ticker" in columns:
        if (
            copy["ticker"].isna().any()
            or not copy["ticker"]
            .map(lambda value: isinstance(value, str) and len(value) == 6)
            .all()
        ):
            raise ValueError(f"{name}.ticker는 6자리 문자열이어야 합니다")
    return copy


def _unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}에 중복 키가 있습니다: {keys}")


def build_labels(
    *,
    prices: pd.DataFrame,
    index: pd.DataFrame,
    universe: pd.DataFrame,
    status: pd.DataFrame,
    delisted: pd.DataFrame,
    trading_days: Sequence[date | str | pd.Timestamp],
    signal_dates: Iterable[date | str | pd.Timestamp],
    rules: LabelRules,
    policy: DataAccessPolicy,
    allow_holdout: bool,
    run_date: date,
) -> LabelBuildResult:
    """유효한 U_t 종목의 ``ret``, ``er``, ``y_rank``, ``y_crash``를 계산한다.

    결측 라벨을 0으로 채우지 않는다. ``end_dates``는 DEC-09의 실제 종료일
    purge 합의에 쓰는 내부 진단 자료다. 이 함수는 파일을 쓰지 않는다.
    """

    days = tuple(_day(day) for day in trading_days)
    if not days or tuple(sorted(set(days))) != days:
        raise ValueError("trading_days는 중복 없는 오름차순 KRX 거래일이어야 합니다")
    selected = tuple(sorted({_day(day) for day in signal_dates}))
    assert_outcome_dates_allowed(
        selected, policy=policy, allow_holdout=allow_holdout, run_date=run_date
    )
    day_position = {day: number for number, day in enumerate(days)}
    if any(day not in day_position for day in selected):
        raise ValueError("신호일이 KRX 거래일 목록에 없습니다")

    price_frame = _frame(prices, {"date", "ticker", "close"}, "prices")
    index_frame = _frame(index, {"date", "index_code", "close"}, "index")
    universe_frame = _frame(universe, {"date", "ticker", "in_universe"}, "universe")
    status_frame = _frame(status, {"date", "ticker", "is_halted"}, "status")
    delisted_frame = _frame(
        delisted, {"ticker", "delist_date", "last_price"}, "delisted"
    )
    for name, frame, keys in (
        ("prices", price_frame, ["date", "ticker"]),
        ("index", index_frame, ["date", "index_code"]),
        ("universe", universe_frame, ["date", "ticker"]),
        ("status", status_frame, ["date", "ticker"]),
        ("delisted", delisted_frame, ["ticker"]),
    ):
        _unique(frame, keys, name)

    benchmark_frame = index_frame[index_frame["index_code"] == "KOSPI200"]
    benchmark = dict(zip(benchmark_frame["date"], benchmark_frame["close"]))
    if any(day not in benchmark for day in days):
        raise ValueError("거래일 목록에 대응하는 KOSPI200 가격이 없습니다")
    if any(pd.isna(value) or float(value) <= 0 for value in benchmark.values()):
        raise ValueError("KOSPI200 종가는 양수여야 합니다")

    closes = {(row.date, row.ticker): row.close for row in price_frame.itertuples()}
    halts = {(row.date, row.ticker): row.is_halted for row in status_frame.itertuples()}
    if not pd.api.types.is_bool_dtype(status_frame["is_halted"]):
        raise ValueError("status.is_halted는 bool이어야 합니다")
    if any(pd.isna(value) for value in halts.values()):
        raise ValueError("status.is_halted에 결측이 있습니다")
    if not pd.api.types.is_bool_dtype(universe_frame["in_universe"]):
        raise ValueError("universe.in_universe는 bool이어야 합니다")
    if universe_frame["in_universe"].isna().any():
        raise ValueError("universe.in_universe에 결측이 있습니다")
    exits = {
        row.ticker: (row.delist_date, row.last_price)
        for row in delisted_frame.itertuples()
    }
    # DEC-28의 두 재사용 종목코드는 폐지 뒤 가격 공백 다음에 신회사가
    # 상장한다. 구회사 폐지 기록을 신회사 라벨에 적용하지 않는다.
    relisted_at = {}
    for ticker in ("101970", "036220"):
        exit_info = exits.get(ticker)
        if exit_info is not None and exit_info[0] is not None:
            later = price_frame.loc[
                (price_frame["ticker"] == ticker)
                & (price_frame["date"] >= exit_info[0]), "date"
            ]
            if not later.empty:
                relisted_at[ticker] = later.min()
    members = {
        day: tuple(frame.loc[frame["in_universe"], "ticker"])
        for day, frame in universe_frame.groupby("date", sort=False)
    }

    records: list[dict] = []
    end_records: list[dict] = []
    missing: Counter[str] = Counter()

    for signal_day in selected:
        tickers = members.get(signal_day)
        if not tickers:
            raise ValueError(f"{signal_day}의 U_t 종목이 없습니다")
        entry_position = day_position[signal_day] + 1
        if entry_position >= len(days):
            missing["entry_day_not_available"] += len(tickers) * len(
                rules.horizons_days
            )
            continue
        entry_day = days[entry_position]
        for ticker in tickers:
            exit_info = exits.get(ticker)
            if ticker in relisted_at and entry_day >= relisted_at[ticker]:
                exit_info = None
            if exit_info and exit_info[0] is not None and exit_info[0] <= entry_day:
                missing["delisted_before_entry"] += len(rules.horizons_days)
                continue
            entry_halt = halts.get((entry_day, ticker))
            if entry_halt is None:
                raise ValueError(f"{ticker} {entry_day} 거래정지 상태가 없습니다")
            if entry_halt:
                missing["entry_halted"] += len(rules.horizons_days)
                continue
            entry_close = closes.get((entry_day, ticker))
            if entry_close is None or pd.isna(entry_close) or entry_close <= 0:
                raise ValueError(f"{ticker} {entry_day} 진입 수정종가가 없습니다")

            for horizon in rules.horizons_days:
                nominal_position = entry_position + horizon
                if nominal_position >= len(days):
                    missing["label_end_not_available"] += 1
                    continue
                nominal_end = days[nominal_position]
                if nominal_end > run_date:
                    missing["label_not_mature"] += 1
                    continue
                actual_end = nominal_end
                if (
                    exit_info
                    and exit_info[0] is not None
                    and exit_info[0] <= nominal_end
                ):
                    last_price = exit_info[1]
                    end_close = 0.0 if pd.isna(last_price) else float(last_price)
                    actual_end = exit_info[0]
                    if end_close < 0:
                        raise ValueError("상장폐지 정리매매 가격은 음수일 수 없습니다")
                else:
                    end_close = None
                    for day in days[nominal_position:]:
                        if day > run_date:
                            break
                        if (
                            exit_info
                            and exit_info[0] is not None
                            and exit_info[0] <= day
                        ):
                            last_price = exit_info[1]
                            end_close = (
                                0.0 if pd.isna(last_price) else float(last_price)
                            )
                            actual_end = exit_info[0]
                            break
                        is_halted = halts.get((day, ticker))
                        if is_halted is None:
                            raise ValueError(f"{ticker} {day} 거래정지 상태가 없습니다")
                        if not is_halted:
                            end_close = closes.get((day, ticker))
                            if (
                                end_close is None
                                or pd.isna(end_close)
                                or end_close <= 0
                            ):
                                raise ValueError(
                                    f"{ticker} {day} 종료 수정종가가 없습니다"
                                )
                            actual_end = day
                            break
                    if end_close is None:
                        missing["no_resumption_yet"] += 1
                        continue
                    if pd.isna(end_close) or end_close < 0:
                        raise ValueError(
                            f"{ticker} {actual_end} 종료 수정종가가 잘못됐습니다"
                        )

                stock_return = float(end_close / entry_close - 1.0)
                bm_return = float(benchmark[nominal_end] / benchmark[entry_day] - 1.0)
                excess = stock_return - bm_return
                records.append(
                    {
                        "date": signal_day,
                        "ticker": ticker,
                        "h": horizon,
                        "ret": stock_return,
                        "er": excess,
                        "y_crash": (
                            int(excess <= rules.crash_excess_threshold)
                            if horizon == rules.main_horizon_days
                            else pd.NA
                        ),
                    }
                )
                end_records.append(
                    {
                        "date": signal_day,
                        "ticker": ticker,
                        "h": horizon,
                        "nominal_end": nominal_end,
                        "actual_end": actual_end,
                    }
                )

    labels = pd.DataFrame.from_records(
        records, columns=[name for name in LABEL_COLUMNS if name != "y_rank"]
    )
    if not labels.empty:
        labels["y_rank"] = labels.groupby(["date", "h"])["er"].rank(
            pct=True, method="average"
        )
    else:
        labels["y_rank"] = pd.Series(dtype="float64")
    labels = labels[list(LABEL_COLUMNS)]
    labels["date"] = pd.to_datetime(labels["date"])
    labels["h"] = labels["h"].astype("int8")
    labels["ret"] = labels["ret"].astype("float64")
    labels["er"] = labels["er"].astype("float64")
    labels["y_rank"] = labels["y_rank"].astype("float64")
    labels["y_crash"] = labels["y_crash"].astype("Int8")
    labels = labels.sort_values(["date", "h", "ticker"]).reset_index(drop=True)
    return LabelBuildResult(
        labels=labels,
        missing_counts=dict(missing),
        end_dates=pd.DataFrame.from_records(
            end_records, columns=["date", "ticker", "h", "nominal_end", "actual_end"]
        ),
    )
