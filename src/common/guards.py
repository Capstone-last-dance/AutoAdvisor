"""C 소유의 학습·홀드아웃 시점 보호 함수.

호출자가 사전등록 설정의 날짜를 전달한다. 파일을 읽거나 현재 시각을
직접 조회하지 않아 학습, 라벨 생성, 평가 실행에서 같은 규칙을 쓸 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Sequence


class GuardViolation(ValueError):
    """사전등록된 데이터 접근 시점을 어긴 경우."""


@dataclass(frozen=True)
class DataAccessPolicy:
    latest_training_signal_date: date
    holdout_start: date
    holdout_end: date
    holdout_open_date: date

    def __post_init__(self) -> None:
        dates = (
            self.latest_training_signal_date,
            self.holdout_start,
            self.holdout_end,
            self.holdout_open_date,
        )
        if any(type(value) is not date for value in dates):
            raise TypeError("정책 날짜는 datetime이 아닌 date여야 합니다")
        if not self.latest_training_signal_date < self.holdout_start:
            raise ValueError("학습 종료일은 홀드아웃 시작 전이어야 합니다")
        if self.holdout_end < self.holdout_start:
            raise ValueError("홀드아웃 종료일이 시작일보다 빠릅니다")


def _date(value: date | str) -> date:
    if type(value) is date:
        return value
    if isinstance(value, str):
        if len(value) != 10 or value[4] != "-" or value[7] != "-":
            raise ValueError("신호일은 YYYY-MM-DD 형식이어야 합니다")
        return date.fromisoformat(value)
    raise TypeError("신호일은 date 또는 YYYY-MM-DD 문자열이어야 합니다")


def assert_training_dates_allowed(
    signal_dates: Iterable[date | str], *, policy: DataAccessPolicy
) -> None:
    """2025년 말 이후 신호일이 학습·튜닝에 들어오지 않게 한다 (4.7절)."""

    for raw in signal_dates:
        signal_date = _date(raw)
        if signal_date > policy.latest_training_signal_date:
            raise GuardViolation(
                f"학습·튜닝에 허용되지 않는 신호일: {signal_date.isoformat()}"
            )


def assert_outcome_dates_allowed(
    signal_dates: Iterable[date | str],
    *,
    policy: DataAccessPolicy,
    allow_holdout: bool,
    run_date: date,
) -> None:
    """홀드아웃 신호일의 라벨·성과 계산을 개봉 전까지 차단한다 (4.7절)."""

    if type(run_date) is not date:
        raise TypeError("실행일은 KST 기준 date여야 합니다")
    if type(allow_holdout) is not bool:
        raise TypeError("allow_holdout은 bool이어야 합니다")
    for raw in signal_dates:
        signal_date = _date(raw)
        if policy.holdout_start <= signal_date <= policy.holdout_end:
            if not allow_holdout or run_date < policy.holdout_open_date:
                raise GuardViolation(
                    "홀드아웃 라벨·성과는 허용 플래그와 개봉일 이후 실행이 "
                    f"모두 필요합니다: {signal_date.isoformat()}"
                )


def purge_signal_dates(
    signal_dates: Iterable[date | str],
    *,
    trading_days: Sequence[date | str],
    h: int,
    next_segment_first_signal: date | str,
    actual_end_dates: Mapping[date | str, date | str] | None = None,
) -> tuple[date, ...]:
    """명목 종료일 또는 지연된 실제 종료일이 다음 구간에 닿는 신호일을 제거한다.

    actual_end_dates는 신호일별 모든 라벨의 최종 실제 종료일이다. 라벨을
    이용하는 실행에서는 C의 end_dates에서 최댓값을 만들어 전달하고,
    A·B·C가 반환된 같은 신호일 집합을 사용한다.
    """

    if type(h) is not int or h < 1:
        raise ValueError("h는 양의 거래일 수여야 합니다")
    days = tuple(_date(value) for value in trading_days)
    if not days or tuple(sorted(set(days))) != days:
        raise ValueError("trading_days는 중복 없는 오름차순이어야 합니다")
    first_next = _date(next_segment_first_signal)
    if first_next not in days:
        raise ValueError("다음 구간 첫 신호일이 거래일 목록에 없습니다")
    day_index = {day: index for index, day in enumerate(days)}
    actual_ends: dict[date, date] = {}
    if actual_end_dates is not None:
        for raw_signal, raw_end in actual_end_dates.items():
            signal_day, end_day = _date(raw_signal), _date(raw_end)
            if signal_day not in day_index or end_day not in day_index:
                raise ValueError("실제 종료일과 신호일은 거래일 목록에 있어야 합니다")
            if end_day <= signal_day:
                raise ValueError("실제 종료일은 신호일보다 늦어야 합니다")
            actual_ends[signal_day] = end_day
    kept: list[date] = []
    for raw in signal_dates:
        signal_day = _date(raw)
        if signal_day not in day_index:
            raise ValueError(f"신호일이 거래일 목록에 없습니다: {signal_day}")
        end_index = day_index[signal_day] + 1 + h
        if (
            signal_day < first_next
            and end_index < len(days)
            and days[end_index] < first_next
            and actual_ends.get(signal_day, days[end_index]) < first_next
        ):
            kept.append(signal_day)
    return tuple(sorted(set(kept)))
