"""모듈 간 입출력 (M11, 명세서 6.12절, 9.3절). D 소유.

모든 모듈은 9.3절의 파일을 이 모듈의 함수로만 읽고 쓴다 (R-18).
읽기와 쓰기 모두 9.3절 스키마로 검증하며, 어기면 SchemaError를 낸다.

경로 규칙 (6.12절)
- date가 있으면 {폴더}/{date}.parquet
- run_id가 있으면 {폴더}/wf_{run_id}.parquet
- 둘 다 없으면 data/processed/{name}.parquet (forward_log는 루트의 forward_log.csv)
- stub=True이면 같은 상대 경로를 _stub/ 아래에 쓰고, 읽을 때도 _stub/ 아래에서 읽는다
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 저장소 루트. 테스트에서는 임시 폴더로 바꿔 쓴다.
ROOT = Path(__file__).resolve().parents[2]

STUB_DIR = "_stub"

# 2.3절 고지 문구. evidence의 disclaimer는 이 문구와 같아야 한다 (R-12).
DISCLAIMER = (
    "본 정보는 정보 제공 목적이며 특정 종목의 매수·매도를 권유하지 않습니다. "
    "투자 판단과 그 결과의 책임은 투자자 본인에게 있습니다."
)

_KST_OFFSET = pd.Timedelta(hours=9)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RUN_ID_RE = re.compile(r"^\d{8}_(PM|A|B|C|D)_[A-Za-z0-9_\-]+$")  # 9.3절 run_id 형식
# 0.4절: 6자리 문자열. 2025년 하반기 신규 상장부터 영문이 섞인다 (예: "0030R0")
_TICKER_RE = r"[0-9A-Z]{6}"


class SchemaError(ValueError):
    """9.3절 스키마를 어긴 입력."""


@dataclass(frozen=True)
class Col:
    """열 하나의 규칙.

    kind: date, datetime_kst, str, ticker, ticker_or_cash, float64, int64, int32, int8,
          nullable_int8, bool, nullable_bool, number
    """

    kind: str
    nullable: bool = False
    allowed: frozenset | None = None
    bounds: tuple[float, float] | None = None


def _c(kind: str, nullable: bool = False, allowed=None, bounds=None) -> Col:
    return Col(kind, nullable, frozenset(allowed) if allowed is not None else None, bounds)


# ---------------------------------------------------------------------------
# 9.3절 스키마
# ---------------------------------------------------------------------------

SCHEMAS: dict[str, dict[str, Col]] = {
    "prices": {
        "date": _c("date"),
        "ticker": _c("ticker"),
        "market": _c("str", allowed={"KOSPI", "KOSDAQ"}),
        "close": _c("float64"),
        "close_raw": _c("float64"),
        "volume": _c("int64"),
        "value": _c("int64"),
        "mcap": _c("int64"),
        "listed_days": _c("int32"),
    },
    "index": {
        "date": _c("date"),
        "index_code": _c("str", allowed={"KOSPI200"}),
        "close": _c("float64"),
    },
    "tickers": {
        "ticker": _c("ticker"),
        "name": _c("str"),
        "market": _c("str"),
        "listing_date": _c("date"),
        "delisting_date": _c("date", nullable=True),
        "security_type": _c("str"),
    },
    # DEC-24: 구성종목 2열은 쓰는 곳이 없어 뺀다. is_admin은 v2 전까지만 결측 허용 (False로 채우지 않는다)
    "status": {
        "date": _c("date"),
        "ticker": _c("ticker"),
        "is_admin": _c("nullable_bool", nullable=True),
        "is_halted": _c("bool"),
    },
    "universe": {
        "date": _c("date"),
        "ticker": _c("ticker"),
        "mcap_rank": _c("int32"),
        "in_universe": _c("bool"),
    },
    "delisted": {
        "ticker": _c("ticker"),
        "delist_date": _c("date"),
        "last_price": _c("float64", nullable=True),
    },
    # DEC-28: 재사용 종목코드의 옛 회사 상장폐지 보완. delisted와 같은 열·정의
    "delisted_reused": {
        "ticker": _c("ticker"),
        "delist_date": _c("date"),
        "last_price": _c("float64", nullable=True),
    },
    "sector": {
        "ticker": _c("ticker"),
        "sector": _c("str"),
        "valid_from": _c("date"),
        "valid_to": _c("date", nullable=True),
    },
    # DEC-22: investor_flows는 원천이 없어 만들지 않는다.
    "disclosures": {
        "rcept_no": _c("str"),
        "rcept_dt": _c("datetime_kst"),
        "ticker": _c("ticker", nullable=True),  # 비상장은 null
        "corp_name": _c("str"),
        "report_nm": _c("str"),
        "report_nm_norm": _c("str"),
    },
    "news": {
        "news_id": _c("str"),
        "published_at": _c("datetime_kst"),
        "ticker": _c("ticker"),
        "title": _c("str"),
        "url": _c("str"),
        "source": _c("str", allowed={"bigkinds", "naver"}),
        "sentiment": _c("float64", nullable=True, bounds=(-1.0, 1.0)),
    },
    "labels": {
        "date": _c("date"),
        "ticker": _c("ticker"),
        "h": _c("int8", allowed={1, 5, 20}),
        "ret": _c("float64"),
        "er": _c("float64"),
        "y_rank": _c("float64"),
        "y_crash": _c("nullable_int8", nullable=True, allowed={0, 1}),  # h = 20 행만
    },
    # features는 date, ticker 외의 열을 FEATURE_COLUMNS로 따로 검증한다.
    "features": {
        "date": _c("date"),
        "ticker": _c("ticker"),
    },
    "signals": {
        "date": _c("date"),
        "ticker": _c("ticker"),
        "rank_score": _c("float64"),
        "rank": _c("int32"),
        "crash_prob": _c("float64"),
        "crash_flag": _c("int8", allowed={0, 1}),
        "model_version": _c("str"),
    },
    "positions": {
        "date": _c("date"),
        "ticker": _c("ticker_or_cash"),
        "target_weight": _c("float64"),
        "constructor": _c("str", allowed={"buffer", "rl"}),
        "exposure": _c("float64"),
        "regime_mc": _c("int8", allowed={0, 1}),
    },
    "alerts": {
        "date": _c("date"),
        "ticker": _c("ticker"),
        "trigger_type": _c("str", allowed={"T1", "T2", "T3", "T4", "T5", "T6"}),
        "rcept_no": _c("str"),
        "report_nm": _c("str"),
        "rcept_dt": _c("datetime_kst"),
    },
    "forward_log": {
        "signal_date": _c("date"),
        "commit_hash": _c("str"),
        "signal_source": _c("str", allowed={"baseline", "ml"}),
        "has_positions": _c("int8", allowed={0, 1}),
        "label_ready_date": _c("date"),
        # 사후 열은 확정 전까지 비워 둔다
        "mean_er_top20": _c("float64", nullable=True),
        "mean_er_flagged": _c("float64", nullable=True),
        "portfolio_return": _c("float64", nullable=True),
        "bm_return": _c("float64", nullable=True),
    },
}

# 6.2절 피처. 순위 변환 피처만 {피처}_na 열을 가진다.
RANK_FEATURES = (
    "mom_12_1", "mom_12_7", "rev_1m", "rev_1w", "high52_prox", "vol_20", "vol_60",
    "max_ret_20", "turnover_spike", "log_mcap", "news_cnt_spike", "news_sent_5",
)
RAW_FEATURES = (
    "penny_flag", "ss_allowed",
    "ev_earnings", "ev_buyback", "ev_cancel", "ev_dividend", "ev_rights", "ev_cbbw",
    "mkt_vol_20", "mkt_ret_20", "regime_mc", "ss_regime",
)
FEATURE_COLUMNS = frozenset(RANK_FEATURES) | frozenset(RAW_FEATURES)
NA_FLAG_COLUMNS = frozenset(f"{f}_na" for f in RANK_FEATURES)

# date 또는 run_id로 파일을 나누는 이름과 그 폴더
DATED_FOLDERS = {
    "features": "features",
    "signals": "signals",
    "positions": "positions",
    "alerts": "alerts",
}

EVIDENCE_FOLDER = "evidence"
EVIDENCE_KEYS = ("date", "ticker", "signal_type", "summary", "sources", "disclaimer")
EVIDENCE_SOURCE_KEYS = ("n", "type", "id", "title", "url", "timestamp")
EVIDENCE_SIGNAL_TYPES = frozenset({"crash", "rank", "alert"})


# ---------------------------------------------------------------------------
# 공개 함수 (6.12절)
# ---------------------------------------------------------------------------

def read_table(name: str, date: str | None = None, run_id: str | None = None,
               stub: bool = False) -> pd.DataFrame:
    # stub=True는 _stub/ 아래 스텁을 읽는다. 6.12절 시그니처에 추가한 인자 (decisions.md 기록 대상)
    # 파일이 없으면 FileNotFoundError만 낸다. 호출하는 쪽은 이것으로 '새 파일'을 판단한다.
    rel = _relative_path(name, date, run_id)
    path = ROOT / (Path(STUB_DIR) / rel if stub else rel)
    _require_file(path)
    if name == "forward_log":
        df = _read_forward_log(path)
    else:
        df = pd.read_parquet(path)
    df = _normalize_dates(name, df)
    validate(name, df)
    return df


def write_table(name: str, df: pd.DataFrame, date: str | None = None,
                run_id: str | None = None, stub: bool = False) -> Path:
    rel = _relative_path(name, date, run_id)
    validate(name, df)
    if date is not None:
        _check_rows_match_date(df, date)
    path = ROOT / (Path(STUB_DIR) / rel if stub else rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "forward_log":
        df.to_csv(path, index=False, date_format="%Y-%m-%d", encoding="utf-8")
    else:
        df.to_parquet(path, index=False)
    return path


def validate(name: str, df: pd.DataFrame) -> None:
    """9.3절 스키마를 어기면 SchemaError. 문제를 모두 모아 한 번에 알린다."""
    if name not in SCHEMAS:
        raise ValueError(f"알 수 없는 테이블 이름: {name!r}")
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"{name}: DataFrame이 아니다 ({type(df).__name__})")

    schema = SCHEMAS[name]
    errors: list[str] = []

    missing = [c for c in schema if c not in df.columns]
    if missing:
        errors.append(f"없는 열: {missing}")

    extra = [c for c in df.columns if c not in schema]
    if name == "features":
        errors.extend(_check_feature_columns(df, extra))
    elif extra:
        errors.append(f"스키마에 없는 열: {extra} (열 추가는 생산자·소비자 합의 후 decisions.md 기록, 9.3절)")

    for col, rule in schema.items():
        if col in df.columns:
            errors.extend(f"{col}: {msg}" for msg in _check_column(df[col], rule))

    if name == "labels" and not missing:
        errors.extend(_check_labels_rows(df))

    if errors:
        raise SchemaError(f"{name} 스키마 위반\n- " + "\n- ".join(errors))


def read_evidence(date: str, stub: bool = False) -> list[dict]:
    rel = _evidence_path(date)
    path = ROOT / (Path(STUB_DIR) / rel if stub else rel)
    _require_file(path)
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    _validate_evidence(records, date)
    return records


def write_evidence(date: str, records: list[dict], stub: bool = False) -> Path:
    rel = _evidence_path(date)
    _validate_evidence(records, date)
    path = ROOT / (Path(STUB_DIR) / rel if stub else rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------

def _relative_path(name: str, date: str | None, run_id: str | None) -> Path:
    if name not in SCHEMAS:
        raise ValueError(f"알 수 없는 테이블 이름: {name!r}")
    if date is not None and run_id is not None:
        raise ValueError("date와 run_id는 함께 줄 수 없다")

    if name == "forward_log":
        if date is not None or run_id is not None:
            raise ValueError("forward_log는 date와 run_id를 받지 않는다")
        return Path("forward_log.csv")

    if date is not None or run_id is not None:
        folder = DATED_FOLDERS.get(name)
        if folder is None:
            raise ValueError(f"{name}은 data/processed/의 단일 파일이라 date나 run_id를 받지 않는다")
        if date is not None:
            _check_date_arg(date)
            return Path(folder) / f"{date}.parquet"
        _check_run_id_arg(run_id)
        return Path(folder) / f"wf_{run_id}.parquet"

    if name in DATED_FOLDERS:
        raise ValueError(f"{name}은 date(날짜별 파일) 또는 run_id(백테스트 파일)가 필요하다")
    return Path("data") / "processed" / f"{name}.parquet"


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"파일이 없다: {path}")


def _evidence_path(date: str) -> Path:
    _check_date_arg(date)
    return Path(EVIDENCE_FOLDER) / f"{date}.jsonl"


def _check_date_arg(date: str) -> None:
    if not isinstance(date, str) or not _DATE_RE.match(date):
        raise ValueError(f"date는 'YYYY-MM-DD' 문자열이어야 한다: {date!r}")
    try:
        dt.date.fromisoformat(date)
    except ValueError as e:
        raise ValueError(f"존재하지 않는 날짜: {date!r}") from e


def _check_run_id_arg(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        raise ValueError(f"run_id는 '{{YYYYMMDD}}_{{역할ID}}_{{짧은설명}}' 형식이어야 한다: {run_id!r}")


# ---------------------------------------------------------------------------
# 읽기 보조
# ---------------------------------------------------------------------------

def _read_forward_log(path: Path) -> pd.DataFrame:
    schema = SCHEMAS["forward_log"]
    if path.stat().st_size == 0:
        # 첫 기록 전의 빈 파일
        return pd.DataFrame({
            "signal_date": pd.Series(dtype="datetime64[ns]"),
            "commit_hash": pd.Series(dtype="str"),
            "signal_source": pd.Series(dtype="str"),
            "has_positions": pd.Series(dtype="int8"),
            "label_ready_date": pd.Series(dtype="datetime64[ns]"),
            **{c: pd.Series(dtype="float64") for c in list(schema)[5:]},
        })
    df = pd.read_csv(path, dtype={"commit_hash": "str", "signal_source": "str"}, encoding="utf-8")
    for col in ("signal_date", "label_ready_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")
    if "has_positions" in df.columns and df["has_positions"].notna().all():
        df["has_positions"] = df["has_positions"].astype("int8")
    for col in list(schema)[5:]:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    return df


def _normalize_dates(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """parquet에서 datetime.date 객체로 읽힌 date 열을 datetime64로 맞춘다."""
    for col, rule in SCHEMAS[name].items():
        if rule.kind == "date" and col in df.columns and df[col].dtype == object:
            values = df[col].dropna()
            if len(values) and all(isinstance(v, dt.date) and not isinstance(v, dt.datetime) for v in values):
                df[col] = pd.to_datetime(df[col])
    return df


# ---------------------------------------------------------------------------
# 검증 보조
# ---------------------------------------------------------------------------

def _check_column(s: pd.Series, rule: Col) -> list[str]:
    errors: list[str] = []
    nulls = s.isna()
    if nulls.any() and not rule.nullable:
        errors.append(f"결측 {int(nulls.sum())}건 (결측 허용 안 함)")
    values = s[~nulls]

    kind = rule.kind
    if kind == "date":
        errors.extend(_check_date_series(s, values))
    elif kind == "datetime_kst":
        errors.extend(_check_kst_series(s, values))
    elif kind in ("str", "ticker", "ticker_or_cash"):
        if not values.map(lambda v: isinstance(v, str)).all():
            errors.append("문자열이 아닌 값이 있다")
        elif kind != "str" and len(values):
            ok = values.str.fullmatch(_TICKER_RE)
            if kind == "ticker_or_cash":
                ok = ok | (values == "CASH")
            if not ok.all():
                bad = values[~ok].head(3).tolist()
                errors.append(f"6자리 종목코드 문자열(숫자·영문 대문자)이 아니다 (0.4절), 예: {bad}")
    elif kind in ("float64", "int64", "int32", "int8", "bool"):
        if s.dtype != np.dtype(kind):
            errors.append(f"자료형이 {kind}이 아니다 ({s.dtype})")
    elif kind == "nullable_bool":
        if s.dtype != np.dtype("bool") and s.dtype != pd.BooleanDtype():
            errors.append(f"자료형이 bool(결측 허용 시 boolean)이 아니다 ({s.dtype})")
    elif kind == "nullable_int8":
        if s.dtype != np.dtype("int8") and s.dtype != pd.Int8Dtype():
            errors.append(f"자료형이 int8(결측 허용 시 Int8)이 아니다 ({s.dtype})")
    elif kind == "number":
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            errors.append(f"숫자 자료형이 아니다 ({s.dtype})")
    else:
        raise AssertionError(f"알 수 없는 kind: {kind}")

    if errors:
        return errors

    if rule.allowed is not None and len(values):
        bad = sorted(set(values.tolist()) - rule.allowed, key=str)
        if bad:
            errors.append(f"허용되지 않은 값: {bad[:5]} (허용: {sorted(rule.allowed, key=str)})")
    if rule.bounds is not None and len(values):
        lo, hi = rule.bounds
        if ((values < lo) | (values > hi)).any():
            errors.append(f"범위 {lo}~{hi}를 벗어난 값이 있다")
    return errors


def _check_date_series(s: pd.Series, values: pd.Series) -> list[str]:
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return ["date 열에 시간대가 있다 (날짜만 저장한다)"]
    if pd.api.types.is_datetime64_dtype(s):
        if len(values) and (values != values.dt.normalize()).any():
            return ["date 열에 시각이 들어 있다 (YYYY-MM-DD 날짜만, 0.4절)"]
        return []
    if s.dtype == object and all(isinstance(v, dt.date) and not isinstance(v, dt.datetime) for v in values):
        return []
    return [f"날짜 자료형이 아니다 ({s.dtype}). 문자열 날짜는 pd.to_datetime으로 바꿔서 넣는다"]


def _check_kst_series(s: pd.Series, values: pd.Series) -> list[str]:
    if not isinstance(s.dtype, pd.DatetimeTZDtype):
        return [f"시간대가 있는 datetime이 아니다 ({s.dtype}). KST(+09:00)로 넣는다 (0.4절)"]
    if len(values):
        offset = values.dt.tz_localize(None) - values.dt.tz_convert("UTC").dt.tz_localize(None)
        if (offset != _KST_OFFSET).any():
            return [f"KST(+09:00)가 아닌 시각이 있다 (시간대: {s.dt.tz})"]
    return []


def _check_feature_columns(df: pd.DataFrame, extra: list[str]) -> list[str]:
    errors: list[str] = []
    unknown = [c for c in extra if c not in FEATURE_COLUMNS and c not in NA_FLAG_COLUMNS]
    if unknown:
        errors.append(f"6.2절에 없는 피처 열: {unknown}")
    for col in extra:
        if col in unknown:
            continue
        errors.extend(f"{col}: {msg}" for msg in _check_column(df[col], Col("number", nullable=True)))
        if col in NA_FLAG_COLUMNS:
            if df[col].isna().any() or not df[col].isin([0, 1]).all():
                errors.append(f"{col}: 0 또는 1만 들어가야 한다 (6.2절)")
    return errors


def _check_labels_rows(df: pd.DataFrame) -> list[str]:
    """y_crash는 h = 20 행에만 있다 (9.3절)."""
    errors: list[str] = []
    is20 = df["h"] == 20
    if df.loc[is20, "y_crash"].isna().any():
        errors.append("y_crash: h = 20 행에 결측이 있다")
    if df.loc[~is20, "y_crash"].notna().any():
        errors.append("y_crash: h = 20이 아닌 행에 값이 있다 (h = 20 행만)")
    return errors


def _check_rows_match_date(df: pd.DataFrame, date: str) -> None:
    if "date" not in df.columns or df.empty:
        return
    day = pd.Timestamp(date)
    if not (pd.to_datetime(df["date"]) == day).all():
        raise SchemaError(f"date={date} 파일에 다른 날짜의 행이 있다")


def _validate_evidence(records: list[dict], date: str) -> None:
    """6.7절 evidence 형식."""
    if not isinstance(records, list):
        raise SchemaError("evidence: records는 dict의 list여야 한다")
    errors: list[str] = []
    for i, rec in enumerate(records):
        where = f"evidence[{i}]"
        if not isinstance(rec, dict):
            errors.append(f"{where}: dict가 아니다")
            continue
        keys = set(rec)
        if keys != set(EVIDENCE_KEYS):
            errors.append(f"{where}: 키가 다르다 (없음 {sorted(set(EVIDENCE_KEYS) - keys)}, "
                          f"초과 {sorted(keys - set(EVIDENCE_KEYS))})")
            continue
        if rec["date"] != date:
            errors.append(f"{where}: date {rec['date']!r}가 파일 날짜 {date!r}와 다르다")
        if not isinstance(rec["ticker"], str) or not re.fullmatch(_TICKER_RE, rec["ticker"]):
            errors.append(f"{where}: ticker가 6자리 문자열이 아니다")
        if rec["signal_type"] not in EVIDENCE_SIGNAL_TYPES:
            errors.append(f"{where}: signal_type {rec['signal_type']!r}는 crash, rank, alert 중 하나여야 한다")
        if not isinstance(rec["summary"], str):
            errors.append(f"{where}: summary가 문자열이 아니다")
        if rec["disclaimer"] != DISCLAIMER:
            errors.append(f"{where}: disclaimer가 2.3절 고지 문구와 다르다 (R-12)")
        if not isinstance(rec["sources"], list):
            errors.append(f"{where}: sources가 list가 아니다")
            continue
        for j, src in enumerate(rec["sources"]):
            swhere = f"{where}.sources[{j}]"
            if not isinstance(src, dict) or set(src) != set(EVIDENCE_SOURCE_KEYS):
                errors.append(f"{swhere}: 키가 {list(EVIDENCE_SOURCE_KEYS)}와 다르다")
                continue
            if not isinstance(src["n"], int) or isinstance(src["n"], bool):
                errors.append(f"{swhere}: n이 정수가 아니다")
            try:
                ts = dt.datetime.fromisoformat(src["timestamp"])
            except (TypeError, ValueError):
                errors.append(f"{swhere}: timestamp가 ISO 8601이 아니다")
            else:
                if ts.utcoffset() != dt.timedelta(hours=9):
                    errors.append(f"{swhere}: timestamp가 KST(+09:00)가 아니다 (0.4절)")
    if errors:
        raise SchemaError("evidence 형식 위반 (6.7절)\n- " + "\n- ".join(errors))
