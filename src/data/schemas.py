"""9.3절 스키마 중 A가 생산하는 테이블. D의 src/common/io.py가 생기기 전까지의 임시 검증이다 (R-17, 6.12절 스텁).
io.py가 생기면 shim이 io.validate를 쓰므로 이 파일은 단위 테스트용으로만 남긴다."""
import pandas as pd

D = "datetime64[ns]"
SCHEMAS = {
    "prices": {"date": D, "ticker": "object", "market": "object", "close": "float64", "close_raw": "float64",
               "volume": "int64", "value": "int64", "mcap": "int64", "listed_days": "int32"},
    "index": {"date": D, "index_code": "object", "close": "float64"},
    "tickers": {"ticker": "object", "name": "object", "market": "object", "listing_date": D, "delisting_date": D, "security_type": "object"},
    "status": {"date": D, "ticker": "object", "is_admin": "boolean", "is_halted": "bool", "in_kospi200": "boolean", "in_kosdaq150": "boolean"},
    "universe": {"date": D, "ticker": "object", "mcap_rank": "int32", "in_universe": "bool"},
    "delisted": {"ticker": "object", "delist_date": D, "last_price": "float64"},
    "sector": {"ticker": "object", "sector": "object", "valid_from": D, "valid_to": D},
    "investor_flows": {"date": D, "ticker": "object", "retail_net": "int64", "foreign_net": "int64", "inst_net": "int64"},
}


def _dtype_ok(s: pd.Series, dt: str) -> bool:
    """pandas 2·3 모두에서 통한다: 문자열은 object 또는 str, 날짜는 해상도(ns/us)를 가리지 않는다."""
    if dt == "object":
        return pd.api.types.is_string_dtype(s.dtype) or s.dtype == object
    if dt == D:
        return pd.api.types.is_datetime64_any_dtype(s)
    return str(s.dtype) == dt


def validate(name: str, df: pd.DataFrame) -> None:
    spec = SCHEMAS[name]
    if list(df.columns) != list(spec):
        raise ValueError(f"{name}: 열이 스키마와 다르다 {list(df.columns)} != {list(spec)}")
    for col, dt in spec.items():
        if not _dtype_ok(df[col], dt):
            raise ValueError(f"{name}.{col}: 자료형 {df[col].dtype} != {dt}")
    if "ticker" in df and not df["ticker"].dropna().str.fullmatch(r"[0-9A-Z]{6}").all():
        raise ValueError(f"{name}: 6자리 문자열이 아닌 ticker가 있다")
    key = {"prices": ["date", "ticker"], "status": ["date", "ticker"], "universe": ["date", "ticker"],
           "investor_flows": ["date", "ticker"], "tickers": ["ticker"], "index": ["date"], "delisted": ["ticker"]}.get(name)
    if key and df.duplicated(key).any():
        raise ValueError(f"{name}: 키 {key} 중복 {int(df.duplicated(key).sum())}건")
