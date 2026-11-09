"""src/common/io.py(D 소유, D-02)가 있으면 그 함수로, 없으면 _stub/ 아래에 쓴다 (R-17, R-18, 6.12절).
스텁으로 쓴 결과는 커밋·forward test·대시보드 대상이 아니다."""
from pathlib import Path

import pandas as pd

from src.data.krx_client_paths import ROOT
from src.data.schemas import validate as local_validate

try:
    from src.common import io as _io  # D-02 완료 후
except ImportError:
    _io = None

IO_AVAILABLE = _io is not None


def write(name: str, df: pd.DataFrame, force_stub: bool = False, date: str | None = None) -> Path:
    if IO_AVAILABLE and not force_stub:
        _io.validate(name, df)
        return _io.write_table(name, df, date=date, stub=False)
    local_validate(name, df)
    # 9.3절 경로 규칙: date가 있으면 {폴더}/{date}.parquet, 없으면 data/processed/{name}.parquet. stub이면 _stub/ 아래
    path = ROOT / "_stub" / (f"{name}/{date}.parquet" if date else f"data/processed/{name}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read(name: str, force_stub: bool = False) -> pd.DataFrame:
    """앞 단계가 쓴 표를 읽는다. write와 같은 경로 규칙(R-18)."""
    if IO_AVAILABLE and not force_stub:
        return _io.read_table(name)
    return pd.read_parquet(ROOT / "_stub" / "data" / "processed" / f"{name}.parquet")
