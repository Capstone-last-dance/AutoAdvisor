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


def write(name: str, df: pd.DataFrame, force_stub: bool = False) -> Path:
    if IO_AVAILABLE and not force_stub:
        _io.validate(name, df)
        return _io.write_table(name, df)
    local_validate(name, df)
    path = ROOT / "_stub" / "data" / "processed" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
