"""증권 종류 판별 (부록 3 Q-15). 값은 tickers.parquet의 security_type.

현재 상장 종목: KRX [12005] 전종목 기본정보의 SECUGRP_NM(증권구분)과 KIND_STKCERT_TP_NM(주식종류).
  실측: 보통주 2,762개는 모두 코드 끝자리가 '0', 우선주 99개는 끝자리가 5·7·9·K·L이다.
폐지 종목: FDR KRX-DELISTING의 SecuGroup, Kind. 단 Kind가 비어 있는(NaN) 종목이 있고,
  Kind='보통주'인데 코드 끝자리가 0이 아닌 종목이 있어(실측) 코드 끝자리 규칙을 쓸 수 없다. Kind가 없으면 종목명 규칙을 쓰고 출처를 기록한다.
"""
import re

import pandas as pd

PREFERRED_KINDS = {"구형우선주", "신형우선주", "종류주권"}
_PREF_NAME = re.compile(r"우[A-Z]?(\(전환\))?$|[0-9]우[A-Z]?(\(전환\))?$")

NON_STOCK_GROUPS = {
    "부동산투자회사": "reit",
    "사회간접자본투융자회사": "infra_fund",
    "투자회사": "investment_company",
    "선박투자회사": "ship_fund",
    "외국주권": "foreign_stock",
    "주식예탁증권": "dr",
    "수익증권": "beneficiary_cert",
    "신주인수권증서": "warrant_cert",
    "신주인수권증권": "warrant",
}


def classify(secu_group, kind, name) -> tuple:
    """(security_type, source). source는 'krx_field' 또는 'name_rule'."""
    if secu_group in NON_STOCK_GROUPS:
        return NON_STOCK_GROUPS[secu_group], "krx_field"
    if secu_group != "주권":
        return "other", "krx_field"
    if isinstance(kind, str) and kind:
        return ("preferred" if kind in PREFERRED_KINDS else "common"), "krx_field"
    if isinstance(name, str) and _PREF_NAME.search(name.strip()):
        return "preferred", "name_rule"
    return "common", "name_rule"


def classify_frame(df: pd.DataFrame, group_col, kind_col, name_col) -> pd.DataFrame:
    res = [classify(g, k, n) for g, k, n in zip(df[group_col], df[kind_col], df[name_col])]
    out = df.copy()
    out["security_type"] = [r[0] for r in res]
    out["security_type_source"] = [r[1] for r in res]
    return out
