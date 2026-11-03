"""`listed_days`·`build_universe` (A-02): 재상장 종목코드 구간 분리와 4.4절 유니버스 3단계 판정.

예전 pykrx 기반 수집·정제 파이프라인(수집·tickers·prices·status·delisted·sector·investor_flows 생성, CLI 진입점)은
약관 위반(자동화 수집)으로 폐기했다. 그 경로의 산출물 스키마는 DEC-24 이전 기준이라 지금 공용 I/O와도 안 맞는다.
현재 수집·정제 진입점은 `src/data/build_openapi.py`(KRX Open API)이며, 이 파일의 `build_universe`·`listed_days`만
그대로 가져다 쓴다.
"""
import numpy as np
import pandas as pd

from src.data.universe import avg_trading_value, universe_for_date

RELISTING_GAP_TRADING_DAYS = 60   # 종목코드의 거래 공백이 이 거래일수를 넘으면 재상장(다른 회사가 같은 코드를 받음)으로
                                  # 보고 구간을 나눈다. 정상적인 거래정지보다 훨씬 길고, 공백 쪽 구간은 유니버스 60일
                                  # 기준에 어차피 못 미치므로 경계값이 며칠 달라져도 영향이 없다
                                  # (실례: 101970, 036220 — C 리뷰 2026-11-03, docs/appendix3_A_results.md 참고)


def _trading_episodes(p: pd.DataFrame, pos: pd.Series, gap: int = RELISTING_GAP_TRADING_DAYS) -> tuple:
    """종목코드별로 거래일 공백이 gap을 넘으면 구간을 나눈다. 반환: (구간 시작 거래일 위치, 구간 끝 거래일 위치),
    각각 p와 같은 인덱스. 재상장으로 같은 코드를 재사용해도 옛 회사·새 회사 구간이 분리된다."""
    d = p[["ticker", "date"]].sort_values(["ticker", "date"])
    p_pos = d["date"].map(pos)
    prev_pos = p_pos.groupby(d["ticker"]).shift(1)
    new_episode = prev_pos.isna() | ((p_pos - prev_pos) > gap)
    episode_id = new_episode.groupby(d["ticker"]).cumsum()
    key = [d["ticker"], episode_id]
    start = p_pos.groupby(key).transform("min").reindex(p.index)
    end = p_pos.groupby(key).transform("max").reindex(p.index)
    return start, end


def listed_days(p: pd.DataFrame, tickers: pd.DataFrame, days: list) -> pd.Series:
    """실제 상장일부터 센 거래일 수(상장일 당일이 1). 상장일이 days의 첫날보다 앞이면 그 앞 구간은 주중 일수로 센다
    (휴장일을 몰라 약간 크게 나오지만 60일 기준 판정에는 상장일이 수집 시작 직전인 소수 종목만 영향이 있다).

    종목코드는 재사용될 수 있다(폐지 후 몇 년 뒤 다른 회사가 같은 코드로 재상장). tickers의 listing_date는 그 코드의
    '가장 최근' 회사 기준이라서, 공백을 두고 거래 구간을 나눈 뒤(_trading_episodes) 공식 상장일이 설명하는 구간(그
    구간 끝에서 gap 거래일 이내)에서만 상장일-첫 시세일 보정을 적용한다. 그보다 이전 구간(옛 회사)은 그 구간 자체의
    첫 시세일부터 센다 — 새 회사의 상장일을 옛 회사 쪽에 적용하면 음수가 돼 1로 잘리고, 반대로 옛 회사의 첫 시세일을
    새 회사 쪽에 적용하면 신규 상장을 오래된 종목으로 잘못 본다."""
    first = pd.Timestamp(days[0])
    pos = pd.Series(np.arange(len(days)), index=pd.DatetimeIndex(days))
    lst = tickers.set_index("ticker")["listing_date"]
    L = p["ticker"].map(lst)
    official_pos = pd.Series(
        np.searchsorted(np.array(days, dtype="datetime64[ns]"), L.fillna(first).values.astype("datetime64[ns]")), index=p.index)

    ep_start, ep_end = _trading_episodes(p, pos)
    # 공식 상장일이 '이 구간' 안에 있거나(소급 재분류 등) 구간 시작 바로 앞(상장일이 약간 어긋난 정상적인 경우)이면
    # 이 구간이 공식 상장일이 설명하는 구간이다. 구간 끝과 비교하면 안 된다 — 2010년 이전부터 계속 거래된 정상
    # 종목도 상장일(예: 1995년, 수집 시작 전이라 위치 0)과 구간 끝(최근)이 멀어서 잘못 걸린다
    describes_this_episode = (official_pos >= ep_start - RELISTING_GAP_TRADING_DAYS) & (official_pos <= ep_end + RELISTING_GAP_TRADING_DAYS)
    start_pos = np.where(describes_this_episode.values, np.minimum(official_pos.values, ep_start.values), ep_start.values)
    start_pos = pd.Series(start_pos, index=p.index)

    pre = pd.Series(0, index=p.index)
    before = L.notna() & (L < first)
    pre[before] = np.busday_count(L[before].values.astype("datetime64[D]"), np.datetime64(first, "D"))
    out = p["date"].map(pos) - start_pos + 1 + pre
    out[L.isna()] = 10_000                       # 상장일을 모르면 충분히 오래된 것으로 본다 (리포트에서 건수를 센다)
    return out.clip(lower=1).astype("int32")


def build_universe(prices: pd.DataFrame, tickers: pd.DataFrame, status: pd.DataFrame, sig: list, first_year: int, provisional: bool):
    vp = prices.pivot(index="date", columns="ticker", values="value")
    avg = avg_trading_value(vp)
    info = tickers.set_index("ticker")[["name", "security_type"]]
    use = {t for t in sig if t.year >= first_year}
    # 신호일 행과 상태를 한 번에 합친다 (종목마다 조회하면 신호일 수 × 종목 수만큼 느려진다)
    ps = prices[prices["date"].isin(use)][["date", "ticker", "mcap", "listed_days"]]
    ps = ps.merge(status[["date", "ticker", "is_halted", "is_admin"]], on=["date", "ticker"], how="left")
    ps["is_halted"] = ps["is_halted"].fillna(False).astype(bool)
    if ps["is_admin"].isna().any() and not provisional:
        raise RuntimeError("is_admin 결측: 관리종목 이력 원천이 확정되지 않았다(Q-06). 4.4절 3단계를 실행할 수 없다 (R-08)")
    ps["is_admin"] = ps["is_admin"].fillna(False).astype(bool)
    ps = ps.join(info, on="ticker")
    out = []
    for t, day in ps.groupby("date", sort=True):
        u = universe_for_date(day, avg.loc[t])
        u.insert(0, "date", t)
        out.append(u)
    u = pd.concat(out, ignore_index=True)
    u["date"] = pd.to_datetime(u["date"])
    return u.astype({"mcap_rank": "int32", "in_universe": "bool"})
