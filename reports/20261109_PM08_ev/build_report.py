"""PM-08 ev_* 실데이터 생성과 완료 기준 점검 (명세서 6.2절).

보관된 OpenDART 목록 원본과 A 11/8판 tickers·universe·index만 읽는다. API 호출과 원본 수정은 없다.
수익률·라벨은 쓰지 않는다. 사용: python -B -I build_report.py <worktree> <repo root>
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
WORK, ROOT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(WORK))

import yaml  # noqa: E402

from src.common import io  # noqa: E402
from src.data.disclosures import build_disclosures  # noqa: E402
from src.events.alerts import attach_stock_codes  # noqa: E402
from src.features.event_features import build_event_features, tag_events  # noqa: E402

OUT = Path(__file__).resolve().parent
RAWS = [ROOT / "data/raw/2026-10-29/20261029_PM_disclosures_full",
        ROOT / "data/raw/2026-11-09/20261109_PM_disclosures_backfill"]
A_DATA = ROOT / "data/processed"
STORE = ROOT / "data/interim/pm08-ev-output"
SEED = 20261109

cfg = yaml.safe_load((WORK / "configs/prereg.yaml").read_text(encoding="utf-8"))
tickers = pd.read_parquet(A_DATA / "tickers.parquet")
universe = pd.read_parquet(A_DATA / "universe.parquet")
index = pd.read_parquet(A_DATA / "index.parquet")
days = pd.DatetimeIndex(sorted(index.loc[index["index_code"] == "KOSPI200", "date"]))

result = build_disclosures(RAWS, today=date(2026, 11, 9), tickers=tickers)
frame = attach_stock_codes(result.df, result.observations)
ends = pd.to_datetime(result.windows["end"])
first, last = pd.Timestamp("2010-01-01"), ends.max()
assert result.windows.loc[ends == last, "state"].eq("complete").all() and not result.windows["open"].any()
coverage = (first, last)
print("공시", len(frame), frame["rcept_date"].min().date(), frame["rcept_date"].max().date(), "보관 끝", last.date(),
      "충돌", len(result.conflicts), "구간 상태", result.windows["state"].value_counts().to_dict())

keywords = cfg["features"]["event_keywords"]
names = list(keywords)
all_signals = sorted(universe.loc[universe["in_universe"], "date"].unique())
members = universe[universe["date"] <= last]
dropped = [pd.Timestamp(t) for t in all_signals if pd.Timestamp(t) > last]
print("보관 구간 뒤라 제외한 신호일:", [t.date().isoformat() for t in dropped])
features = build_event_features(frame, tickers, members, days, cfg, coverage)
io.validate("features", features)

# 완료 기준 (3) + 병합 전제: 신호일별 U_t 전 종목과 정확히 같다.
key = ["date", "ticker"]
want = members.loc[members["in_universe"], key].sort_values(key).reset_index(drop=True)
have = features[key].sort_values(key).reset_index(drop=True)
same_keys = bool((want["date"].to_numpy() == have["date"].to_numpy()).all()
                 and (want["ticker"].astype(str).to_numpy() == have["ticker"].astype(str).to_numpy()).all()
                 and len(want) == len(have))
print("U_t와 행 일치", same_keys, len(have), "중복", int(features.duplicated(key).sum()))
assert same_keys

# 독립 재계산: 공시별로 켜지는 신호일을 구해 1의 집합을 비교한다.
events = tag_events(frame, tickers, keywords, cfg["events"]["title_filters"])
signals = pd.DatetimeIndex(sorted(features["date"].unique()))
window = int(cfg["features"]["event_window_days"])
position = days.searchsorted(signals)
starts = days[position - window + 1]
cut = signals.tz_localize("Asia/Seoul") + pd.Timedelta(hours=15, minutes=30)
ons = set()
received = pd.to_datetime(events["rcept_date"]).to_numpy("datetime64[ns]")
available = pd.DatetimeIndex(events["available_at"]).tz_convert("UTC")
cut_utc = cut.tz_convert("UTC")
first_signal = cut_utc.searchsorted(available, side="right")              # available < cut인 첫 신호일
last_signal = starts.searchsorted(pd.DatetimeIndex(received), side="right") - 1   # start <= 접수일인 끝 신호일
in_u = set(zip(want["date"], want["ticker"].astype(str)))
for name, ticker, a, b in zip(events["feature"], events["ticker"].astype(str), first_signal, last_signal):
    for i in range(int(a), int(b) + 1):
        pair = (signals[i], ticker)
        if pair in in_u:
            ons.add((signals[i], ticker, name))
melted = features.melt(id_vars=key, value_vars=names, var_name="feature", value_name="v")
mine = {(d, str(t), f) for d, t, f in melted.loc[melted["v"] == 1, ["date", "ticker", "feature"]].itertuples(index=False)}
print("독립 재계산 1의 개수", len(ons), "본 계산", len(mine), "차이", len(ons ^ mine))
assert ons == mine

# 완료 기준 (2): 무작위 신호일 20개, t+1일 이후를 모두 지운 입력
rng = np.random.default_rng(SEED)
picked = sorted(rng.choice(len(signals), size=20, replace=False))
leak = []
for i in picked:
    t = signals[i]
    known_tickers = tickers[tickers["listing_date"] <= t].copy()
    known_tickers.loc[known_tickers["delisting_date"] > t, "delisting_date"] = pd.NaT
    cut_frame = frame[pd.to_datetime(frame["rcept_date"]) <= t]
    obs = build_event_features(cut_frame, known_tickers, members[members["date"] == t], days[days <= t], cfg, coverage)
    expected = features[features["date"] == t]
    a, b = obs.sort_values("ticker").reset_index(drop=True), expected.sort_values("ticker").reset_index(drop=True)
    leak.append({"date": t.date().isoformat(), "n": len(a), "identical": bool(a[names].equals(b[names])
                                                                               and a["ticker"].astype(str).equals(b["ticker"].astype(str)))})
leak = pd.DataFrame(leak)
print("누수 테스트 일치", int(leak["identical"].sum()), "/", len(leak))
assert leak["identical"].all()

# 완료 기준 (1): 결측률 표와 신호일별 분포
missing = pd.DataFrame({"missing_rate": features[names].isna().mean(), "rows": len(features)})
by_date = features.groupby("date")[names].mean()
summary = by_date.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
by_year = features.assign(year=features["date"].dt.year).groupby("year")[names].mean()
by_month = features.assign(month=features["date"].dt.month).groupby("month")[names].mean()
missing.to_csv(OUT / "missing_rate.csv", encoding="utf-8-sig")
summary.to_csv(OUT / "signal_date_distribution.csv", encoding="utf-8-sig")
by_year.to_csv(OUT / "mean_by_year.csv", encoding="utf-8-sig")
by_month.to_csv(OUT / "mean_by_month.csv", encoding="utf-8-sig")
by_date.to_csv(OUT / "mean_by_signal_date.csv", encoding="utf-8-sig")
leak.to_csv(OUT / "leak_test_20_dates.csv", index=False, encoding="utf-8-sig")
print(missing.to_string()); print(summary.round(4).to_string()); print(by_year.round(3).to_string()); print(by_month.round(3).to_string())

STORE.mkdir(parents=True, exist_ok=True)
path = STORE / "ev_features_20261109.parquet"
if path.exists():                                     # 공시를 뒤에 덧붙여도 이전 신호일 값은 바뀌면 안 된다
    old = pd.read_parquet(path).sort_values(key).reset_index(drop=True)
    now = features[features["date"].isin(old["date"])].sort_values(key).reset_index(drop=True)
    same = len(old) == len(now) and bool((old[names].to_numpy() == now[names].to_numpy()).all())
    print("이전 결과와 비교", old["date"].nunique(), "개 신호일", len(old), "행, 완전히 같음:", same)
    assert same
features.to_parquet(path, index=False)
commit = subprocess.check_output(["git", "-C", str(WORK), "rev-parse", "HEAD"], text=True).strip()
(OUT / "summary.json").write_text(json.dumps({
    "signal_dates": int(features["date"].nunique()), "first": str(signals[0].date()), "last": str(signals[-1].date()),
    "rows": len(features), "dropped_signal_dates": [t.date().isoformat() for t in dropped],
    "coverage_end": str(last.date()), "raw_runs": [p.name for p in RAWS], "independent_diff": len(ons ^ mine), "ones": len(mine),
    "leak_identical": f"{int(leak['identical'].sum())}/{len(leak)}", "base_commit": commit,
    "seed": SEED}, ensure_ascii=False, indent=2), encoding="utf-8")
print("저장", path)
