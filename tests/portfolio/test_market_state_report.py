"""작업 B-04: 6.9절 완료 기준 (3) 일별 E_t·regime_mc 그림 리포트 (market_state_report)."""
from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
import yaml

from src.portfolio import market_state_report as report
from src.portfolio.params import PreregValueMissing, RiskLayerParams
from src.portfolio.risk_layer import InsufficientHistory, market_state_table
from tests.portfolio.conftest import VOL_TARGET_FOR_TESTS


@pytest.fixture(scope="module")
def cfg_filled(cfg) -> dict:
    """그림 연도를 2011~2012년으로 줄이고 vol_target_annual을 테스트 값으로 채운 설정 (R-08)."""
    c = copy.deepcopy(cfg)
    c["risk_layer"]["vol_target_annual"] = VOL_TARGET_FOR_TESTS
    c["splits"]["walk_forward_test_years"] = [2012]
    return c


def _bm() -> pd.Series:
    """2009~2013년 1월 코스피 200 대용 경로. 2011년 1~3월에 급락하고 반등해 regime_mc = 1인 날이 생긴다."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2009-01-01", "2013-01-31")
    drift = np.full(len(days), 0.0003)
    crash = int(days.searchsorted(pd.Timestamp("2011-01-03")))
    drift[crash : crash + 60] = -0.006
    drift[crash + 60 : crash + 85] = 0.009  # 20일 반등 약 18%로 rebound_t = 1
    return pd.Series(100.0 * np.exp(np.cumsum(drift + rng.normal(0.0, 0.008, len(days)))), index=days)


def test_figure_years_follow_config(cfg):
    assert report.figure_years(cfg) == (2011, 2025)


def test_daily_states_cover_years_and_ignore_later_data(cfg_filled):
    params = RiskLayerParams.from_prereg(cfg_filled)
    bm = _bm()
    daily = report.daily_market_states(bm, params, (2011, 2012))
    expected_days = bm.loc["2011-01-01":"2012-12-31"].index
    assert daily.index.equals(expected_days) and daily.index.name == "date"
    assert not daily[["sigma", "regime_mc", "exposure"]].isna().any().any()
    full = market_state_table(bm, params).loc[expected_days]  # 각 행이 그날까지만 쓰므로 전체로 계산해도 같다
    assert np.allclose(daily["exposure"], full["exposure"]) and (daily["regime_mc"] == full["regime_mc"]).all()
    assert daily["exposure"].between(params.exposure_min, params.exposure_max).all()
    assert (daily.loc[daily["regime_mc"] == 1, "exposure"] <= params.momentum_crash_regime.exposure_cap).all()

    changed = bm.copy()
    changed.loc["2013-01-01":] *= 3.0  # 그림 기간 뒤의 값은 결과를 바꾸지 않는다
    pd.testing.assert_frame_equal(report.daily_market_states(changed, params, (2011, 2012)), daily)


def test_short_history_or_unfinished_year_stops(cfg_filled):
    params = RiskLayerParams.from_prereg(cfg_filled)
    bm = _bm()
    with pytest.raises(InsufficientHistory):  # 2011년 첫날 sigma_t 창을 채울 2010년 종가가 없다
        report.daily_market_states(bm.loc["2011-01-01":], params, (2011, 2012))
    with pytest.raises(ValueError, match="끝까지"):  # 2012년 다음 거래일이 없으면 그해가 다 들어왔는지 모른다
        report.daily_market_states(bm.loc[:"2012-12-31"], params, (2011, 2012))


def test_metrics_summarize_daily_table(cfg_filled):
    daily, m = report.market_state_metrics(_bm(), cfg_filled)
    assert m["years"] == [2011, 2012] and m["n_days"] == len(daily) and m["vol_target_annual"] == VOL_TARGET_FOR_TESTS
    assert m["regime_mc"]["n_days"] == int(daily["regime_mc"].sum()) > 0
    assert sum(ep["n_days"] for ep in m["regime_mc"]["episodes"]) == m["regime_mc"]["n_days"]
    assert sorted(m["by_year"]) == ["2011", "2012"]
    assert sum(y["n_days"] for y in m["by_year"].values()) == len(daily)
    assert m["exposure"]["mean"] == pytest.approx(daily["exposure"].mean())
    assert m["kospi200_input"]["last_date"] == daily.index[-1].date().isoformat()

    later = _bm()
    later.loc["2013-01-01":] *= 3.0
    assert report.market_state_metrics(later, cfg_filled)[1]["kospi200_input"] == m["kospi200_input"]
    earlier = _bm()
    earlier.iloc[5] *= 1.01
    assert report.market_state_metrics(earlier, cfg_filled)[1]["kospi200_input"] != m["kospi200_input"]


def test_svg_is_well_formed_and_deterministic(cfg_filled):
    params = RiskLayerParams.from_prereg(cfg_filled)
    daily, m = report.market_state_metrics(_bm(), cfg_filled)
    svg = report.svg_figure(daily, params)
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    assert root.tag == f"{ns}svg"
    lines = root.findall(f"{ns}polyline")
    assert len(lines) == 2 and all(len(p.get("points").split()) == len(daily) for p in lines)  # E_t, 코스피 200
    shaded = [r for r in root.findall(f"{ns}rect") if r.get("fill") == "#f4b183" and r.get("opacity")]
    assert len(shaded) == 2 * len(m["regime_mc"]["episodes"]) + 1  # 구간마다 두 칸 + 범례
    assert svg == report.svg_figure(daily, params)


def test_report_writes_run_folder_once(cfg, cfg_filled, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "load_kospi200", _bm)
    monkeypatch.setattr(report, "_git_commit", lambda: "abc\n")
    config = tmp_path / "prereg.yaml"
    config.write_text(yaml.safe_dump(cfg_filled, allow_unicode=True), encoding="utf-8")

    out = report.main(["--run-id", "20261116_B_test", "--config", str(config)])
    assert out == tmp_path / "reports" / "20261116_B_test"
    assert sorted(p.name for p in out.iterdir()) == ["config.yaml", "figures", "git_commit.txt", "metrics.json"]
    assert sorted(p.name for p in (out / "figures").iterdir()) == [f"{report.FIGURE}.csv", f"{report.FIGURE}.svg"]
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert (metrics["run_id"], metrics["task"]) == ("20261116_B_test", "B-04")
    csv = pd.read_csv(out / "figures" / f"{report.FIGURE}.csv", parse_dates=["date"], index_col="date")
    assert len(csv) == metrics["n_days"]
    assert list(csv.columns) == ["close", "sigma", "bear", "rebound", "regime_mc", "exposure"]
    with pytest.raises(SystemExit, match="R-11"):  # 같은 run_id는 덮어쓰지 않는다
        report.main(["--run-id", "20261116_B_test", "--config", str(config)])

    unfilled_cfg = copy.deepcopy(cfg_filled)
    unfilled_cfg["risk_layer"]["vol_target_annual"] = cfg["risk_layer"]["vol_target_annual"]  # FILL_
    unfilled = tmp_path / "unfilled.yaml"
    unfilled.write_text(yaml.safe_dump(unfilled_cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(PreregValueMissing):  # vol_target_annual = FILL_이면 계산하지 않는다 (R-08)
        report.main(["--run-id", "20261116_B_fill", "--config", str(unfilled)])
    assert not (tmp_path / "reports" / "20261116_B_fill").exists()
