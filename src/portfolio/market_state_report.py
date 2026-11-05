"""작업 B-04 결과 리포트: 2011~2025년 일별 E_t와 regime_mc 그림 (6.9절 완료 기준 (3)).

    python -m src.portfolio.market_state_report --run-id 20261116_B_m8_states [--config configs/prereg.yaml]

reports/{run_id}/에 config.yaml(사본), git_commit.txt, metrics.json, figures/e_t_regime_mc.svg와 그 그림의
원자료 figures/e_t_regime_mc.csv를 쓴다 (9.3절, R-10). 코스피 200은 io.py로만 읽는다 (R-18). 같은 run_id
폴더가 있으면 덮어쓰지 않고 멈춘다 (R-11). requirements에 그림 라이브러리가 없어서 SVG를 직접 쓴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.params import PREREG_PATH, RiskLayerParams, load_prereg
from src.portfolio.risk_layer import InsufficientHistory, _as_bm, load_kospi200, market_state_table
from src.portfolio.vol_target_report import ROOT, RUN_ID_RE, _git_commit

FIGURE = "e_t_regime_mc"  # figures/ 아래 파일 이름 (.svg, .csv)


def figure_years(cfg: Mapping[str, Any]) -> tuple[int, int]:
    """그림의 연도. 시작은 data.first_train_signal_year, 끝은 walk-forward 마지막 테스트 연도다 (2011, 2025)."""
    first = int(cfg["data"]["first_train_signal_year"])
    last = max(int(y) for y in cfg["splits"]["walk_forward_test_years"])
    if last < first:
        raise ValueError("splits.walk_forward_test_years가 data.first_train_signal_year보다 이르다")
    return first, last


def daily_market_states(bm_close: pd.Series, params: RiskLayerParams, years: tuple[int, int]) -> pd.DataFrame:
    """years[0]년 첫 거래일부터 years[1]년 마지막 거래일까지 매 거래일의 Step 4~5 (6.9절).

    각 행은 그날 종가까지만 쓴다. years[1]년보다 늦은 종가는 계산 전에 잘라낸다 (홀드아웃 구간을 읽지 않는다).
    다만 그해가 끝까지 들어왔는지 보려고 다음 거래일이 있는지는 확인한다. E_t나 regime_mc가 하루라도
    결측이면 InsufficientHistory다. bear는 2011년 초처럼 창이 차지 않으면 결측이고, 그날 regime_mc는
    rebound = 0이라 0이다 (D4).
    """
    first, last = years
    bm = _as_bm(bm_close)
    end = pd.Timestamp(f"{last}-12-31")
    if not (bm.index > end).any():
        raise ValueError(f"{last}년 다음 거래일 종가가 없어 {last}년이 끝까지 들어왔는지 확인할 수 없다")
    bm = bm.loc[:end]
    table = market_state_table(bm, params).loc[pd.Timestamp(f"{first}-01-01") :]
    if table.empty:
        raise ValueError(f"{first}~{last}년 코스피 200 종가가 없다")
    missing = table.index[table[["sigma", "regime_mc", "exposure"]].isna().any(axis=1)]
    if len(missing):
        raise InsufficientHistory(f"{missing[0].date()} 등 {len(missing)}일의 E_t·regime_mc를 계산할 이력이 부족하다")
    daily = pd.DataFrame(
        {
            "close": bm.loc[table.index],
            "sigma": table["sigma"],
            "bear": table["bear"].astype("Int8"),
            "rebound": table["rebound"].astype("Int8"),
            "regime_mc": table["regime_mc"].astype("int8"),
            "exposure": table["exposure"],
        }
    )
    daily.index.name = "date"
    return daily


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """flag가 참인 연속 구간의 (시작 위치, 끝 위치). 끝 위치도 구간에 들어간다."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(flag, dtype=np.int8), [0]])))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2])]


def market_state_metrics(bm_close: pd.Series, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """그림의 원자료(일별 표)와 metrics.json의 계산 결과. 입력 종가의 지문(sha256)을 남긴다."""
    params = RiskLayerParams.from_prereg(cfg)
    vol_target = params.require_vol_target()  # FILL_이면 계산 전에 멈춘다 (R-08)
    years = figure_years(cfg)
    daily = daily_market_states(bm_close, params, years)
    used = _as_bm(bm_close).loc[: daily.index[-1]]
    e, regime = daily["exposure"], daily["regime_mc"]
    cap = params.momentum_crash_regime.exposure_cap
    days = daily.index
    metrics = {
        "years": list(years),
        "vol_target_annual": vol_target,
        "n_days": len(daily),
        "first_date": days[0].date().isoformat(),
        "last_date": days[-1].date().isoformat(),
        "exposure": {
            "mean": float(e.mean()),
            "min": float(e.min()),
            "max": float(e.max()),
            "days_at_exposure_min": int((e == params.exposure_min).sum()),
            "days_at_exposure_max": int((e == params.exposure_max).sum()),
            "days_capped_by_regime": int(((regime == 1) & (e == cap)).sum()),
        },
        "regime_mc": {
            "n_days": int(regime.sum()),
            "episodes": [
                {"start": days[s].date().isoformat(), "end": days[t].date().isoformat(), "n_days": t - s + 1}
                for s, t in _runs(regime.to_numpy() == 1)
            ],
        },
        "bear_missing_days": int(daily["bear"].isna().sum()),  # 이날 regime_mc는 rebound = 0이라 0 (D4)
        "by_year": {
            str(y): {
                "n_days": len(g),
                "exposure_mean": float(g["exposure"].mean()),
                "regime_mc_days": int(g["regime_mc"].sum()),
            }
            for y, g in daily.groupby(days.year)
        },
        "kospi200_input": {
            "first_date": used.index[0].date().isoformat(),
            "last_date": used.index[-1].date().isoformat(),
            "n_closes": len(used),
            "sha256_float64_closes": hashlib.sha256(used.to_numpy().tobytes()).hexdigest(),
        },
    }
    return daily, metrics


def svg_figure(daily: pd.DataFrame, params: RiskLayerParams) -> str:
    """위 칸은 E_t, 아래 칸은 코스피 200 종가다. regime_mc = 1인 날은 두 칸 모두 음영으로 칠한다.

    x축은 거래일 순서이고, 해마다 첫 거래일에 눈금을 둔다. 음영은 짧은 구간도 보이게 폭을 2px 이상으로 둔다.
    같은 입력이면 같은 문자열이다.
    """
    width, height, left, right = 960, 470, 64, 16
    e_box, c_box = (44, 244), (284, 424)  # 칸마다 (위 y, 아래 y) 픽셀
    e_lo, e_hi = 0.2, 1.05
    n = len(daily)
    closes = daily["close"].to_numpy(dtype=np.float64)
    c_lo, c_hi = float(closes.min()), float(closes.max())
    c_pad = 0.05 * (c_hi - c_lo) or 1.0

    def x(i: float) -> float:
        return left + (width - left - right) * i / max(n - 1, 1)

    def y_e(v: float) -> float:
        return e_box[1] - (e_box[1] - e_box[0]) * (v - e_lo) / (e_hi - e_lo)

    def y_c(v: float) -> float:
        return c_box[1] - (c_box[1] - c_box[0]) * (v - c_lo + c_pad) / (c_hi - c_lo + 2 * c_pad)

    def points(ys: Sequence[float]) -> str:
        return " ".join(f"{x(i):.1f},{y:.1f}" for i, y in enumerate(ys))

    first, last = daily.index[0].year, daily.index[-1].year
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'font-family="sans-serif" font-size="12">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="22" font-size="15">M8 노출 상한 E_t와 regime_mc (코스피 200, {first}~{last}년 매 거래일)</text>',
    ]
    inner = width - left - right
    step = inner / max(n - 1, 1)
    shade = 'fill="#f4b183" opacity="0.6"'
    for s, t in _runs(daily["regime_mc"].to_numpy() == 1):
        w = max((t - s + 1) * step, 2.0)  # 1~2일짜리 구간도 보이게 최소 2px
        x0 = (x(s) + x(t)) / 2 - w / 2
        for top, bottom in (e_box, c_box):
            out.append(f'<rect x="{x0:.1f}" y="{top}" width="{w:.1f}" height="{bottom - top}" {shade}/>')
    cap = params.momentum_crash_regime.exposure_cap
    for v, label in ((params.exposure_min, "하한"), (cap, "국면 상한"), (params.exposure_max, "")):
        y = y_e(v)
        dashed = 'stroke="#999" stroke-dasharray="4 3"'
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" {dashed}/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{v:.2f}</text>')
        if label:
            out.append(f'<text x="{width - right - 4}" y="{y - 4:.1f}" text-anchor="end" fill="#666">{label}</text>')
    for ys, color in (([y_e(v) for v in daily["exposure"]], "#1f4e79"), ([y_c(v) for v in closes], "#333")):
        out.append(f'<polyline points="{points(ys)}" fill="none" stroke="{color}" stroke-width="1"/>')
    for v in (c_lo, c_hi):
        out.append(f'<text x="{left - 6}" y="{y_c(v) + 4:.1f}" text-anchor="end">{v:,.0f}</text>')
    for top, bottom, name in ((*e_box, "E_t"), (*c_box, "코스피 200 종가")):
        out.append(f'<rect x="{left}" y="{top}" width="{inner}" height="{bottom - top}" fill="none" stroke="#333"/>')
        out.append(f'<text x="{left}" y="{top - 6}">{name}</text>')  # 칸 위에 둬서 선과 겹치지 않게
    years = pd.Series(np.arange(n), index=daily.index).groupby(daily.index.year).min()
    for year, i in years.items():
        out.append(f'<line x1="{x(i):.1f}" y1="{c_box[1]}" x2="{x(i):.1f}" y2="{c_box[1] + 5}" stroke="#333"/>')
        out.append(f'<text x="{x(i):.1f}" y="{c_box[1] + 18}" text-anchor="middle">{year}</text>')
    legend_y = height - 16
    out.append(f'<rect x="{left}" y="{legend_y - 10}" width="18" height="10" {shade}/>')
    out.append(f'<text x="{left + 24}" y="{legend_y}">regime_mc = 1 (bear × rebound, E_t ≤ 0.70)</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="B-04 일별 E_t·regime_mc 그림 (6.9절 완료 기준 (3))")
    ap.add_argument("--run-id", required=True, help="{YYYYMMDD}_B_{짧은설명}")
    ap.add_argument("--config", default=str(PREREG_PATH), help="부록 1 설정 (기본 configs/prereg.yaml)")
    args = ap.parse_args(argv)
    if not RUN_ID_RE.match(args.run_id):
        raise SystemExit(f"run_id는 '{{YYYYMMDD}}_B_{{짧은설명}}' 형식이어야 한다: {args.run_id!r}")

    out = ROOT / "reports" / args.run_id
    if out.exists():
        raise SystemExit(f"{out}가 이미 있다. 결과를 덮어쓰지 않는다 (R-11)")
    cfg_path = Path(args.config)
    cfg = load_prereg(cfg_path)
    daily, metrics = market_state_metrics(load_kospi200(), cfg)  # vol_target_annual이 FILL_이면 여기서 멈춘다 (R-08)
    svg = svg_figure(daily, RiskLayerParams.from_prereg(cfg))

    (out / "figures").mkdir(parents=True)
    shutil.copyfile(cfg_path, out / "config.yaml")
    (out / "git_commit.txt").write_text(_git_commit(), encoding="utf-8")
    metrics = {"run_id": args.run_id, "task": "B-04", **metrics}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "figures" / f"{FIGURE}.svg").write_text(svg, encoding="utf-8")
    daily.to_csv(out / "figures" / f"{FIGURE}.csv", date_format="%Y-%m-%d", lineterminator="\n")
    print(f"{metrics['first_date']} ~ {metrics['last_date']} {metrics['n_days']}일, regime_mc = 1: "
          f"{metrics['regime_mc']['n_days']}일 → {out}")
    return out


if __name__ == "__main__":
    main()
