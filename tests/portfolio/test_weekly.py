"""run_weekly 4·5단계 (src/portfolio/weekly.py): forward positions가 백테스트 버퍼 규칙과 같은지."""
from __future__ import annotations

import copy
import inspect
import logging
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.common import io
from src.events.alerts import AlertsNotCollected, AlertsNotStarted
from src.pipeline.run_weekly import StepContext
from src.portfolio import weekly
from src.portfolio.buffer_rule import run_buffer_rule
from src.portfolio.engine_adapter import build_engine, engine_inputs
from src.portfolio.params import BufferRuleParams, load_prereg
from src.portfolio.rl_data import build_snapshots
from tests.portfolio.fakes import market_tables

N_FORWARD = 7  # 합성 시장 끝쪽 신호일 7개를 forward 기간으로 본다


@dataclass
class TableCtx(StepContext):
    """io 대신 메모리 표를 읽고 쓰는 StepContext. 스텁 판단(uses_stub, stub_tables)은 원래 것을 쓴다."""

    tables: dict = field(default_factory=dict)
    written: dict = field(default_factory=dict)  # (name, date) → (df, 스텁 여부)

    def read_table(self, name, **kwargs):
        df = self.tables[name]
        if name == "signals":
            return df.loc[pd.to_datetime(df["date"]) == pd.Timestamp(kwargs["date"])].reset_index(drop=True)
        return df

    def write_table(self, name, df, **kwargs):
        self.written[(name, kwargs.get("date"))] = (df, self.uses_stub)
        if self.uses_stub:
            self.stub_tables.add(name)
        return f"{'_stub/' if self.uses_stub else ''}{name}/{kwargs.get('date')}.parquet"


class PeriodAsForward:
    """main의 C 엔진에 run_to_signal_close가 없을 때 쓰는 대역. run_period의 weights_obs도 t′ 종가 보유이고
    state를 그대로 이어 넘길 수 있어 재생한 보유가 같다 (C PR #22 test_forward_replay_uses_actual_holdings...)."""

    def __init__(self, engine):
        self.engine = engine

    def initial_state(self):
        return self.engine.initial_state()

    def run_to_signal_close(self, state, targets, signal_date, next_signal_date):
        return self.engine.run_period(state, targets, signal_date, next_signal_date)


@pytest.fixture(scope="module")
def setup(cfg, risk_params, market):
    # 정지 10%: 체결되지 않은 목표가 생겨 '기록된 목표 ≠ 실제 보유'인 주가 나온다 (3%에서는 7주 동안 안 생긴다)
    tables, kept = market_tables(market, halt_share=0.1)
    dates = kept.signal_dates[-N_FORWARD - 1 : -1]  # 마지막 신호일 뒤에도 거래일이 있게 한다
    fcfg = copy.deepcopy(cfg)
    fcfg["splits"]["forward_test"]["positions_from"] = dates[0]
    buffer = BufferRuleParams.from_prereg(fcfg)
    cooldown = lambda d: kept.cooldown.get(d, frozenset())  # noqa: E731
    snaps = build_snapshots(dates, kept.signals, kept.features, kept.universe, kept.sector, kept.bm_close,
                            cooldown, risk_params)
    engine = build_engine(engine_inputs(tables["prices"], tables["status"], tables["index"], tables["delisted"]), fcfg)
    backtest = run_buffer_rule(snaps, engine, buffer, risk_params)
    all_tables = {
        **tables,
        "delisted_reused": pd.DataFrame(columns=["ticker", "delist_date", "last_price"]).astype({"last_price": "float64"}),
        "signals": kept.signals, "universe": kept.universe, "sector": kept.sector,
    }
    by_date = {d: g.reset_index(drop=True) for d, g in backtest.positions.groupby(backtest.positions["date"].dt.strftime("%Y-%m-%d"))}
    return SimpleNamespace(cfg=fcfg, buffer=buffer, risk=risk_params, dates=dates, cooldown=cooldown,
                           tables=all_tables, backtest=by_date, earlier=kept.signal_dates[-N_FORWARD - 2])


@pytest.fixture
def wired(setup, monkeypatch):
    """PM·D 함수를 연결한 상태. 재생 엔진은 C의 run_to_signal_close가 있으면 그것을, 없으면 대역을 쓴다."""
    monkeypatch.setattr(weekly, "_load_params", lambda: (setup.cfg, setup.buffer, setup.risk))
    monkeypatch.setattr(weekly, "M9_COOLDOWN_ENTRY", setup.cooldown)
    monkeypatch.setattr(weekly, "M9_ENGINE_ENTRY", lambda first, last: ({}, {}))
    real_build = weekly.build_engine

    def build(inputs, cfg, **kw):
        engine = real_build(inputs, cfg, **kw)
        return engine if hasattr(engine, "run_to_signal_close") else PeriodAsForward(engine)

    monkeypatch.setattr(weekly, "build_engine", build)
    return setup


def _run(setup, date, recorded=None, monkeypatch=None):
    if recorded is not None:
        monkeypatch.setattr(weekly, "RECORDED_POSITIONS_ENTRY", lambda t: recorded)
    ctx = TableCtx(date=date, tables=setup.tables, log=logging.getLogger("test_weekly"))
    weekly.m4_step(ctx)
    weekly.m8_step(ctx)
    return ctx


def test_forward_positions_match_the_backtest_buffer_rule(wired, monkeypatch):
    """매주 forward 단계가 내는 positions = 같은 신호일의 run_buffer_rule positions.
    지난 보유는 그 전 주까지의 기록(백테스트 positions)을 엔진으로 재생해 만든다."""
    s = wired
    replayed, real_replay = [], weekly.replay_holdings

    def spy(engine, recorded, dates, date):
        held = real_replay(engine, recorded, dates, date)
        if dates:
            replayed.append((held, set(weekly._targets(recorded[dates[-1]], dates[-1]))))
        return held

    monkeypatch.setattr(weekly, "replay_holdings", spy)
    for k, t in enumerate(s.dates):
        recorded = {d: s.backtest[d] for d in s.dates[:k]}
        ctx = _run(s, t, recorded, monkeypatch)
        df, stub = ctx.written[("positions", t)]
        assert not stub and not ctx.uses_stub
        want = s.backtest[t]
        assert df["ticker"].tolist() == want["ticker"].tolist()
        np.testing.assert_allclose(df["target_weight"], want["target_weight"], rtol=0, atol=1e-12)
        assert (df["constructor"] == "buffer").all()
        assert df["exposure"].iloc[0] == pytest.approx(want["exposure"].iloc[0], abs=1e-12)
        assert df["regime_mc"].iloc[0] == want["regime_mc"].iloc[0]
    held_later = [set(s.backtest[d]["ticker"]) - {"CASH"} for d in s.dates[1:]]
    assert any(h & set(s.backtest[s.dates[0]]["ticker"]) for h in held_later)  # 버퍼로 유지된 종목이 있다
    assert len(replayed) == N_FORWARD - 1
    assert any(held != targets for held, targets in replayed)  # 기록된 목표와 실제 보유가 다른 주가 있었다


def test_no_positions_before_positions_from(wired, monkeypatch):
    ctx = _run(wired, wired.earlier)
    assert ctx.memory["w_raw"].empty and ctx.written == {}


def test_first_record_needs_no_past_records_but_the_next_one_does(wired, monkeypatch):
    monkeypatch.setattr(weekly, "RECORDED_POSITIONS_ENTRY", None)
    ctx = _run(wired, wired.dates[0])
    assert ("positions", wired.dates[0]) in ctx.written  # 첫 기록은 현금 100%에서 시작한다
    with pytest.raises(RuntimeError, match="D 함수"):
        _run(wired, wired.dates[1])


def _no_alert_files(*args):
    raise AlertsNotStarted("alerts/에 날짜별 경보 파일이 없다")  # PR #51 _require_started와 같은 경우


@pytest.fixture
def alerts_root(tmp_path, monkeypatch):
    """io.ROOT를 빈 임시 폴더로 바꾼다. alerts/{d}.parquet을 만들면 '경보 파일이 있는' 상태다."""
    monkeypatch.setattr(io, "ROOT", tmp_path)

    def add(day):
        (tmp_path / "alerts").mkdir(exist_ok=True)
        (tmp_path / "alerts" / f"{day}.parquet").write_bytes(b"")  # 날짜 목록만 세는 경우만 쓴다 (내용은 읽지 않는다)

    return add


@pytest.mark.parametrize(
    ("cooldown", "engine", "reason"),
    [
        (None, None, "함수가 연결되지 않았다"),
        ("src.events.no_such_module:cooldown_at", "src.events.no_such_module:engine_alerts", "찾지 못했다"),  # PR 미병합
        ("src.events.alerts:no_such_function", "src.events.alerts:no_such_function", "찾지 못했다"),
        (_no_alert_files, _no_alert_files, "날짜별 경보 파일이 하나도 없다"),
        # 실제 PM 함수 (PR #51): 빈 alerts/에서 AlertsNotStarted를 낸다
        ("src.events.alerts:cooldown_at", "src.events.alerts:engine_alerts", "날짜별 경보 파일이 하나도 없다"),
    ],
    ids=["not-wired", "not-merged", "no-function", "no-alert-files", "pm-m9-not-started"],
)
def test_missing_m9_records_without_cooldown_as_a_deviation(wired, monkeypatch, alerts_root, cooldown, engine, reason):
    """PM 11/9 결정: M9가 없어도 기록을 미루지 않는다. 결과는 M9가 아무 경보도 주지 않을 때와 같고, 스텁이 아니다."""
    t, recorded = wired.dates[2], {d: wired.backtest[d] for d in wired.dates[:2]}
    monkeypatch.setattr(weekly, "M9_COOLDOWN_ENTRY", lambda d: ())
    monkeypatch.setattr(weekly, "M9_ENGINE_ENTRY", lambda first, last: ({}, {}))
    quiet_m9 = _run(wired, t, recorded, monkeypatch).written[("positions", t)][0]
    monkeypatch.setattr(weekly, "M9_COOLDOWN_ENTRY", cooldown)
    monkeypatch.setattr(weekly, "M9_ENGINE_ENTRY", engine)
    ctx = _run(wired, t, recorded, monkeypatch)
    df, stub = ctx.written[("positions", t)]
    assert ctx.stubbed == [] and not stub  # 7단계가 커밋한다
    pd.testing.assert_frame_equal(df, quiet_m9)
    notes = ctx.memory[weekly.DEVIATIONS_KEY]
    assert len(notes) == 2 and "M9 쿨다운 없이" in notes[0] and "M9 체결 지도 없이" in notes[1]
    assert all(n.startswith(t) and reason in n and "deviation log" in n for n in notes)


def test_other_m9_errors_stop_the_run(wired, monkeypatch):
    """M9가 돌았는데 오류가 나면 쿨다운 없이 넘어가지 않는다 (틀린 경보로 기록하지 않게)."""
    def broken(t):
        raise ValueError("M9 경보 표 스키마 오류")

    monkeypatch.setattr(weekly, "M9_COOLDOWN_ENTRY", broken)
    with pytest.raises(ValueError, match="스키마 오류"):
        _run(wired, wired.dates[0])


@pytest.mark.parametrize("which", ["M9_COOLDOWN_ENTRY", "M9_ENGINE_ENTRY"])
@pytest.mark.parametrize(
    ("error", "files"),
    [
        (AlertsNotCollected, ["2026-11-18"]),  # 일간 경보를 빠뜨렸다 (PM #51)
        (FileNotFoundError, ["2026-11-18"]),  # M9가 읽는 index가 없음 등 (PM #36 리뷰)
        (FileNotFoundError, []),  # 경보 파일이 없어도 AlertsNotStarted가 아니면 'M9 없음'이 아니다
    ],
    ids=["not-collected", "other-file", "other-file-no-alerts"],
)
def test_file_errors_other_than_not_started_stop_the_run(wired, monkeypatch, alerts_root, which, error, files):
    """M9 없이 기록하는 것은 AlertsNotStarted뿐이다 (PM #51). 다른 FileNotFoundError는 쿨다운 없이 기록하지 않고 멈춘다."""
    for day in files:
        alerts_root(day)

    def unreadable(*args):
        raise error("alerts/2026-11-19.parquet")

    monkeypatch.setattr(weekly, which, unreadable)
    t, recorded = wired.dates[2], {d: wired.backtest[d] for d in wired.dates[:2]}
    with pytest.raises(error, match="2026-11-19"):
        _run(wired, t, recorded, monkeypatch)


def test_pm_m9_stops_when_a_needed_alert_file_is_missing(wired, monkeypatch, alerts_root):
    """실제 PM 함수 (PR #51): 경보 파일은 있는데 t까지 필요한 날짜 파일이 빠졌으면 AlertsNotCollected로 멈춘다."""
    days = pd.DatetimeIndex(pd.to_datetime(wired.tables["index"]["date"]))
    monkeypatch.setattr("src.events.alerts.load_trading_days", lambda: days)
    t = wired.dates[2]
    alerts_root(wired.dates[0])  # t 파일은 없다
    monkeypatch.setattr(weekly, "M9_COOLDOWN_ENTRY", "src.events.alerts:cooldown_at")
    with pytest.raises(AlertsNotCollected, match="경보 파일이 없는 날짜") as e:
        _run(wired, t, {d: wired.backtest[d] for d in wired.dates[:2]}, monkeypatch)
    assert not isinstance(e.value, AlertsNotStarted)


def _zero_alerts():
    """9.3절 alerts 형식의 0행 표 (경보 0건으로 확정된 날의 파일, DEC-33)."""
    at = pd.Series(pd.DatetimeIndex([pd.Timestamp("2026-01-05 10:00")]).tz_localize("Asia/Seoul"))
    at = at.astype(pd.DatetimeTZDtype("ns", "Asia/Seoul"))
    return pd.DataFrame({"date": [pd.Timestamp("2026-01-05")], "ticker": ["000001"], "trigger_type": ["T1"],
                         "rcept_no": ["20260105000001"], "report_nm": ["테스트 공시"], "rcept_dt": at,
                         "available_at": at, "generated_at": at}).iloc[0:0]


def test_pm_m9_reads_alert_files_only_through_t(wired, monkeypatch, alerts_root):
    """실제 PM 함수 (PR #51): 첫 체결일의 쿨다운 근거일부터 t까지 0행 파일만 있으면 M9를 적용해 기록한다.
    일간 경보는 t 다음 날 08:00에 t일 파일까지 만들고 주간이 그 뒤에 돈다 (D-06). M9가 t 다음 날 이후 파일을
    찾았다면 AlertsNotCollected로 멈췄을 것이다."""
    days = pd.DatetimeIndex(pd.to_datetime(wired.tables["index"]["date"]))
    monkeypatch.setattr("src.events.alerts.load_trading_days", lambda: days)
    t, recorded = wired.dates[2], {d: wired.backtest[d] for d in wired.dates[:2]}
    monkeypatch.setattr(weekly, "M9_COOLDOWN_ENTRY", lambda d: ())
    quiet_m9 = _run(wired, t, recorded, monkeypatch).written[("positions", t)][0]
    cooldown = int(load_prereg()["events"]["cooldown_trading_days"])  # PM 함수가 읽는 값
    first_execution = int(days.searchsorted(pd.Timestamp(wired.dates[0]), side="right"))
    for day in pd.date_range(days[first_execution - cooldown], t, freq="D"):
        io.write_table("alerts", _zero_alerts(), date=day.strftime("%Y-%m-%d"))
    monkeypatch.setattr(weekly, "M9_COOLDOWN_ENTRY", "src.events.alerts:cooldown_at")
    monkeypatch.setattr(weekly, "M9_ENGINE_ENTRY", "src.events.alerts:engine_alerts")
    ctx = _run(wired, t, recorded, monkeypatch)
    pd.testing.assert_frame_equal(ctx.written[("positions", t)][0], quiet_m9)
    assert weekly.DEVIATIONS_KEY not in ctx.memory and ctx.stubbed == []
    assert max(io.list_dates("alerts")) == t


@pytest.fixture
def broken_m9_module(tmp_path, monkeypatch):
    """모듈은 있는데 그 안의 import가 실패하는 M9 (하위 의존성 없음)."""
    (tmp_path / "b_test_broken_m9.py").write_text(
        "import b_test_no_such_dependency\n\ndef cooldown_at(t):\n    return set()\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "b_test_broken_m9:cooldown_at"
    sys.modules.pop("b_test_broken_m9", None)


def test_import_failure_inside_the_m9_module_stops_the_run(wired, monkeypatch, alerts_root, broken_m9_module):
    monkeypatch.setattr(weekly, "M9_COOLDOWN_ENTRY", broken_m9_module)
    with pytest.raises(ModuleNotFoundError, match="b_test_no_such_dependency"):
        _run(wired, wired.dates[0])


@pytest.mark.parametrize(
    "entry", ["src.no_such_package.alerts:f", "src.events.no_such_module:f", "src.events.alerts:no_such_function"]
)
def test_resolve_reports_missing_modules_and_functions(entry):
    with pytest.raises(weekly.EntryNotFound, match="찾지 못했다"):
        weekly._resolve(entry, "PM M9 쿨다운")


def test_resolve_keeps_import_errors_inside_an_existing_module(broken_m9_module):
    with pytest.raises(ModuleNotFoundError) as e:
        weekly._resolve(broken_m9_module, "PM M9 쿨다운")
    assert not isinstance(e.value, weekly.EntryNotFound) and e.value.name == "b_test_no_such_dependency"


def test_missing_m9_can_still_block_the_commit(wired, monkeypatch):
    monkeypatch.setattr(weekly, "RECORD_WITHOUT_M9", False)
    monkeypatch.setattr(weekly, "M9_COOLDOWN_ENTRY", None)
    monkeypatch.setattr(weekly, "M9_ENGINE_ENTRY", None)
    t = wired.dates[2]
    ctx = _run(wired, t, {d: wired.backtest[d] for d in wired.dates[:2]}, monkeypatch)
    assert ctx.stubbed == ["4(M9 쿨다운 없음)", "4(M9 체결 지도 없음)"]
    _, stub = ctx.written[("positions", t)]
    assert stub  # _stub/에 쓰고 7단계가 forward 커밋을 하지 않는다
    assert weekly.DEVIATIONS_KEY not in ctx.memory


def test_unrecorded_week_does_not_stop_later_records(wired, monkeypatch, caplog):
    """스텁 등으로 한 주가 커밋되지 않아도 다음 주 기록은 이어진다. 그 주는 직전 보유를 유지한 것으로 재생한다."""
    s = wired
    gap, t = s.dates[2], s.dates[4]
    recorded = {d: s.backtest[d] for d in s.dates[:4] if d != gap}
    with caplog.at_level(logging.WARNING, logger="test_weekly"):
        ctx = _run(s, t, recorded, monkeypatch)
    _, stub = ctx.written[("positions", t)]
    assert not stub
    assert any(gap in r.getMessage() and "직전 보유를 유지" in r.getMessage() for r in caplog.records)


def test_no_record_yet_starts_from_cash_without_building_the_engine(wired, monkeypatch, caplog):
    """positions_from 기록부터 커밋되지 않았으면 아직 현금 100%다."""
    s = wired
    monkeypatch.setattr(weekly, "build_engine", lambda *a, **k: pytest.fail("기록이 없는데 엔진을 만들었다"))
    with caplog.at_level(logging.WARNING, logger="test_weekly"):
        ctx = _run(s, s.dates[2], {}, monkeypatch)
    from_cash = weekly.buffer_w_raw(ctx.memory[weekly.MEMORY_KEY], set(), s.buffer)
    pd.testing.assert_series_equal(ctx.memory["w_raw"], from_cash)
    messages = [r.getMessage() for r in caplog.records]
    assert any(f"{s.dates[0]}, {s.dates[1]}" in m and "현금 100%에서 시작" in m for m in messages)
    assert not any("직전 보유를 유지" in m for m in messages)  # A #36 리뷰: 재생하지 않는데 재생한다고 하지 않는다


def test_stub_engine_inputs_give_stub_outputs(wired, monkeypatch):
    """A #36 리뷰: 신호 입력은 진짜인데 엔진 입력(prices 등)만 스텁이면 스텁 가격으로 재생하지 않는다."""
    s = wired
    monkeypatch.setattr(weekly, "build_engine", lambda *a, **k: pytest.fail("스텁 가격으로 엔진을 만들었다"))
    monkeypatch.setattr(weekly, "M9_ENGINE_ENTRY", lambda *a: pytest.fail("스텁 실행인데 M9 체결 지도를 불렀다"))
    monkeypatch.setattr(weekly, "RECORDED_POSITIONS_ENTRY", lambda t: {d: s.backtest[d] for d in s.dates[:2]})
    t = s.dates[2]
    ctx = TableCtx(date=t, tables=s.tables, log=logging.getLogger("test_weekly"))
    ctx.stubbed.append("1(M1)")
    ctx.stub_tables.add("prices")
    weekly.m4_step(ctx)
    weekly.m8_step(ctx)
    df, stub = ctx.written[("positions", t)]
    assert ctx.memory["w_raw"].empty and stub and df.empty
    # 기록이 없어 재생하지 않는 주는 엔진 입력이 스텁이어도 상관없다 (첫 기록)
    first = TableCtx(date=s.dates[0], tables=s.tables, log=logging.getLogger("test_weekly"))
    first.stub_tables.add("prices")
    weekly.m4_step(first)
    assert not first.memory["w_raw"].empty


def test_signal_date_must_be_in_the_index(wired):
    """A #36 리뷰: 1단계가 t의 index 행을 못 넣으면 그 주 신호일이 t 전날로 잡힌다. 계산하지 않고 멈춘다."""
    t = wired.dates[1]
    idx = wired.tables["index"]
    tables = {**wired.tables, "index": idx.loc[pd.to_datetime(idx["date"]) != pd.Timestamp(t)]}
    ctx = TableCtx(date=t, tables=tables, log=logging.getLogger("test_weekly"))
    with pytest.raises(ValueError, match=f"index에 신호일 {t}의 코스피 200 종가가 없다"):
        weekly.m4_step(ctx)


def test_stub_inputs_give_stub_outputs(wired):
    ctx = TableCtx(date=wired.dates[0], tables=wired.tables, log=logging.getLogger("test_weekly"))
    ctx.stubbed.append("3(M3a·M3b)")
    ctx.stub_tables.add("signals")
    weekly.m4_step(ctx)
    weekly.m8_step(ctx)
    df, stub = ctx.written[("positions", wired.dates[0])]
    assert ctx.memory["w_raw"].empty and stub and df.empty
    real = TableCtx(date=wired.dates[0], tables=wired.tables, log=logging.getLogger("test_weekly"))
    real.memory["w_raw"] = pd.Series(dtype="float64")
    with pytest.raises(RuntimeError, match="스텁 실행도 아니다"):
        weekly.m8_step(real)


class _Scripted:
    """run_to_signal_close 호출을 기록하고, 정해 둔 보유를 돌려주는 엔진."""

    def __init__(self, held):
        self.held, self.calls = held, []

    def initial_state(self):
        return 0

    def run_to_signal_close(self, state, targets, signal_date, next_signal_date):
        self.calls.append((state, dict(targets), signal_date, next_signal_date))
        return SimpleNamespace(state=state + 1, weights_obs=self.held)


def _pos(date, weights):
    rows = [*weights.items(), ("CASH", 1.0 - sum(weights.values()))]
    return pd.DataFrame({"date": pd.Timestamp(date), "ticker": [r[0] for r in rows],
                         "target_weight": [r[1] for r in rows], "constructor": "buffer", "exposure": 1.0, "regime_mc": 0})


def test_replay_chains_recorded_targets_and_keeps_actual_holdings():
    """목표에 있었지만 체결되지 않은 종목(정지·상한가)은 H에 없다. 상태는 호출마다 이어진다."""
    rec = {"2026-11-20": _pos("2026-11-20", {"000001": 0.1, "000002": 0.1}), "2026-11-27": _pos("2026-11-27", {"000001": 0.1})}
    eng = _Scripted({"000001": 0.12, "000002": 0.0})
    held = weekly.replay_holdings(eng, rec, ["2026-11-20", "2026-11-27"], "2026-12-04")
    assert held == {"000001"}
    assert eng.calls == [(0, {"000001": 0.1, "000002": 0.1}, "2026-11-20", "2026-11-27"),
                         (1, {"000001": 0.1}, "2026-11-27", "2026-12-04")]


def test_replay_holds_through_unrecorded_signal_dates():
    """기록이 없는 11/27에는 체결하지 않는다. 11/20 목표를 12/4 종가까지 들고 간다 (C evaluate_forward_record와 같다)."""
    rec = {d: _pos(d, {"000001": 0.1}) for d in ("2026-11-20", "2026-12-04")}
    eng = _Scripted({"000001": 0.1})
    held = weekly.replay_holdings(eng, rec, ["2026-11-20", "2026-11-27", "2026-12-04"], "2026-12-11")
    assert held == {"000001"}
    assert [c[2:] for c in eng.calls] == [("2026-11-20", "2026-12-04"), ("2026-12-04", "2026-12-11")]
    assert [c[0] for c in eng.calls] == [0, 1]
    first_missing = _Scripted({"000001": 0.1})
    weekly.replay_holdings(first_missing, {"2026-11-27": _pos("2026-11-27", {"000001": 0.1})},
                           ["2026-11-20", "2026-11-27"], "2026-12-04")
    assert [c[2:] for c in first_missing.calls] == [("2026-11-27", "2026-12-04")]  # 현금 100%에서 11/27부터
    assert weekly.replay_holdings(_Scripted({"000001": 0.1}), {}, ["2026-11-20"], "2026-11-27") == set()


def test_replay_rejects_records_off_the_signal_dates():
    rec = {d: _pos(d, {"000001": 0.1}) for d in ("2026-11-20", "2026-11-27")}
    with pytest.raises(ValueError, match="신호일이 아니다: \\['2026-11-27'\\]"):
        weekly.replay_holdings(_Scripted({}), rec, ["2026-11-20"], "2026-11-27")  # t 이후 기록
    with pytest.raises(ValueError, match="다른 날짜"):
        weekly.replay_holdings(_Scripted({}), {"2026-11-20": _pos("2026-11-27", {})}, ["2026-11-20"], "2026-11-27")


@pytest.mark.parametrize("entry", ["src.pipeline.forward_test:no_such_function", "src.no_such_module:f"])
def test_unresolvable_entry_says_which_function_is_missing(entry):
    with pytest.raises(RuntimeError, match="D 지난 forward positions 함수 .*를 찾지 못했다"):
        weekly._resolve(entry, "D 지난 forward positions")


@pytest.mark.parametrize(
    ("constant", "entry", "params"),
    [
        ("M9_COOLDOWN_ENTRY", "src.events.alerts:cooldown_at", ["t"]),  # PM PR #34
        ("M9_ENGINE_ENTRY", "src.events.alerts:engine_alerts", ["first_signal", "t"]),  # PM PR #34
        ("RECORDED_POSITIONS_ENTRY", "src.pipeline.forward_test:past_positions", ["t"]),  # D PR #31
    ],
)
def test_entries_are_the_agreed_functions(constant, entry, params):
    """담당자와 정한 이름 (11/9). 담당 PR이 병합되면 인자 이름까지 확인한다."""
    assert getattr(weekly, constant) == entry
    try:
        fn = weekly._resolve(entry, constant)
    except weekly.EntryNotFound:
        pytest.skip(f"{entry} 담당 PR 병합 전")
    assert list(inspect.signature(fn).parameters) == params


def test_signal_dates_are_last_trading_days_of_each_week():
    days = pd.bdate_range("2026-11-16", "2026-12-04").drop(pd.Timestamp("2026-11-27"))  # 11/27 휴장 가정
    assert weekly.signal_dates_between(days, "2026-11-20", "2026-12-04") == ["2026-11-20", "2026-11-26"]
