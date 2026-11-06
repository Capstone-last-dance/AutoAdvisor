"""src/common/io.py 단위 테스트 (작업 D-02, 6.12절 완료 기준 4).

실행: python -m pytest tests/common
테스트 데이터는 형식 확인용 임의 값이며 실제 시장 데이터가 아니다.
"""

import datetime as dt
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import io

DAY = "2026-11-06"
TS = pd.Timestamp(DAY)
KST = "Asia/Seoul"


def sample(name: str) -> pd.DataFrame:
    """테이블별로 스키마를 만족하는 최소 예시."""
    tickers = ["005930", "000660"]
    d = pd.to_datetime([DAY, DAY])
    kst = pd.to_datetime(["2026-11-06 09:10", "2026-11-06 14:00"]).tz_localize(KST)
    frames = {
        "prices": dict(date=d, ticker=tickers, market=["KOSPI", "KOSDAQ"],
                       close=[1.0, 2.0], close_raw=[1.0, 2.0],
                       volume=np.array([10, 20], dtype="int64"), value=np.array([100, 200], dtype="int64"),
                       mcap=np.array([1000, 2000], dtype="int64"), listed_days=np.array([300, 400], dtype="int32")),
        "index": dict(date=d, index_code=["KOSPI200", "KOSPI200"], close=[300.0, 301.0]),
        "tickers": dict(ticker=tickers, name=["가", "나"], market=["KOSPI", "KOSDAQ"],
                        listing_date=pd.to_datetime(["2000-01-04", "2001-01-04"]),
                        delisting_date=pd.to_datetime([None, "2020-01-02"]),
                        security_type=["common", "common"]),
        "status": dict(date=d, ticker=tickers, is_admin=pd.array([False, None], dtype="boolean"), is_halted=[False, True]),
        "universe": dict(date=d, ticker=tickers, mcap_rank=np.array([1, 2], dtype="int32"),
                         in_universe=[True, True]),
        "delisted": dict(ticker=tickers, delist_date=d, last_price=[100.0, np.nan]),
        "sector": dict(ticker=tickers, sector=["전기전자", "미분류"],
                       valid_from=pd.to_datetime(["2010-01-04", "2010-01-04"]),
                       valid_to=pd.to_datetime([None, None])),
        "disclosures": dict(rcept_no=["1", "2"], rcept_dt=kst, ticker=["005930", None],
                            corp_name=["가", "비상장"], report_nm=["주식 소각 결정", "기타"],
                            report_nm_norm=["주식소각결정", "기타"]),
        "news": dict(news_id=["n1", "n2"], published_at=kst, ticker=tickers, title=["a", "b"],
                     url=["https://a", "https://b"], source=["bigkinds", "naver"], sentiment=[0.5, np.nan]),
        "labels": dict(date=pd.to_datetime([DAY] * 3), ticker=["005930", "005930", "000660"],
                       h=np.array([1, 20, 20], dtype="int8"), ret=[0.01, 0.02, -0.2],
                       er=[0.0, 0.01, -0.18], y_rank=[0.5, 0.6, 0.1],
                       y_crash=pd.array([None, 0, 1], dtype="Int8")),
        "features": dict(date=d, ticker=tickers, mom_12_1=[0.2, 0.8],
                         mom_12_1_na=np.array([0, 1], dtype="int8"), penny_flag=np.array([0, 0], dtype="int8")),
        "signals": dict(date=d, ticker=tickers, rank_score=[0.9, 0.1],
                        rank=np.array([1, 2], dtype="int32"), crash_prob=[0.1, 0.7],
                        crash_flag=np.array([0, 1], dtype="int8"), model_version=["baseline", "baseline"]),
        "positions": dict(date=pd.to_datetime([DAY] * 3), ticker=["005930", "000660", "CASH"],
                          target_weight=[0.1, 0.1, 0.8], constructor=["buffer"] * 3,
                          exposure=[0.9] * 3, regime_mc=np.array([0, 0, 0], dtype="int8")),
        "alerts": dict(date=d, ticker=tickers, trigger_type=["T1", "T6"], rcept_no=["1", "2"],
                       report_nm=["관리종목지정", "유상증자결정"], rcept_dt=kst),
        "forward_log": dict(signal_date=pd.to_datetime([DAY]), commit_hash=["abc1234"],
                            signal_source=["baseline"], has_positions=np.array([0], dtype="int8"),
                            label_ready_date=pd.to_datetime(["2026-12-04"]),
                            mean_er_top20=[np.nan], mean_er_flagged=[np.nan],
                            portfolio_return=[np.nan], bm_return=[np.nan]),
    }
    return pd.DataFrame(frames[name])


def evidence_record(**overrides) -> dict:
    rec = {
        "date": DAY, "ticker": "005930", "signal_type": "crash", "summary": "요약 [1]",
        "sources": [{"n": 1, "type": "disclosure", "id": "1", "title": "t", "url": "https://a",
                     "timestamp": "2026-11-06T09:10:00+09:00"}],
        "disclaimer": io.DISCLAIMER,
    }
    rec.update(overrides)
    return rec


class TempRootTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._root = io.ROOT
        io.ROOT = self.tmp

    def tearDown(self):
        io.ROOT = self._root
        shutil.rmtree(self.tmp, ignore_errors=True)


class PathRuleTest(unittest.TestCase):
    def rel(self, *args):
        return io._relative_path(*args).as_posix()

    def test_paths(self):
        self.assertEqual(self.rel("signals", DAY, None), "signals/2026-11-06.parquet")
        self.assertEqual(self.rel("positions", None, "20261116_B_m8_v1"), "positions/wf_20261116_B_m8_v1.parquet")
        self.assertEqual(self.rel("prices", None, None), "data/processed/prices.parquet")
        self.assertEqual(self.rel("labels", None, None), "data/processed/labels.parquet")
        self.assertEqual(self.rel("forward_log", None, None), "forward_log.csv")

    def test_invalid_arguments(self):
        cases = [
            ("unknown", None, None),             # 없는 이름
            ("signals", DAY, "20261116_B_x"),    # date와 run_id 동시
            ("signals", None, None),             # 날짜별 폴더인데 둘 다 없음
            ("prices", DAY, None),               # data/processed 단일 파일에 date
            ("forward_log", DAY, None),
            ("signals", "2026/11/06", None),     # 날짜 형식
            ("signals", "2026-02-30", None),     # 없는 날짜
            ("signals", None, "m8_v1"),          # run_id 형식
            ("signals", None, "20261116_E_x"),   # 없는 역할 ID
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                io._relative_path(*args)


class RoundTripTest(TempRootTest):
    def test_every_table_round_trips(self):
        for name in io.SCHEMAS:
            with self.subTest(name=name):
                df = sample(name)
                date = DAY if name in io.DATED_FOLDERS else None
                path = io.write_table(name, df, date=date)
                self.assertTrue(path.exists())
                back = io.read_table(name, date=date)
                pd.testing.assert_frame_equal(back, df, check_dtype=False, check_index_type=False)
                io.validate(name, back)

    def test_ticker_keeps_leading_zero(self):
        io.write_table("signals", sample("signals"), date=DAY)
        self.assertEqual(io.read_table("signals", date=DAY)["ticker"].tolist(), ["005930", "000660"])

    def test_python_date_objects_are_accepted(self):
        df = sample("prices")
        df["date"] = [dt.date(2026, 11, 6)] * 2
        io.write_table("prices", df)
        self.assertTrue(pd.api.types.is_datetime64_dtype(io.read_table("prices")["date"]))

    def test_run_id_file(self):
        path = io.write_table("signals", sample("signals"), run_id="20261122_A_gate_v1")
        self.assertEqual(path, self.tmp / "signals" / "wf_20261122_A_gate_v1.parquet")
        io.read_table("signals", run_id="20261122_A_gate_v1")

    def test_stub_goes_under_stub_folder(self):
        path = io.write_table("signals", sample("signals"), date=DAY, stub=True)
        self.assertEqual(path, self.tmp / "_stub" / "signals" / "2026-11-06.parquet")
        self.assertFalse((self.tmp / "signals" / "2026-11-06.parquet").exists())

    def test_stub_read_is_separate_from_real(self):
        io.write_table("signals", sample("signals"), date=DAY, stub=True)
        self.assertEqual(len(io.read_table("signals", date=DAY, stub=True)), 2)
        with self.assertRaises(FileNotFoundError):
            io.read_table("signals", date=DAY)  # 스텁은 실제 경로에서 읽히지 않는다

    def test_missing_file_raises_file_not_found(self):
        # 호출하는 쪽(PM-04 등)은 FileNotFoundError로 '새 파일'을 판단한다
        for name, date in [("disclosures", None), ("signals", DAY), ("forward_log", None)]:
            with self.subTest(name=name), self.assertRaises(FileNotFoundError):
                io.read_table(name, date=date)
        with self.assertRaises(FileNotFoundError):
            io.read_evidence(DAY)

    def test_empty_forward_log_file(self):
        (self.tmp / "forward_log.csv").write_bytes(b"")
        df = io.read_table("forward_log")
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), list(io.SCHEMAS["forward_log"]))

    def test_dated_file_rejects_other_dates(self):
        df = sample("signals")
        df.loc[1, "date"] = pd.Timestamp("2026-11-13")
        with self.assertRaises(io.SchemaError):
            io.write_table("signals", df, date=DAY)

    def test_invalid_frame_is_not_written(self):
        df = sample("signals").drop(columns="rank")
        with self.assertRaises(io.SchemaError):
            io.write_table("signals", df, date=DAY)
        self.assertFalse((self.tmp / "signals" / "2026-11-06.parquet").exists())


class ValidateRejectsTest(unittest.TestCase):
    """6.12절 완료 기준 (4): 9.3절 스키마를 어긴 입력에 예외."""

    def assertRejects(self, name, df):
        with self.assertRaises(io.SchemaError):
            io.validate(name, df)

    def test_every_sample_is_valid(self):
        for name in io.SCHEMAS:
            with self.subTest(name=name):
                io.validate(name, sample(name))

    def test_missing_and_extra_columns(self):
        self.assertRejects("signals", sample("signals").drop(columns="crash_prob"))
        df = sample("signals"); df["note"] = "x"
        self.assertRejects("signals", df)

    def test_ticker_rules(self):
        df = sample("prices"); df["ticker"] = [5930, 660]            # 정수
        self.assertRejects("prices", df)
        df = sample("prices"); df["ticker"] = ["5930", "000660"]     # 5자리
        self.assertRejects("prices", df)
        df = sample("prices"); df["ticker"] = ["CASH", "000660"]     # CASH는 positions만
        self.assertRejects("prices", df)
        df = sample("prices"); df["ticker"] = ["0030r0", "000660"]   # 영문은 대문자만
        self.assertRejects("prices", df)
        df = sample("prices"); df["ticker"] = ["0030R0", "0120G0"]   # 2025년 하반기 이후 영문 포함 코드
        io.validate("prices", df)
        df = sample("news"); df.loc[0, "ticker"] = None              # news는 결측 불가
        self.assertRejects("news", df)

    def test_dtypes(self):
        df = sample("prices"); df["volume"] = df["volume"].astype("float64")
        self.assertRejects("prices", df)
        df = sample("signals"); df["rank"] = df["rank"].astype("int64")
        self.assertRejects("signals", df)
        df = sample("status"); df["is_admin"] = [0, 1]
        self.assertRejects("status", df)

    def test_dates_and_times(self):
        df = sample("prices"); df["date"] = [DAY, DAY]                # 문자열 날짜
        self.assertRejects("prices", df)
        df = sample("prices"); df["date"] = pd.to_datetime(["2026-11-06 15:30"] * 2)
        self.assertRejects("prices", df)                              # 시각 포함
        df = sample("alerts"); df["rcept_dt"] = df["rcept_dt"].dt.tz_localize(None)
        self.assertRejects("alerts", df)                              # 시간대 없음
        df = sample("alerts"); df["rcept_dt"] = df["rcept_dt"].dt.tz_convert("UTC")
        self.assertRejects("alerts", df)                              # KST 아님

    def test_allowed_values_and_bounds(self):
        df = sample("prices"); df["market"] = ["KOSPI", "NYSE"]
        self.assertRejects("prices", df)
        df = sample("alerts"); df["trigger_type"] = ["T1", "T7"]
        self.assertRejects("alerts", df)
        df = sample("signals"); df["crash_flag"] = np.array([0, 2], dtype="int8")
        self.assertRejects("signals", df)
        df = sample("news"); df["sentiment"] = [1.5, 0.0]
        self.assertRejects("news", df)
        df = sample("positions"); df["constructor"] = ["buffer", "rl", "ppo"]
        self.assertRejects("positions", df)

    def test_dec24_status(self):
        df = sample("status"); df["is_admin"] = [False, True]
        io.validate("status", df)                                                   # 결측 없으면 bool도 허용
        df = sample("status"); df["is_admin"] = pd.array([None, None], dtype="boolean")
        io.validate("status", df)                                                   # v2 전까지 전부 결측 허용
        df = sample("status"); df["in_kospi200"] = [True, False]
        self.assertRejects("status", df)                                            # 구성종목 열은 뺐다
        df = sample("status"); df["is_halted"] = pd.array([False, None], dtype="boolean")
        self.assertRejects("status", df)                                            # is_halted는 결측 불가

    def test_nulls(self):
        df = sample("signals"); df["crash_prob"] = [0.1, np.nan]
        self.assertRejects("signals", df)

    def test_labels_y_crash_only_on_h20(self):
        df = sample("labels"); df["y_crash"] = pd.array([0, 0, 1], dtype="Int8")
        self.assertRejects("labels", df)
        df = sample("labels"); df["y_crash"] = pd.array([None, None, 1], dtype="Int8")
        self.assertRejects("labels", df)

    def test_feature_columns(self):
        df = sample("features"); df["my_feature"] = [1.0, 2.0]
        self.assertRejects("features", df)
        df = sample("features"); df["penny_flag_na"] = np.array([0, 0], dtype="int8")
        self.assertRejects("features", df)                            # 원값 피처는 _na 없음
        df = sample("features"); df["mom_12_1_na"] = np.array([0, 2], dtype="int8")
        self.assertRejects("features", df)
        df = sample("features"); df["mom_12_1"] = ["a", "b"]
        self.assertRejects("features", df)

    def test_dec22_no_investor_flows(self):
        """DEC-22: 원천이 없어 investor_flows 테이블·피처(retail/foreign/inst_net_20)를 안 만든다."""
        self.assertNotIn("investor_flows", io.SCHEMAS)
        df = sample("features"); df["retail_net_20"] = [0.1, 0.2]
        self.assertRejects("features", df)

    def test_not_a_dataframe(self):
        self.assertRejects("prices", [{"date": DAY}])


class EvidenceTest(TempRootTest):
    def test_round_trip(self):
        recs = [evidence_record(), evidence_record(ticker="000660", signal_type="rank", sources=[])]
        path = io.write_evidence(DAY, recs)
        self.assertEqual(path, self.tmp / "evidence" / "2026-11-06.jsonl")
        self.assertEqual(io.read_evidence(DAY), recs)

    def test_rejects(self):
        bad = [
            evidence_record(disclaimer="투자 권유 아님"),
            evidence_record(signal_type="buy"),
            evidence_record(ticker="5930"),
            evidence_record(date="2026-11-13"),
            evidence_record(sources=[{"n": 1}]),
            evidence_record(sources=[{**evidence_record()["sources"][0], "timestamp": "2026-11-06T00:10:00+00:00"}]),
            {k: v for k, v in evidence_record().items() if k != "summary"},
        ]
        for rec in bad:
            with self.subTest(rec=rec), self.assertRaises(io.SchemaError):
                io.write_evidence(DAY, [rec])
        self.assertFalse((self.tmp / "evidence" / "2026-11-06.jsonl").exists())

    def test_stub(self):
        path = io.write_evidence(DAY, [evidence_record()], stub=True)
        self.assertEqual(path, self.tmp / "_stub" / "evidence" / "2026-11-06.jsonl")
        self.assertEqual(io.read_evidence(DAY, stub=True), [evidence_record()])
        with self.assertRaises(FileNotFoundError):
            io.read_evidence(DAY)


if __name__ == "__main__":
    unittest.main()
