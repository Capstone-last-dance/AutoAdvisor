"""M1 원천 수집, KRX Open API 경로 (A-02). 응답 본문을 data/raw/{수집일}/openapi/{서비스}/{날짜}.json 에 그대로 저장한다.
이미 있는 파일은 덮어쓰지 않으므로 같은 명령으로 이어받을 수 있다. 빈 목록(휴장일)도 파일로 남겨 다시 호출하지 않는다.

사용:
  python3 -m src.data.collect_openapi --start 2010-01-04 --end 2026-10-30 --snapshot 2026-10-30 \\
      --daily-limit <공식 확인한 일 한도> --services kospi_dd_trd stk_bydd_trd ksq_bydd_trd \\
      --verified kospi_dd_trd stk_bydd_trd ksq_bydd_trd [--dry-run]

순서: 거래일 캘린더 서비스(--calendar-service)를 먼저 받고, 그 결과의 비어 있지 않은 날을 거래일로 삼아 나머지를 받는다 (Q-14).
종료 코드: 0 완료, 1 실패로 중단, 2 인증 거부(401/403), 3 일 한도 도달(다음 날 같은 명령으로 이어받기).
"""
import argparse
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from src.data.calendar import signal_dates
from src.data.krx_client_paths import ROOT
from src.data.openapi_client import (DATA_BLOCK, SERVICES, OpenApiAuthError, OpenApiClient, OpenApiError, Quota,
                                     QuotaExceeded, load_key)

MAX_CONSECUTIVE_FAILURES = 3


def weekdays(start: date, end: date) -> list:
    return [start + timedelta(n) for n in range((end - start).days + 1) if (start + timedelta(n)).weekday() < 5]


def raw_path(out: Path, service: str, d: date) -> Path:
    return out / "openapi" / service / f"{d:%Y-%m-%d}.json"


def write_raw(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)   # os.rename은 Windows에서 path가 이미 있으면 실패한다(WinError 183). replace는 둘 다 덮어쓴다


def calendar_days(out: Path, service: str, days, idx_name: str | None = None) -> list:
    """거래일 = 캘린더 서비스 응답이 비어 있지 않은 날. idx_name을 주면 그 지수의 종가가 있는 날만 (Q-14의 코스피 200 기준).
    IDX_NM, CLSPRC_IDX는 KRX 시리즈 Spec 기준이고 다른 지수 서비스에서는 확인 전이다."""
    missing = [d for d in days if not raw_path(out, service, d).exists()]
    if missing:
        raise OpenApiError(f"캘린더 서비스 {service}에 받지 못한 날이 {len(missing)}일 있습니다(예: {missing[0]}). 같은 명령으로 다시 실행하세요")
    res = []
    for d in days:
        p = raw_path(out, service, d)
        rows = json.loads(p.read_text(encoding="utf-8"))[DATA_BLOCK]
        if idx_name is not None:
            rows = [r for r in rows if r.get("IDX_NM") == idx_name and r.get("CLSPRC_IDX") not in (None, "", "-")]
        if rows:
            res.append(d)
    return res


def dates_for(schedule: str, days_all: list, trading: list | None) -> list:
    if schedule == "weekdays":
        return days_all
    if trading is None:
        raise OpenApiError("거래일 캘린더가 아직 없습니다. --calendar-service를 먼저 수집하세요")
    if schedule == "trading":
        return trading
    if schedule == "signal":
        return [d.date() for d in signal_dates(trading)]
    raise ValueError(schedule)


def check_runnable(names, verified):
    unknown = [n for n in names if n not in SERVICES]
    if unknown:
        raise OpenApiError(f"알 수 없는 서비스: {unknown}. 가능: {sorted(SERVICES)}")
    blocked = [n for n in names if not SERVICES[n].verified and n not in set(verified)]
    if blocked:
        raise OpenApiError(f"Spec 확인 전인 서비스: {blocked}. 서비스별 Spec으로 경로와 필드를 확인한 뒤 --verified로 명시하세요 (R-08)")


def pending(out: Path, service: str, dates) -> list:
    return [d for d in dates if not raw_path(out, service, d).exists()]


def fetch_all(client: OpenApiClient, out: Path, service: str, dates, interval: float, failures: list, sleep=time.sleep):
    """남은 날짜를 순서대로 받는다. 연속 실패가 MAX_CONSECUTIVE_FAILURES에 닿으면 OpenApiError로 중단한다."""
    svc, streak, done = SERVICES[service], 0, 0
    for d in pending(out, service, dates):
        try:
            write_raw(raw_path(out, service, d), client.get(svc, d))
            streak, done = 0, done + 1
        except (OpenApiAuthError, QuotaExceeded):
            raise
        except OpenApiError as e:
            failures.append({"service": service, "date": f"{d:%Y-%m-%d}", "error": str(e)[:200]})
            streak += 1
            if streak >= MAX_CONSECUTIVE_FAILURES:
                raise OpenApiError(f"{service}: 연속 {streak}회 실패로 중단") from e
        sleep(interval)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-04")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--snapshot", default=str(date.today()), help="data/raw/{수집일}. 이어받기는 같은 값을 쓴다")
    ap.add_argument("--services", nargs="+", required=True, help=f"{sorted(SERVICES)}")
    ap.add_argument("--verified", nargs="*", default=[], help="Spec으로 경로·필드를 확인했다고 사용자가 명시하는 서비스")
    ap.add_argument("--calendar-service", default="kospi_dd_trd")
    ap.add_argument("--calendar-idx-name", default=None, help="이 지수명(IDX_NM)의 종가가 있는 날만 거래일로 본다. 예: 코스피 200")
    ap.add_argument("--daily-limit", type=int, required=True, help="하루 호출 한도. 공식 확인한 값을 쓴다 (R-08)")
    ap.add_argument("--interval", type=float, default=0.5, help="호출 간 대기(초)")
    ap.add_argument("--dry-run", action="store_true", help="네트워크 호출 없이 남은 호출 수만 센다")
    a = ap.parse_args(argv)

    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    names = list(dict.fromkeys(a.services))
    needs_calendar = any(SERVICES[n].schedule != "weekdays" for n in names if n in SERVICES)
    if needs_calendar and a.calendar_service not in names:
        names.insert(0, a.calendar_service)
    try:
        check_runnable(names, a.verified)
    except OpenApiError as e:
        print(f"시작 불가: {e}")
        return 1
    names.sort(key=lambda n: n != a.calendar_service)   # 캘린더 서비스를 맨 앞으로

    out = ROOT / "data" / "raw" / a.snapshot
    days_all = weekdays(start, end)
    failures, collected, status = [], {}, 0
    t0 = time.time()

    def calendar_ready():
        return all(raw_path(out, a.calendar_service, d).exists() for d in days_all)

    if a.dry_run:
        for n in names:
            if SERVICES[n].schedule == "weekdays":
                dates, note = days_all, ""
            elif calendar_ready():
                dates, note = dates_for(SERVICES[n].schedule, days_all, calendar_days(out, a.calendar_service, days_all, a.calendar_idx_name)), ""
            else:
                dates, note = days_all, " (캘린더 미수집이라 평일 수 기준 상한)"
            print(f"{n}: 남은 호출 {len(pending(out, n, dates))}{note}")
        return 0

    client = OpenApiClient(load_key(ROOT / ".env"), Quota(ROOT / "data" / "raw" / "_openapi_quota", a.daily_limit))
    try:
        trading = None
        for n in names:
            if SERVICES[n].schedule != "weekdays" and trading is None:
                trading = calendar_days(out, a.calendar_service, days_all, a.calendar_idx_name)
            dates = dates_for(SERVICES[n].schedule, days_all, trading)
            collected[n] = fetch_all(client, out, n, dates, a.interval, failures)
            print(f"{n}: 새로 저장 {collected[n]}건 (오늘 호출 {client.quota.calls}/{client.quota.limit})", flush=True)
            if n == a.calendar_service:
                trading = calendar_days(out, n, days_all, a.calendar_idx_name)
    except OpenApiAuthError as e:
        print(f"인증 거부로 중단: {e}\n마이페이지에서 키와 서비스 승인 상태를 확인하세요", flush=True)
        status = 2
    except QuotaExceeded as e:
        print(f"{e}. 다음 날 같은 명령으로 이어받으세요", flush=True)
        status = 3
    except OpenApiError as e:
        print(f"중단: {e}", flush=True)
        status = 1

    meta = {
        "collected_at": datetime.now().isoformat(timespec="seconds"), "start": a.start, "end": a.end,
        "services": names, "saved": collected, "calls_today": client.quota.calls, "daily_limit": a.daily_limit,
        "exit_status": status, "failures": failures,
    }
    (out / "openapi").mkdir(parents=True, exist_ok=True)
    (out / "openapi" / f"meta_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"종료: 상태 {status}, 실패 {len(failures)}건, {time.time() - t0:.0f}초")
    return 1 if (status == 0 and failures) else status


if __name__ == "__main__":
    raise SystemExit(main())
