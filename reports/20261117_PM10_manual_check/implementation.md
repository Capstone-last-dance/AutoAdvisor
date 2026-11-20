# PM-10 원문 판정 구현 후보

2026-11-17 KST · 기준 커밋 `bd686ca26bb095efbf72ce29cd744cc160618051`

격리된 프로젝트 사본에 M9 원문 판정 경로를 구현했다. 검토용 패치는 `src/events/alerts.py`, `src/events/originals.py`, `src/events/reviewed.py`, `tests/events/test_reviewed.py`를 포함한다. [패치](../../data/interim/pm10-reviewed-alerts-20261117/pm10_reviewed_alerts.patch)

실제 정정·철회 원문과 11/9~11/16 일간·백필 재생을 추가로 검증했다. 운영 적용 보류 사유와 최종 테스트 수는 [실자료 재생 결과](operational_validation.md)에 있다. 아래 검증 목록은 최초 격리 검증 당시의 기록이다.

## 판정 입력과 시점

| 검증 축 | 구현 |
|---|---|
| 공시 버전 | OpenDART ZIP에서 요청한 접수번호의 본문을 찾고 SHA-256을 기록한다. 판정에는 해당 버전에서 확인한 제목·공개 시각·원문의 사실을 사용한다. T6는 같은 버전의 신주 수와 주당가로 금액을 계산한다. 정정·철회는 공개 시각부터 적용한다. 같은 시각에 가용해진 연결 버전은 명시적 계보의 마지막 버전을 평가하고 마지막 버전이 여럿이면 보류한다. |
| 사건 연결 | 당시 같은 회사와 상장 구간 안에서 확인된 원 공시 참조를 받는다. 중복·후속은 최초 경보에 연결한다. 쿨다운 중에도 별개로 확인된 사건은 새 경보가 된다. T3 조회 답변은 제목에 혐의어가 없어도 후보에 넣고, 연결된 원 조회와 회사 답변의 구체적 확인 사실을 요구한다. |
| 종목 식별 | 확인된 회사·종목·상장 구간과 그 식별 근거의 가용 시각을 입력받는다. 코드가 재사용돼도 공시 당시의 법인을 구분하고 유효 구간이 겹쳐 식별이 모호하면 보류한다. T6 분모는 직전 KRX 거래일에 같은 발행 법인의 같은 상장 구간에 대응하는 `prices.mcap`이다. |

`ReviewBook`에는 접수번호별 `DocumentReview`, 검증된 `ListingInterval`, 필요할 때 KIND 원천 메타데이터를 넣는다. `DocumentReview`의 판정 구분·본문 근거·사건 참조는 확인된 원문 사실을 입력하는 계약이다. T1의 두 가지 KIND 서식과 DART 소속부변경, T6의 표본 증자 표는 원문에서 읽는다. T3 등 정형 필드로 확인할 수 없는 사건 사실은 검토자가 입력한다.

`build_alerts(..., review=book, prices=...)`가 공통 판정을 수행한다. `daily_alerts(d, review=book)`과 `backfill_alert_files(first, last, review=book)`은 보관 공시와 원문 검토가 모두 완료된 날짜를 처리한다. 검토용 백필은 `_stub/alerts/`에 기록하며 이미 발행한 경보 행과 최초 `generated_at`을 보존한다. 검토가 일부만 끝난 자료는 `preview_reviewed_alerts`로 확인할 수 있다. 리뷰 입력 없이 호출하는 기존 실행은 현재 운영 경로를 사용한다.

## 검증

- 현행 `main`에서 `git apply --check --whitespace=error-all`로 패치 적용 가능성을 확인했다.
- 보관된 55행, 매일유업 조회 답변 2건, 코아스 KIND 공시 1건을 재생했다. 보관 KIND 표본 7건의 효력일과 DART 신규 지정 2건을 확인했다. T6 ZIP 10건을 접수번호별로 읽었으며 코스모신소재는 최초 예정가 기준 **34.1613%**, T6 미판정은 기존 세 건과 일치했다. [표본 재생 CSV](../../data/interim/pm10-reviewed-alerts-20261117/sample_replay.csv)
- 신풍제약 후속 판결은 이전 공시의 원문이 이 재생 입력에 없어 관계 확정 전 보류다. 대유플러스의 두 번째 부도도 같은 지급제한 아래 다른 어음이 새 사건인지 보류했다. 기존 55행 CSV의 유형 매칭 판정은 별도로 보존했다.
- `tests/events`와 `tests/pipeline/test_run_daily_alerts.py`에서 145개 및 하위 검사 6개가 통과했다. 정확히 10%인 증자, 휴장일 직전 거래일, 같은 가용 시각의 철회, 후속 보고서명 소급, 쿨다운 안팎의 사건 구분, 종목코드 재사용, 같은 보관 입력을 쓴 일간·백필 결과와 최초 생성 시각 일치를 포함한다.
- 추가 OpenDART API 호출 0회. 코스모신소재 후속 정정은 기존 `014` 응답 이후 원문을 확보하지 못해 정정 전후 동작은 합성 자료로 검증했다. 검토 사본에서 표본 재생과 테스트를 실행했다.

## 재현과 운영 의존성

`AutoAdvisor` 루트에서 패치 확인 명령을 실행할 수 있다.

```powershell
git apply --check --whitespace=error-all data/interim/pm10-reviewed-alerts-20261117/pm10_reviewed_alerts.patch
```

`data/interim/pm10-reviewed-alerts-20261117/candidate/`에서 아래 명령으로 격리 사본을 검증한다.

```powershell
..\..\..\..\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/events tests/pipeline/test_run_daily_alerts.py
..\..\..\..\.venv\Scripts\python.exe -B -X utf8 verify_sample.py
```

접수일 기준 종목 매핑과 사건 참조가 검증된 원문을 운영 대상 전체에서 확보해야 한다. KIND 공개 식별자는 9.3절의 `KIND:<acptno>` 형식으로 검증했다. 실제 T6 정정·철회와 일간·백필 재생, 최신 가격 공백은 [실자료 재생 결과](operational_validation.md)에 기록했다.
