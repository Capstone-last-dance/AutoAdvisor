# 공시 시각·종목 매핑 계약

2026-11-03 현재 [PR #2](https://github.com/Capstone-last-dance/AutoAdvisor/pull/2) C 재검토에 따라 DEC-09 (2) 종목 매핑을 다시 검토 중이다. (3) purge는 11/3에 신호일 전체 제거로 확정했다. 라벨 purge 구현은 C 소유다. [결정 기록](decisions.md)

- PM 정제에서는 접수일에 유효한 회사코드↔종목코드 연결 근거를 확보할 때까지 `ticker`를 결측으로 둔다. `tickers`의 상장 기간과 수집 당시의 `stock_code`는 별도 진단에 쓰며 원본 관측을 보존한다.
- `build --publish` 및 `publish_disclosures(..., dry_run=True)`는 공용 스키마 검증을 통과해도 회사↔종목 연결 근거가 없는 현재 결과를 거부한다. 종목별 피처·M9 연동은 시점 매핑 검증 후 재개한다.

## 11/2 DEC-09 당시 결정

- 공개 스키마(`PUBLIC_COLUMNS`): `rcept_no, rcept_date, rcept_dt(결측 허용), available_at(필수), ticker(결측 허용), corp_code, corp_name, report_nm, report_nm_norm`. `flr_nm`은 진단 결과(`EXTENDED_COLUMNS`)에만 둔다.
- 종목코드: A의 `tickers`에서 `listing_date ≤ 접수일 ≤ delisting_date`(상장 중이면 상한 없음)인 경우에만 관측 코드를 붙인다. tickers 없이 만든 결과는 공개하지 않는다(결측이 '비상장'인지 '근거 없음'인지 구분할 수 없으므로).
- 공개 저장: extended 결과만 공용 I/O(`validate`·`write_table`)로 쓴다. v03 추정 시각 결과는 공개하지 않는다. D의 `io.py`가 DEC-09 스키마를 받기 전에는 공용 검증에서 거부된다.
- 뉴스: `published_at`(실제 발행 시각, 모르면 결측)과 `available_at`(필수)을 둔다. 날짜만 아는 기사는 그날 23:59:59 KST다.

아래는 11/2 오전에 낸 PM 안이며, 5·6번은 위 확정 내용으로 바뀌었다.

## PM 안 (11/2 오전)

1. 실제 접수 시각을 모르면 `rcept_dt`는 결측이다. 날짜 기반 추정 시각으로 채우지 않는다.
2. 사용 가능 시점은 `available_at`으로 분리한다. 현재 날짜 전용 원본의 내부 후보값은 `max(rcept_dt 원본 날짜, rcept_no 앞 8자리 날짜) 23:59:59 KST`다. 당일 15:30 신호에는 들어가지 않는다.
3. 명세의 '15:30 이전'을 보수적으로 해석해 **`available_at < cutoff_at`**으로 비교한다. 같은 시각은 제외하고 `.date()` 비교는 금지한다. cutoff는 실제 실행 시각이 아니라 `calendar.signal_cutoff_time`으로 만든 신호일의 시간대 포함 시각이다. YAML의 15:30 값과 거래일/피처 기간 설정은 바꾸지 않았다.
4. 날짜만 아는 자료는 다음 신호일의 컷오프가 지나야 후보가 된다. 이 선택은 시각 적격성만 확인하며 거래일 달력·이벤트 lookback·기간 완전성·매핑 적격성을 대신하지 않는다.
5. 과거 유효 근거가 없는 ticker는 결측으로 유지한다. 현재/최초/최신 관측의 stock_code로 과거를 채우지 않는다. 원본 값·수집 시각과 안정적인 corp_code를 보존해 검증된 매핑을 추후 연결한다.
6. 현재 공개 저장은 계속 거부한다. 공동 확정과 D의 공용 I/O 연결을 코드로 반영하고 C 검증을 통과하기 전에는 파일·플래그로 차단을 해제하지 않는다.

## 분리된 내부 스키마

진단 build/CLI의 기본은 `extended`다. `v03`는 과거 결과 대조를 위해 명시적으로 선택하는 진단 호환 모드이며 공개할 수 없다.

| 필드 | 의미·자료형 |
|---|---|
| rcept_no | 14자리 접수번호 문자열 |
| rcept_date | 원본 접수 날짜, datetime64[ns] 자정; 실제 시각 아님 |
| rcept_dt | 실제 접수 시각, KST datetime64[ns] nullable; 현재 원천에서는 결측 |
| available_at | 이용 가능 시각 후보, KST datetime64[ns], 결측 추정 금지 |
| ticker | 검증된 과거 매핑만 허용할 6자리 문자열; 현재는 결측 |
| corp_code | 원본 회사코드 문자열, 매핑 연결용 |
| corp_name, flr_nm | 수집 시 관측한 표시 정보; 과거 동일 이름이나 시점 적격성을 보증하지 않음 |
| report_nm, report_nm_norm | 원본 보고서명과 기존 규칙의 정규화 결과 |

이 10열은 **공동 확정 전 진단 스키마**다. D의 현재 `disclosures` 검증 스키마가 이 열을 지원한다고 가정하지 않으며, 기존 6열 공개 경로를 우회해 쓰지 않는다.

## 구현·검증 경계

`src/data/disclosures.py`의 `filter_disclosures_before(frame, cutoff_at=...)`는 시각을 분리한 입력에 정확한 timestamp 비교를 적용한다. 시간대가 없는 cutoff/열, 이용 가능 시각 결측, 실제 접수보다 빠른 이용 가능 시각은 거부한다. aware UTC 입력도 같은 순간으로 비교하고 결과를 KST로 정규화한다. 입력을 변경하지 않으며 결과도 진단용이다. legacy 6열 입력을 접수 시각처럼 읽어 선택하지 않는다.

`tests/unit/test_disclosures_contract.py`는 실제 합성 archive→build→선택에서 당일 제외·다음 신호 포함, 15:30 직전/동일/직후·나노초 경계·UTC 동치·결측/순서 오류·입력 불변을 검증한다. 기존 공개 거부 테스트도 유지한다. 이는 PM 후보 구현의 검증이며 C/D 승인이나 실제 원천에서 정확한 시각을 확보한 증거가 아니다.

## 리뷰 요청 범위

- **C:** 실제/추정 시각 분리, 엄격한 이전(`<`) 및 소비자 비교 규칙, 과거 매핑 미검증 시 결측 처리가 평가 입력 계약에 맞는지 확인한다. 라벨 purge의 나머지 DEC-09 항목은 C 소유 설계에서 별도로 확인한다.
- **D:** 합의 후 `read_table`/`validate`/`write_table`의 공시 스키마에 nullable 실제 시각·별도 이용 가능 시각을 반영할 접점을 확인한다. 이 PR은 D 소유 io.py를 대신 구현하지 않는다.
- **A/C 연결 후속:** 회사코드↔종목코드의 유효 기간과 당시 알려진 근거가 있는 매핑 입력이 필요하다. 기존 tickers.parquet에 그런 이력 열이 존재한다고 가정하지 않는다. 증거·유효 기간·관측 시점의 공급 계약을 별도로 확인하기 전 매핑을 채우지 않는다.

PM은 이 보수안을 제안으로 선택했다. 공동 검토 결과는 decisions.md에 별도로 기록하고, 승인되지 않은 부분을 '합의 완료'로 표시하지 않는다. 개인 작업 이력·원본 데이터는 이 문서에 포함하지 않는다.
