# 작업 진행 현황

마지막 갱신: 2026-11-03

아래 표는 전체 명세의 작업 ID 단위 상태이며 세부 단계 완료와 구분한다. 개인 작업 기록·실행 로그·수집 원본은 Git에서 제외하고, 이 문서에는 팀 진행 현황과 남은 과제를 기록한다.

| 작업 ID | 담당 | 기한 | 상태 | 근거 | 메모 |
|---|---|---|---|---|---|
| PM-01 | PM 장지원 | 10/28 | 진행 중 | docs/status.md; docs/decisions.md; docs/deviation_log.md; docs/preregistration.md | 명세에서 요구하는 공용 기록 4개 작성. 주간 회의 공지는 미완료 |
| PM-02 | PM 장지원 | 10/30 | 진행 중 | README.md; .env.example; requirements-lock.txt; [PR #2](https://github.com/Capstone-last-dance/AutoAdvisor/pull/2); docs/decisions.md | D-01 설치 재현·README 기존 .env 보호 검증 및 PR 공유 완료. 변수명 OPENDART_API_KEY·NAVER_CLIENT_ID·NAVER_CLIENT_SECRET 유지. 키 공유/교체·팀원별 재현·네이버 인증 확인은 남음 |
| PM-03 | PM 장지원 | 10/30 | 진행 중 | docs/프로젝트_구조_및_역할분담_v0.3.md 부록 3 Q-06·Q-08 | Q-08 문헌·DART 누락 반례·KRX 공식 CSV 2개·KIND 과거/상폐 표시 회사/해제 원문 3개 검증. 표본·실행 기록은 별도 보관. A 공동 확인·전체 기간 상태 데이터 검증 남음 |
| PM-04 | PM 장지원 | 11/2 | 진행 중 | src/data/disclosures.py; [공시 계약](disclosures_contract.md); [PR #2](https://github.com/Capstone-last-dance/AutoAdvisor/pull/2) | 원본 수집·legacy 859,133건 대조 완료. DEC-09 시각 분리·15:30 경계·공용 I/O는 C 재검토에서 확인. 로컬에서 접수일 회사↔종목 연결 근거 없는 ticker를 결측으로 유지하고 공개를 차단했다. 실제 매핑 원천·D 스키마 연동·purge 합의·C 재리뷰 남음 |
| PM-05 | PM 장지원 | 11/6 | 진행 중 | src/data/naver_client.py; src/data/news.py; src/data/collect_news.py; tests/unit/test_news.py | 1회 원본 수집 CLI·키 비노출/무재시도·필수 관측 시각/종목 매핑 정규화 구현. 전체 회귀 167개 테스트 통과. 실제 네이버 인증과 빅카인즈 계정 준비 후 기사 표본 확보 필요. news.parquet·BigKinds 가져오기·최종 뉴스 수집 미완료 |
| PM-06 | PM 장지원 | 11/8 | 대기 | | |
| PM-07 | PM 장지원 | 11/9 | 대기 | | |
| PM-08 | PM 장지원 | 11/16 | 대기 | | |
| PM-09 | PM 장지원 | 11/16 | 대기 | | |
| PM-10 | PM 장지원 | 11/16 | 대기 | | |
| PM-11 | PM 장지원 | 12/4 | 대기 | | |
| PM-12 | PM 장지원 | 12/7 | 대기 | | |
| PM-13 | PM 장지원 | 12/11 | 대기 | | |
| PM-14 | PM 장지원 | 12/11 | 대기 | | |
| PM-15 | PM 장지원 | 12/14 | 대기 | | |
| PM-16 | PM 장지원 | 매주 | 대기 | | |
| PM-17 | PM 장지원 | 11/8 | 진행 중 | docs/프로젝트_구조_및_역할분담_v0.3.md 부록 3 Q-07·Q-09·Q-10·Q-11, 4.9절; docs/decisions.md DEC-10 | 원문 4건·상세 API 2회에서 후속 정정값 소급 위험 확인, 최초 ZIP 2건 확보/수치 대조, 원저자 코드와 가중치 이용 조건 구분, MIT 기반 후보 검토. Q-10 검증 모델/환경 실측·Q-11 날짜표 보완 유지. T6 금액 계약·감성 모델 이용 조건/학습 시점·실제 추론과 D/A/C 접점은 미완료. FILL_ 유지 |
| A-01 | A 김윤하 | 10/30 | 대기 | | |
| A-02 | A 김윤하 | 11/2 | 진행 중 | 11/3 A·C 인계 (`processed_20261103.zip`) | A 보고: 2010-01-04~2026-10-30 핵심 4표 공유. C 실측 보고: 주간 `status`의 관리종목·지수 편입 3열 전부 결측으로 공용 I/O 검증 실패. 일별 `is_halted`와 평가 가격 공백 규칙 확인 필요. `universe` 스텁, `sector`·`investor_flows` 미포함. 11/3 PM 결정: `investor_flows`는 만들지 않음(DEC-22), `sector` v1은 KIND 현재 분류와 상폐 '미분류'(DEC-23), `status`는 일별로 하고 지수 편입 2열 삭제, `is_admin`은 v2까지 결측 허용, `universe` v1은 관리종목 단계 미적용(DEC-24). 정지일 평가 가격은 DEC-25 대기 |
| A-03 | A 김윤하 | 11/2 | 대기 | | |
| A-04 | A 김윤하 | 11/6 | 대기 | | |
| A-05 | A 김윤하 | 11/16 | 대기 | | |
| A-06 | A 김윤하 | 11/22 | 대기 | | |
| A-07 | A 김윤하 | 11/22 | 대기 | | |
| A-08 | A 김윤하 | 11/27 | 대기 | | |
| A-09 | A 김윤하 | 11/30 | 대기 | | |
| A-10 | A 김윤하 | 12/7 | 대기 | | |
| A-11 | A 김윤하 | 기한 없음 | 대기 | | |
| B-01 | B 조윤상 | 11/1 | 완료 | docs/design_B.md; src/portfolio/; [PR #4](https://github.com/Capstone-last-dance/AutoAdvisor/pull/4) | M4·M8 설계와 PPO 골격. 합성 입력으로 학습 루프 테스트 통과, A 리뷰 후 병합 |
| B-02 | B 조윤상 | 11/8 | 완료 | src/portfolio/risk_layer.py; src/portfolio/vol_target_report.py; reports/20261103_B_vol_target/; [PR #11](https://github.com/Capstone-last-dance/AutoAdvisor/pull/11) | vol_target_annual = 0.157263 계산·부록 1 기입, 2011~2015 데이터만 사용 테스트 통과. compute_regime_mc 공개, DEC-19 반영 |
| B-03 | B 조윤상 | 11/13 | 진행 중 | src/portfolio/buffer_rule.py; tests/portfolio/test_buffer_rule.py | 단계 1~5 단위 테스트 통과. PR #8 병합 뒤 main에서 C의 M5 위 합성 시장 실행 테스트 통과(건너뛰지 않음). A 코드 리뷰 승인 남음. 실제 신호(A-04) 실행은 B-05에서 확인 |
| B-04 | B 조윤상 | 11/16 | 완료 | src/portfolio/risk_layer.py; src/portfolio/market_state_report.py; tests/portfolio/test_risk_layer.py; tests/portfolio/test_market_state_report.py; reports/20261104_B_m8_states/; [PR #13](https://github.com/Capstone-last-dance/AutoAdvisor/pull/13) | 불변식·경계 사례 테스트 통과, 2011~2025년 E_t·regime_mc 그림(prereg.yaml 초안 기준). A 승인. 11/9 공식 사전등록에서 쓰는 키가 바뀌면 새 run_id로 다시 만듦 |
| B-05 | B 조윤상 | 11/16 | 대기 | docs/design_B.md 5.1절 | 어댑터 규칙에 DEC-25·26 반영. PR #8(c82fa1b)에서 DEC-25 재개일 처리·정지일 평가와 DEC-28 재사용 코드(옛 회사 청산, 새 회사 매수)를 합성 시장으로 확인. 실제 데이터 확인은 A 보완 데이터와 PR #8 병합 뒤 |
| B-06 | B 조윤상 | 11/23 | 대기 | | |
| B-07 | B 조윤상 | 11/30 | 대기 | | |
| B-08 | B 조윤상 | 12/7 | 대기 | | |
| B-09 | B 조윤상 | 12/7 | 대기 | | |
| B-10 | B 조윤상 | 12/14 | 대기 | | |
| C-01 | C 손석범 | 11/1 | 완료 | [PR #3](https://github.com/Capstone-last-dance/AutoAdvisor/pull/3); [설계](design_C.md) | 4.6·6.6절 라벨·평가·체결 설계 작성 및 병합. 후속 PM 결정은 PR #8에 반영 |
| C-02 | C 손석범 | 11/6 | 진행 중 | [병합 PR #8](https://github.com/Capstone-last-dance/AutoAdvisor/pull/8) | 라벨·홀드아웃 가드 구현은 병합됨. A의 구회사 폐지 보완 데이터로 실제 라벨 전수 검증이 남음 |
| C-03 | C 손석범 | 11/6 | 진행 중 | [병합 PR #8](https://github.com/Capstone-last-dance/AutoAdvisor/pull/8) | 공통 체결·비용·NAV 엔진과 평가지표 구현은 병합됨. B 최종 승인과 B-05 실제 데이터 미리보기에서 781개 신호일 실행·같은 비중의 체결/비용/NAV 일치를 확인함. A 최종본 반영 후 C 평가 실행이 남음 |
| C-04 | C 손석범 | 11/16 | 진행 중 | [병합 PR #8](https://github.com/Capstone-last-dance/AutoAdvisor/pull/8) | G0 검사 선행 구현. forward test의 실제 보유 재생용 `run_to_signal_close`를 추가 중. 개인 행동 프록시·forward test 사후 성과와 실데이터 검증은 남음 |
| C-05 | C 손석범 | 11/23 | 대기 | | |
| C-06 | C 손석범 | 12/4 | 대기 | | |
| C-07 | C 손석범 | 11/30 | 대기 | | |
| C-08 | C 손석범 | 12/7 | 대기 | | |
| C-09 | C 손석범 | 12/7 | 대기 | | |
| C-10 | C 손석범 | 12/7 | 대기 | | |
| C-11 | C 손석범 | 12/11 | 대기 | | |
| C-12 | C 손석범 | 12/14 | 대기 | | |
| D-01 | D 임소현 | 10/30 | 완료 | README.md; requirements.txt; requirements-lock.txt | 설치 절차·버전 고정, PM 재현 확인 |
| D-02 | D 임소현 | 11/2 | 완료 | src/common/io.py; tests/common/test_io.py; PR #6 | io.py·스키마 검증, PM 리뷰 승인 후 병합 |
| D-03 | D 임소현 | 11/6 | 완료 | src/pipeline/run_weekly.py; tests/pipeline/test_run_weekly.py; PR #7 | 주간 실행기, 미연결 단계는 _stub/ 처리. Q-17(자동 실행 방식) 결정 남음 |
| D-04 | D 임소현 | 11/6 | 대기 | | |
| D-05 | D 임소현 | 11/16 | 대기 | | |
| D-06 | D 임소현 | 11/20 | 대기 | | |
| D-07 | D 임소현 | 11/27 | 대기 | | |
| D-08 | D 임소현 | 12/4 | 대기 | | |
| D-09 | D 임소현 | 12/7 | 대기 | | |
| D-10 | D 임소현 | 12/11 | 대기 | | |
| D-11 | D 임소현 | 12/11 | 대기 | | |
| D-12 | D 임소현 | 12/14~18 | 대기 | | |
