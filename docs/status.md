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
| PM-10 | PM 장지원 | 11/16 | 진행 중 | src/events/; [PR #34](https://github.com/Capstone-last-dance/AutoAdvisor/pull/34)·[PR #51](https://github.com/Capstone-last-dance/AutoAdvisor/pull/51) | 일간 경보 입력과 재생 규칙(DEC-32·33) 구현. #51에 신호일 생성 시각 제한(C 리뷰), `AlertsNotStarted` 경로 제거(D 리뷰), 늦은 경보의 생성 뒤 첫 체결일 청산(DEC-32 정정)을 반영했고 C 재검토가 필요하다. 유형별 10건 수작업 점검, 백테스트 청산 분포, T1·T3·T6 원문 판정은 남음 |
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
| A-09 | A 김윤하 | 11/30 | 완료 | src/models/ranker.py(train_ranker); tests/models/test_ranker.py | `train_ranker(h, exclude_features)`는 h=1·5·20, exclude_features 옵션 모두 이미 공개돼 있었음. 실데이터(C 공식 라벨·실제 종료일 purge, 폴드 2016)로 셋 다 오류 없이 실행됨을 확인(검증 rank IC: h=1 0.0545 / h=5 0.0517 / h=20 0.0814; exclude_features로 mom_12_1 제외 시 0.0814→0.0768로 하락해 옵션이 실제 반영됨을 확인). 합성 데이터 기반 회귀 테스트 4개 추가(파라미터화 h=1·5·20 + exclude_features), `pytest tests` 744 passed |
| A-10 | A 김윤하 | 12/7 | 대기 | | |
| A-11 | A 김윤하 | 기한 없음 | 대기 | | |
| B-01 | B 조윤상 | 11/1 | 완료 | docs/design_B.md; src/portfolio/; [PR #4](https://github.com/Capstone-last-dance/AutoAdvisor/pull/4) | M4·M8 설계와 PPO 골격. 합성 입력으로 학습 루프 테스트 통과, A 리뷰 후 병합 |
| B-02 | B 조윤상 | 11/8 | 완료 | src/portfolio/risk_layer.py; src/portfolio/vol_target_report.py; reports/20261103_B_vol_target/; [PR #11](https://github.com/Capstone-last-dance/AutoAdvisor/pull/11) | vol_target_annual = 0.157263 계산·부록 1 기입, 2011~2015 데이터만 사용 테스트 통과. compute_regime_mc 공개, DEC-19 반영 |
| B-03 | B 조윤상 | 11/13 | 완료 | src/portfolio/buffer_rule.py; tests/portfolio/test_buffer_rule.py; [PR #13](https://github.com/Capstone-last-dance/AutoAdvisor/pull/13) | 단계 1~5 단위 테스트 통과. PR #8 병합 뒤 main에서 C의 M5 위 합성 시장 실행 테스트 통과(건너뛰지 않음). A 승인. 실제 신호(A-04) 실행은 B-05에서 확인 |
| B-04 | B 조윤상 | 11/16 | 완료 | src/portfolio/risk_layer.py; src/portfolio/market_state_report.py; tests/portfolio/test_risk_layer.py; tests/portfolio/test_market_state_report.py; reports/20261104_B_m8_states/; [PR #13](https://github.com/Capstone-last-dance/AutoAdvisor/pull/13) | 불변식·경계 사례 테스트 통과, 2011~2025년 E_t·regime_mc 그림(prereg.yaml 초안 기준). A 승인. 11/9 공식 사전등록에서 쓰는 키가 바뀌면 새 run_id로 다시 만듦 |
| B-05 | B 조윤상 | 11/16 | 완료 | src/portfolio/engine_adapter.py; src/portfolio/env_check.py; src/portfolio/env_check_report.py; tests/portfolio/test_engine_adapter.py; tests/portfolio/test_env_check.py; tests/portfolio/test_env_check_report.py; reports/20261110_B_env_check/; docs/design_B.md 5.1절; [PR #21](https://github.com/Capstone-last-dance/AutoAdvisor/pull/21); [PR #36](https://github.com/Capstone-last-dance/AutoAdvisor/pull/36) | 9.3절 표 → C 엔진 어댑터(DEC-24~28 반영), A 승인(PR #21). A 베이스라인 신호(PR #35)로 완료 기준 (1) 781개 신호일 끝까지 실행, (2) 버퍼 규칙과 체결·비용·NAV 차이 0 통과 (reports/20261110_B_env_check/, 입력 파일 SHA-256 기록). 리포트 코드·결과 A 승인 (PR #36 재리뷰 11/10, 병합 c014969) |
| B-06 | B 조윤상 | 11/23 | 완료 | src/portfolio/rl_fold_report.py; src/portfolio/params.py; tests/portfolio/test_rl_fold_report.py; reports/20261110_B_rl_y2016_v2/; docs/design_B.md 3.4절; [PR #47](https://github.com/Capstone-last-dance/AutoAdvisor/pull/47) | 폴드 하나의 λ 탐색·PPO 학습 실행과 R-10 기록 구현(A make_folds 폴드, h = 20 purge, portfolio.rl.env 8개 검사). 폴드 2016 베이스라인 신호 리허설에서 6.5절 λ 선택 절차가 끝까지 돌았다(λ = 0, 시드 0 본 학습 300,000스텝, 약 14분). 체크포인트는 팀 드라이브, metrics.json에 지문 (PM 11/10). 다시 실행해도 가중치 지문이 같음. A 승인 (PR #47, 11/11, 병합 8828d6e) |
| B-07 | B 조윤상 | 11/30 | 진행 중 | src/portfolio/rl_fold_report.py; src/portfolio/rl_wf_report.py; tests/portfolio/test_rl_wf_report.py; docs/design_B.md 3.5절 | 폴드 학습에 A ML 신호(rlsig)·C 실제 라벨 종료일·M9 백테스트 경보 입력, 시험 연도 운용(positions/wf 시드별, ANOVA 2 입력) 구현. A·C 인계본이 B 폴드와 맞음. C 엔진 수정(PR #52) 병합, PM M9 백테스트 경보(`20261112_PM_m9_b07_v1`) 받음(11/12). M9를 연결한 실데이터 스모크(스텝 축소)로 10폴드 → 시험 운용 끝까지 실행 확인. D #50 병합으로 경보 파일 읽힘(11/14). A 뉴스 반영판 rlsig(`20261113_A_rlsig`, 11/14 받음)가 B 폴드와 10폴드 모두 맞고 스모크 통과 → 폴드 학습 가능. 뉴스 반영판 시험 신호(`20261113_A_wfnews`)와 rlsig meta.json도 11/14에 받음. B 시험 신호일과 맞고, 폴드별 하이퍼파라미터가 rlsig·wfnews 10폴드 모두 같음. 판정 입력이 이 판으로 모두 갖춰짐. 남은 것은 판정 실행(10폴드 학습 → 시험 운용). M9 없는 실행은 리허설로만 |
| B-08 | B 조윤상 | 12/7 | 대기 | | |
| B-09 | B 조윤상 | 12/7 | 대기 | | |
| B-10 | B 조윤상 | 12/14 | 대기 | | |
| C-01 | C 손석범 | 11/1 | 완료 | [PR #3](https://github.com/Capstone-last-dance/AutoAdvisor/pull/3); [설계](design_C.md) | 4.6·6.6절 라벨·평가·체결 설계 작성 및 병합. 후속 PM 결정은 PR #8에 반영 |
| C-02 | C 손석범 | 11/6 | 완료 | [병합 PR #8](https://github.com/Capstone-last-dance/AutoAdvisor/pull/8), [병합 PR #26](https://github.com/Capstone-last-dance/AutoAdvisor/pull/26) | A 11/8 데이터에서 라벨·종료일 각 978,983행을 생성해 스키마·키 대응·2025년 말 결과 컷오프를 검증함. C 공식 인계본을 받은 A가 자체 재현본과 전 행·전 열 값 일치를 확인함([이슈 #5](https://github.com/Capstone-last-dance/AutoAdvisor/issues/5)). 재사용 종목코드의 구회사 폐지 처리도 병합됨. |
| C-03 | C 손석범 | 11/6 | 진행 중 | [병합 PR #8](https://github.com/Capstone-last-dance/AutoAdvisor/pull/8), [B-05 병합 PR #21](https://github.com/Capstone-last-dance/AutoAdvisor/pull/21) | 공통 체결·비용·NAV 엔진과 평가지표 구현 및 B 최종 리뷰 완료. B-05 연결 검증에서 같은 비중의 체결·비용·NAV 일치 확인. 실제 신호 기반 전체 평가가 남음 |
| C-04 | C 손석범 | 11/16 | 진행 중 | [병합 PR #22](https://github.com/Capstone-last-dance/AutoAdvisor/pull/22), [병합 PR #23](https://github.com/Capstone-last-dance/AutoAdvisor/pull/23) | G0·G2 개인 행동 프록시, forward 실제 보유 재생·사후 성과 함수 구현·병합. 11/20 이후 실제 positions와 평가 검증이 남음 |
| C-05 | C 손석범 | 11/23 | 진행 중 | [A 판정용 재실행](https://github.com/Capstone-last-dance/AutoAdvisor/issues/5), [A PR #44](https://github.com/Capstone-last-dance/AutoAdvisor/pull/44) | C 공식 라벨과 PM `ev_*` 피처를 반영한 A의 10폴드 재실행(2016~2025년, 520 신호일, 228,422행)을 대조함. G1 ROC-AUC 0.6916 대 베이스라인 0.6975로 차이 −0.0058(기준 ≥ 0.03), G3 rank IC 0.0645 대 최선 베이스라인 0.0734로 차이 −0.0089(기준 ≥ 0.01): 둘 다 상대성과 조건 미달. 뉴스 피처 반영 재실행과 C-05 나머지 평가가 남음. |
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
| D-04 | D 임소현 | 11/6 | 완료 | src/pipeline/forward_test.py; forward_log.csv; PR #19, #28; 태그 ft-20261106 | 11/6 신호를 11/8 22:27 KST에 기록(마감 11/9 09:00, DEC-04 변경). 11/13 기록 예정 |
| D-05 | D 임소현 | 11/16 | 진행 중 | app/; src/pipeline/make_dashboard_sample.py; tests/app/test_dashboard.py | 샘플 데이터로 6.11절 완료 기준 (1)~(4) 확인. 진입 후보의 M9 쿨다운은 최근 20거래일 경보로 임시 처리. Q-16 결정: Streamlit Community Cloud, 팀원 이메일로 접속 제한. tickers 공개 범위 확인 전까지 종목코드만 표시 |
| D-06 | D 임소현 | 11/20 | 진행 중 | src/pipeline/run_daily_alerts.py; src/common/io.py; [PR #50](https://github.com/Capstone-last-dance/AutoAdvisor/pull/50) | 일간 경보 실행기: 달력일 d로 M9 `daily_alerts(d)`를 불러 저장, 0건은 0행 파일, 미수집·실패는 파일을 쓰지 않음. alerts `generated_at` 열, 임시 파일 교체 저장. M9 PR #51 병합 뒤 실제 공시로 실행 확인 남음 |
| D-07 | D 임소현 | 11/27 | 대기 | | |
| D-08 | D 임소현 | 12/4 | 대기 | | |
| D-09 | D 임소현 | 12/7 | 대기 | | |
| D-10 | D 임소현 | 12/11 | 대기 | | |
| D-11 | D 임소현 | 12/11 | 대기 | | |
| D-12 | D 임소현 | 12/14~18 | 대기 | | |
