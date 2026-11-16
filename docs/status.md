# 작업 진행 현황

마지막 갱신: 2026-11-16

아래 표는 전체 명세의 작업 ID 단위 상태이며 세부 단계 완료와 구분한다. 개인 작업 기록·실행 로그·수집 원본은 Git에서 제외하고, 이 문서에는 팀 진행 현황과 남은 과제를 기록한다.

| 작업 ID | 담당 | 기한 | 상태 | 근거 | 메모 |
|---|---|---|---|---|---|
| PM-01 | PM 장지원 | 10/28 | 완료 | docs/status.md; docs/decisions.md; docs/deviation_log.md; docs/preregistration.md; [PR #9](https://github.com/Capstone-last-dance/AutoAdvisor/pull/9)·[#10](https://github.com/Capstone-last-dance/AutoAdvisor/pull/10)·[#15](https://github.com/Capstone-last-dance/AutoAdvisor/pull/15) | 공용 기록 4개 작성, status.md에 부록 2의 모든 작업 ID가 들어 있다(완료 기준). 결정·변경 기록 갱신은 PM-16에서 이어간다 |
| PM-02 | PM 장지원 | 10/30 | 완료 | README.md; .env.example; requirements-lock.txt; [PR #2](https://github.com/Capstone-last-dance/AutoAdvisor/pull/2) | 저장소에 키 값이 없고 `.env.example` 변수명을 확정했다. D-01에서 README대로 설치·재현을 확인했다(완료 기준). 뉴스 원천이 빅카인즈로 바뀌어(DEC-06) 네이버 인증은 필수가 아니다. 키 교체는 R-16 절차를 따른다 |
| PM-03 | PM 장지원 | 10/30 | 진행 중 | docs/프로젝트_구조_및_역할분담_v0.3.md 부록 3 Q-06·Q-08 | Q-08 문헌·DART 누락 반례·KRX 공식 CSV 2개·KIND 과거/상폐 표시 회사/해제 원문 3개 검증. 표본·실행 기록은 별도 보관. A 공동 확인·전체 기간 상태 데이터 검증 남음 |
| PM-04 | PM 장지원 | 11/2 | 완료 | src/data/disclosures.py; [공시 계약](disclosures_contract.md); [PR #2](https://github.com/Capstone-last-dance/AutoAdvisor/pull/2) (11/4 병합 `2bdc09e`) | OpenDART 목록 2010-01-01~2026-11-08 860,976건 보관, `report_nm_norm` 정규화(6.10절). 원천에 접수 시각이 없어 `rcept_dt`는 결측이고 `available_at` 규약(DEC-09)으로 대신한다. C 승인 후 병합. 회사↔종목 연결 근거(DEC-09 (2) 재검토)는 결정 대기로 남긴다 |
| PM-05 | PM 장지원 | 11/6 | 진행 중 | src/data/news.py; src/data/collect_news.py; tests/unit/test_news.py; `data/interim/news_bigkinds_excel/`(로컬, 이용 조건상 비공개) | 빅카인즈 일별 트렌드 1,262개 키워드(11/5), 기사 엑셀 3,329,059행 수집·대조(11/7), (기사, 종목) 표 3,331,437행(트렌드 대조 99.89%, 11/8), 제목 감성 점수(11/9). 이름 검토 250종목 보완 수집(11/13). 11/16 #56 리뷰로 상호 변경 처리의 누수를 확인해 옛 이름 보완 수집(159종목)을 진행한다. 10/31 이후 기사는 아직 수집하지 않아 forward 신호일은 결측 |
| PM-06 | PM 장지원 | 11/8 | 완료 | [PR #16](https://github.com/Capstone-last-dance/AutoAdvisor/pull/16) (11/5 C 승인 후 병합 `bd46cff`) | 2015~2025년 공시 제목 588,868건 표본으로 T1~T6·`ev_*` 키워드 확정안과 키워드별 매칭·오매칭 예시를 작성했다(완료 기준). 회사 답변 원문·거래일 검증은 PM-10 유형별 점검에서 이어간다 |
| PM-07 | PM 장지원 | 11/9 | 완료 | configs/prereg.yaml; docs/preregistration.md; [PR #29](https://github.com/Capstone-last-dance/AutoAdvisor/pull/29) (11/9 18:02 병합 `89074af`); [PR #39](https://github.com/Capstone-last-dance/AutoAdvisor/pull/39) | `FILL_` 0개, 커밋 해시와 시각을 preregistration.md에 기록했다(완료 기준). 이후 변경은 deviation_log.md에 적는다 |
| PM-08 | PM 장지원 | 11/16 | 완료 | src/features/event_features.py; tests/features/test_event_features.py; reports/20261109_PM08_ev/; [PR #37](https://github.com/Capstone-last-dance/AutoAdvisor/pull/37) (11/9 병합 `9a416f9`) | `ev_*` 6개. 결측률 0, 실데이터 무작위 신호일 20개 누수 테스트 20/20, 826개 신호일 348,678행이 U_t와 일치(6.2절 완료 기준 (1)~(3)). C 승인. 11/10 A에 인계 |
| PM-09 | PM 장지원 | 11/16 | 진행 중 | src/features/news_features.py; tests/features/test_news_features.py; [PR #56](https://github.com/Capstone-last-dance/AutoAdvisor/pull/56) (11/16 병합 `79ff49f`); reports/20261116_PM09_rename_pit/ | 코드는 C 승인 후 병합(테스트 24개). 실데이터 v6(`pm_features_20261113e`, 11/13 A 인계)는 C 확인 요청으로 신호일 뒤 상호 변경이 과거 결측 표시를 정하는 누수를 확인했다(16,044행, 4.6%, 174종목, [#56 답글](https://github.com/Capstone-last-dance/AutoAdvisor/pull/56#issuecomment-6907615460)). 옛 이름 기사 보완 수집(159종목, 엑셀 17행) → v7 생성, 신호일별 `covered_from` 재구성 누수 테스트, A 재인계와 deviation log 기록이 남았다 |
| PM-10 | PM 장지원 | 11/16 | 진행 중 | src/events/; reports/20261109_PM_m9_v1/; [PR #34](https://github.com/Capstone-last-dance/AutoAdvisor/pull/34) (11/9)·[PR #51](https://github.com/Capstone-last-dance/AutoAdvisor/pull/51) (11/12) | T2~T5 제목 판정·종목 연결·쿨다운·청산 일정과 일간 경보 `daily_alerts`·`generated_at` 재생 규칙(DEC-32·33) 병합. 2015~2025 경보 2,047건(11/9 v1, 완료 기준 (1)). 백테스트 경보 `20261112_PM_m9_b07_v1`을 B-07에 전달. 유형별 무작위 10건 수작업 점검(완료 기준 (2)), T1·T3·T6 원문 판정이 남았다 |
| PM-11 | PM 장지원 | 12/4 | 진행 중 | src/rag/; tests/rag/test_evidence.py; [PR #54](https://github.com/Capstone-last-dance/AutoAdvisor/pull/54) (11/16 병합 `a1f85fd`) | 11/20 forward용 공시 템플릿 근거 요약(시각 컷오프, 경보는 `generated_at` < 기록 마감, 고지 문구) C 승인 후 병합. LLM 요약(Qwen3-1.7B 시험)은 12/4까지 |
| PM-12 | PM 장지원 | 12/7 | 대기 | | |
| PM-13 | PM 장지원 | 12/11 | 대기 | | |
| PM-14 | PM 장지원 | 12/11 | 대기 | | |
| PM-15 | PM 장지원 | 12/14 | 대기 | | |
| PM-16 | PM 장지원 | 매주 | 진행 중 | docs/decisions.md; docs/deviation_log.md | 주간 회의 진행, decisions·deviation 기록 갱신 중 (DEC-34까지) |
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
| B-07 | B 조윤상 | 11/30 | 진행 중 | src/portfolio/rl_fold_report.py; src/portfolio/rl_wf_report.py; src/portfolio/buffer_wf_report.py; tests/portfolio/test_rl_wf_report.py; tests/portfolio/test_buffer_wf_report.py; reports/20261114_B_rlwf_y{2016..2025}/; reports/20261114_B_rl_wf/; positions/wf_20261114_B_rl_wf_s{0..4}.parquet; docs/design_B.md 3.5절; [PR #58](https://github.com/Capstone-last-dance/AutoAdvisor/pull/58) | 폴드 학습에 A ML 신호(rlsig)·C 실제 라벨 종료일·M9 백테스트 경보 입력, 시험 연도 운용(positions/wf 시드별, ANOVA 2 입력) 구현. A·C 인계본이 B 폴드와 맞음. C 엔진 수정(PR #52) 병합, PM M9 백테스트 경보(`20261112_PM_m9_b07_v1`) 받음(11/12). M9를 연결한 실데이터 스모크(스텝 축소)로 10폴드 → 시험 운용 끝까지 실행 확인. D #50 병합으로 경보 파일 읽힘(11/14). A 뉴스 반영판 rlsig(`20261113_A_rlsig`, 11/14 받음)가 B 폴드와 10폴드 모두 맞고 스모크 통과 → 폴드 학습 가능. 뉴스 반영판 시험 신호(`20261113_A_wfnews`)와 rlsig meta.json도 11/14에 받음. B 시험 신호일과 맞고, 폴드별 하이퍼파라미터가 rlsig·wfnews 10폴드 모두 같음. 판정 입력이 이 판으로 모두 갖춰짐. 판정 실행 완료(11/14, 코드 커밋 099a511): 10폴드 × 5시드 학습과 시험 운용의 완료 기준 6개 모두 통과(`b07_complete = True`), 체크포인트 50개 지문 일치. B 확인 IR 시드 평균 −1.013(현금 비중 평균 55%와 비용이 주된 원인, design_B 3.5절. 설계 변경 없음). A·PM 승인 (PR #58, 11/14). 남은 것: 6.5절 완료 기준의 버퍼 규칙 비교 표(G5). B가 버퍼 규칙 + M8 + M9 기준선 positions(G7 미적용·G4 진단 포함)를 만들고 G5 판정표를 C의 M5 함수로 계산(C 동의, 11/16). C 사전 점검: RL positions G0 위반 0건, fold_seed_ir ANOVA 통과. 11/18 기준선 인계 때 run_id·M8 모드·M9 경보 run_id·원본 파일 지문을 같이 남김. G5 비교 표 코드(`buffer_wf_report.py`) 작성. 정식 실행 `20261117_B_buffer_wf`(커밋 7b42490) 완료 기준 통과, G5 미달(IR 차 −0.087, 회전율 비 0.74). DEC-35(11/16)로 A 신호 v7 재인계 → B-07 폴드 학습·시험 운용·기준선 재실행 필요(v6 결과는 원래 설정 결과로 함께 보고) |
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
| D-03 | D 임소현 | 11/6 | 완료 | src/pipeline/run_weekly.py; tests/pipeline/test_run_weekly.py; PR #7; [PR #38](https://github.com/Capstone-last-dance/AutoAdvisor/pull/38) | 주간 실행기, 미연결 단계는 _stub/ 처리. #38로 3~5단계(M3a·M3b·M4·M8), A PR #55로 1·2단계(M1·M2) 연결. 6단계(M6 `m6_step`)는 PR #54 리뷰 뒤 연결. Q-17(자동 실행 방식) 결정 남음 |
| D-04 | D 임소현 | 11/6 | 완료 | src/pipeline/forward_test.py; forward_log.csv; PR #19, #28; 태그 ft-20261106 | 11/6 신호를 11/8 22:27 KST에 기록(마감 11/9 09:00, DEC-04 변경). 11/13 기록 예정 |
| D-05 | D 임소현 | 11/16 | 진행 중 | app/; src/pipeline/make_dashboard_sample.py; tests/app/test_dashboard.py | 샘플 데이터로 6.11절 완료 기준 (1)~(4) 확인. 진입 후보는 M9 `cooldown_at`으로 쿨다운 종목을 뺀다 (경보 미시작·미수집이면 계산 보류). Q-16 결정: Streamlit Community Cloud, 팀원 이메일로 접속 제한. tickers 공개 범위 확인 전까지 종목코드만 표시 |
| D-06 | D 임소현 | 11/20 | 진행 중 | src/pipeline/run_daily_alerts.py; src/common/io.py; [PR #50](https://github.com/Capstone-last-dance/AutoAdvisor/pull/50) | 일간 경보 실행기: 달력일 d로 M9 `daily_alerts(d)`를 불러 저장, 0건은 0행 파일, 미수집·실패는 파일을 쓰지 않음. alerts `generated_at` 열, 임시 파일 교체 저장. M9 PR #51과 함께 #50 병합(11/16). 실제 공시로 첫 실행 확인 남음 |
| D-07 | D 임소현 | 11/27 | 대기 | | |
| D-08 | D 임소현 | 12/4 | 대기 | | |
| D-09 | D 임소현 | 12/7 | 대기 | | |
| D-10 | D 임소현 | 12/11 | 대기 | | |
| D-11 | D 임소현 | 12/11 | 대기 | | |
| D-12 | D 임소현 | 12/14~18 | 대기 | | |
