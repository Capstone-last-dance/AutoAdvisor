# 사전등록 기록

명세서 8장. PM이 부록 1을 `configs/prereg.yaml`로 저장하고 `FILL_` 값을 모두 채운 뒤 2026-11-09에 커밋한다 (작업 PM-07). 커밋한 뒤에는 값을 바꾸지 않는다 (R-02). 바꿔야 하면 `docs/deviation_log.md` 절차를 따른다.

2026-11-03부터 main의 `configs/prereg.yaml`에 초안이 있다. 내용은 명세서 부록 1과 같고, `vol_target_annual`을 채워 `FILL_`은 1개(`news_sentiment_model`)가 남았다. 11/9 공식 커밋 전까지는 초안이므로 아래 표의 값과 결정을 반영해 고칠 수 있다. 커밋 해시·시각은 11/9 공식 확정 커밋에 대해서만 기입한다.

## 커밋 정보

| 항목 | 값 |
|---|---|
| 커밋 예정일 | 2026-11-09 |
| 대상 파일 | `configs/prereg.yaml` |
| 커밋 해시 | (커밋 후 기입) |
| 커밋 시각 (KST) | (커밋 후 기입) |

## 전원 확인

| 역할 | 담당자 | 확인일 |
|---|---|---|
| PM | 장지원 | |
| A | 김윤하 | |
| B | 조윤상 | |
| C | 손석범 | |
| D | 임소현 | |

## 커밋 전에 채울 값

| 키 | 담당 | 확인 항목 | 기한 | 상태 |
|---|---|---|---|---|
| `features.news_sentiment_model` | PM | Q-09 | 11/8 | 대기. 뉴스 피처를 빼면(사전 허용 변경 3번) 해당 없음으로 바꾼다 |
| `risk_layer.vol_target_annual` | B | Q-12 | 11/8 | 기입: 0.157263 (`reports/20261103_B_vol_target/`). B-02 [PR #11](https://github.com/Capstone-last-dance/AutoAdvisor/pull/11) 리뷰 중이며, 값이 바뀌면 갱신 |
| `features.event_keywords`, `events.triggers` 확정안 | PM | Q-13 | 11/8 | 대기 |

## 커밋 전 선행 작업 (PM-07)

| 작업 ID | 담당 | 기한 | 내용 | 상태 (11/3) |
|---|---|---|---|---|
| PM-03 | PM | 10/30 | Q-06, Q-08 조사 | 진행 중 |
| A-01 | A | 10/30 | Q-01~Q-05, Q-14, Q-15 조사 | 대기. Q-01·Q-04는 결과가 DEC-22·23에 반영됐고 부록 3 칸 기입이 남음 |
| D-01 | D | 10/30 | 저장소 골격, 라이브러리 버전(Q-18) | 완료 |
| PM-06 | PM | 11/8 | 키워드 검증 (Q-13) | 대기 |
| PM-17 | PM | 11/8 | Q-07, Q-09, Q-10, Q-11 조사 | 진행 중 |
| B-02 | B | 11/8 | `vol_target_annual` 계산 (Q-12) | PR #11 리뷰 중 |

## 커밋 전에 닫을 결정

`docs/decisions.md`의 결정 대기 안건 중 기한이 11/9 전인 것은 커밋 전에 결정해 반영한다.

| ID | 안건 | 기한 | 사전등록에 미치는 영향 |
|---|---|---|---|
| DEC-04 | forward test 신호 전용 커밋 모드 | 11/6 전 | 6.8·6.12절 운영 규칙 |
| DEC-05 | 과제 원안 필수 항목 대응 (교수 의견 반영) | 11/9 전 | 비교 기준·보상·지표가 바뀌면 부록 1 수정 |
| DEC-06 | 범위 예산 (뉴스 피처 유지 여부, RL 폴드·시드·스텝) | 11/9 전 | `features`, `portfolio.rl` |
| DEC-07 | 판정 규칙 (G2 지표, 성공 조건, G6 누수 규칙) | 11/9 전 | `kpi` |
| DEC-08 | RL 학습 구간에 공급할 신호 규칙 | 11/9 전 | `portfolio.rl`, 4.7절 |
| DEC-10 | T6 발행금액 의미와 정정 시점 | 11/8 | `events.triggers.T6` |
| DEC-20 | RL 보상의 DD 측정 기준 | 11/9 전 | 6.5절 보상 |
| DEC-21 | RL 환경 세부값 8개 | 11/9 전 | `portfolio.rl`에 추가 |

이미 반영한 결정: DEC-19(`regime_mc` 이력 부족 처리, 식과 값 변경 없음), DEC-22(수급 피처 제외와 개인 행동 프록시 대체 정의, [deviation log](deviation_log.md)). DEC-22는 부록 1 값을 바꾸지 않는다. 대체 정의는 `retail_proxy.fallback_if_flows_unavailable`에 이미 있다.

11/4 PM 결정은 [DEC-25~29](decisions.md)에 기록했다. 정지 중 평가·재개일 처리, 거래량 0 대체, 공용 지수 분리, 재사용 두 건의 보완 데이터 기준을 v1 설계에 반영한다. `features.model_input_exclusions`에 `ss_allowed`를 넣고 `ss_regime`은 모델 입력으로 유지한다. 확인 불가한 `ss_allowed`는 피처 파일에 결측으로 저장한다. 담당 파트의 코드·데이터 반영과 검증 결과는 별도로 확인한다.
