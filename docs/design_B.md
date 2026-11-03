# B 파트 설계: M4 포트폴리오 구성기와 M8 리스크 레이어

> 작업 B-01 · 기준 명세서 v0.3 · 작성 2026-10-28 · 담당 B 조윤상 · 리뷰어 A 김윤하 (9.2절)
>
> 6.5절(M4)과 6.9절(M8)을 코드 구조로 옮긴 설계다. 이 문서와 명세서가 다르면 명세서를 따른다 (0.1절).
> 8절의 [결정 필요] 다섯 건은 사전등록(2026-11-09) 전에 정해야 한다.

---

## 1. 구현 상태

| 파일 | 내용 | 상태 |
|---|---|---|
| `src/portfolio/params.py` | 부록 1 `portfolio`, `risk_layer` 읽기. `FILL_` 값은 None으로 읽고, 실제로 쓰는 순간 `PreregValueMissing` (R-08) | 구현 |
| `src/portfolio/risk_layer.py` | M8 Step 1~6, 불변식, positions 행 생성, t일 유효 업종표 | 구현. 경계 사례 테스트와 2011~2025년 그림은 B-04 |
| 〃 `compute_regime_mc(date)`, `compute_vol_target_annual` | 공개 함수, Q-12 값 계산 | 구현 (B-02). 부록 1에 넣을 값은 실제 코스피 200 데이터(A-02)로 계산한다 |
| `src/portfolio/vol_target_report.py` | B-02 결과 리포트: `python -m src.portfolio.vol_target_report --run-id {YYYYMMDD}_B_{설명}` → `reports/{run_id}/`에 config.yaml 사본, git_commit.txt, metrics.json (R-10) | 구현 (B-02) |
| `src/portfolio/buffer_rule.py` | 버퍼 규칙 단계 1~4 | 골격 (B-03) |
| `src/portfolio/interfaces.py` | B가 C(M5)와 PM(M9)에 기대하는 호출 형태 | 초안. C·PM 확인 필요 (5절) |
| `src/portfolio/rl_data.py` | 9.3절 표 → 신호일별 스냅샷 | 구현 |
| `src/portfolio/rl_env.py` | gymnasium 환경: 관측, 행동 해석, M8 호출, 보상 | 구현. 실제 M5 연결은 B-05 |
| `src/portfolio/rl_train.py` | λ 선택, 본 학습, 검증 IR 체크포인트, 결정적 운용 | 구현. 폴드 데이터 구성과 R-10 기록은 B-06 |
| `tests/portfolio/` | 단위 테스트와 테스트 전용 가짜 입력(`fakes.py`) | 구현 (7절) |

모듈 의존 방향은 `params` ← `risk_layer` ← `rl_data` ← `rl_env` ← `rl_train`이다. `risk_layer.py`는 numpy와 pandas만 쓴다. A가 피처 `regime_mc`를 만들 때 import해도 torch나 gymnasium이 필요 없다.

---

## 2. 6.5절 대응: 버퍼 규칙

| 6.5절 | 코드 | 상태 |
|---|---|---|
| 단계 1 보유 집합 H | `buffer_rule_weights(holdings, ...)`. H는 M5 상태(직전 체결 후 실제 보유)에서 읽는다 | B-03 |
| 단계 2 유지 조건 | 〃 rank ≤ `exit_rank_greater_than`, crash_flag = 0, U_t 소속, 쿨다운 아님 | B-03 |
| 단계 3 추가 | 〃 rank ≤ `enter_rank_max`, rank 오름차순, `max_holdings`까지 | B-03 |
| 단계 4 동일가중 | 〃 1 ÷ 종목 수. 종목이 없으면 빈 Series (현금 100%) | B-03 |
| 단계 5 M8 | `apply_risk_layer(..., constructor="buffer")` | 구현 |

---

## 3. 6.5절 대응: RL 정책

### 3.1 항목별 대응

| 6.5절 | 코드 | 비고 |
|---|---|---|
| 프레임워크 | `PortfolioEnv(gymnasium.Env)`, `PPO("MlpPolicy", ...)` (`rl_train.train_one`) | 하이퍼파라미터는 SB3 기본값. 계산 장치만 `device="cpu"`로 고정한다 (재현성, R-10) |
| 스텝 | `PortfolioEnv.step`: 신호일 1개 | |
| 에피소드 | `reset()`: `episode="random"`이면 `np_random.integers(0, n − 52)`로 시작 신호일을 뽑아 52스텝. 초기 상태는 현금 100% | 마지막 스텝의 보상 구간(다음 신호일)까지 구간 안에 들어가는 시작점만 뽑는다 |
| 후보 슬롯 | `_observe()`: (rank ≤ 40) ∪ (보유)를 rank 오름차순으로, U_t 밖 보유는 맨 뒤, 남는 슬롯은 빈 슬롯 | 60개를 넘을 때는 D3(c) |
| 관측 | 3.2절 | 366차원 float32 |
| 행동 | `action_space = Box(−10, 10, (61,))`, 로짓 = 행동 | 범위는 D3(a) |
| 행동 해석 (1) 마스킹 | `_interpret()`: 빈 슬롯, crash_flag = 1, 쿨다운, U_t 밖 → −∞ | 현금 로짓은 마스킹하지 않는다 |
| (2) softmax | 〃 | |
| (3) 상위 20개, 0.005 미만 제거 | 〃 `np.lexsort`. 잘린 비중은 현금 | 동률은 D3(d) |
| (4) M8 | `risk_layer_weights`, `check_invariants` | 매 스텝 G0 불변식을 검사한다 |
| 로그확률 | SB3 DiagGaussian 기본 동작. 마스킹, softmax, 절단, M8은 환경 안의 결정적 변환 | 별도 정책 클래스를 만들지 않는다 |
| 보상 | `step()`: r = R_net − R_BM − λ × max(0, DD(t′) − DD(t)) | R_net, R_BM은 M5가 계산. DD 기준은 D2 |
| λ 선택 | `select_lambda`: 격자 {0, 0.5, 1, 2}, 시드 0, 100,000스텝 | λ의 점수는 D3(f) |
| 본 학습 | `train_fold` → `train_one`: 시드 0~4, 시드마다 300,000스텝 | |
| 체크포인트 | `ValidationIRCallback`: 10,000스텝마다 검증 구간을 결정적 행동으로 운용해 IR(M5)을 계산한다. 최고 IR만 `best.zip`으로 저장하고, 모든 평가는 `eval_log.csv`에 남긴다 | 동률이면 먼저 나온 체크포인트. 평가 기록은 지우지 않는다 (R-11) |
| 테스트 행동 | `rollout()`: `predict(deterministic=True)` (가우시안 평균) | `policy_for(신호일)`로 연도 경계에서 모델을 바꾸고 보유는 유지한다 (4.7절) |
| 환경 규칙 | 체결, 비용, 상·하한가, 거래정지, 이벤트 청산은 `ExecutionEngine.run_period`(M5), 한도와 노출 조절은 M8 함수 | 다시 구현하지 않는다 |

### 3.2 관측 벡터

`obs[6k + j]`는 슬롯 k(0~59)의 특성 j이고, `obs[360 + c]`는 공통 특성 c다.

| j | 슬롯 특성 | 출처 | U_t 밖 보유 슬롯 | 빈 슬롯 |
|---|---|---|---|---|
| 0 | rank_score 백분위 | `signals.rank_score`를 U_t 안에서 `rank(pct=True, method="average")` | 0 (D3(b)) | 0 |
| 1 | crash_prob | `signals.crash_prob` | 0 (D3(b)) | 0 |
| 2 | 현재 비중 | M5: 신호일 t 종가로 평가한 보유 비중 (D3(h)) | 실제 비중 | 0 |
| 3 | vol_20 순위 | `features.vol_20` (이미 순위로 바뀐 값) | 0 (D3(b)) | 0 |
| 4 | 보유 여부 | | 1 | 0 |
| 5 | 유효 슬롯 여부 | | 1 | 0 |

| c | 공통 특성 | 출처 |
|---|---|---|
| 0 | 현금 비중 | M5: 신호일 t 종가 기준 |
| 1 | 직전 스텝 회전율 | M5: 직전 체결의 회전율. 에피소드 첫 스텝은 0 |
| 2 | mkt_vol_20 | `features` |
| 3 | mkt_ret_20 | `features` |
| 4 | regime_mc | M8 Step 5. `features.regime_mc`가 있으면 같은 값인지 확인한다 |
| 5 | E_t | M8 Step 4~5 |

### 3.3 시간축

```text
신호일 t 15:30 이후     t+1일 종가                        t′(다음 신호일)          t′+1일 종가
관측 s_t, 행동 a_t  →  M8 비중 체결(비용) ── 보유 ──→  관측 s_t′, 행동 a_t′  →  체결 …
                       NAV_pre(t+1)                                            NAV_pre(t′+1)

R_net(t→t′) = NAV_pre(t′+1) / NAV_pre(t+1) − 1        NAV_pre: 체결일 종가, 체결 직전
R_BM(t→t′)  = BM(t′+1) / BM(t+1) − 1
r_t         = R_net − R_BM − λ × max(0, DD(t′) − DD(t))
```

관측 s_t′는 t′일까지의 정보만 쓴다 (R-03). 보상 r_t는 t′+1일 종가를 쓰지만, 행동이 끝난 뒤 계산하는 값이라 관측에 들어가지 않는다.

R_net을 체결 직전 NAV로 재면, 행동 a_t가 일으킨 t+1일 체결 비용이 그 행동의 보상 r_t에 들어간다. 체결 직후 NAV로 재면 그 비용이 앞 스텝의 보상으로 넘어가서, 비용을 인식하는 정책(주장 3)을 학습하기 어렵다.

---

## 4. 6.9절 대응: M8

| 6.9절 | 코드 |
|---|---|
| 입력 | `apply_risk_layer(w_raw, signals, sectors, bm_close, cooldown, universe, date, params, constructor, mode)` |
| Step 1 필터 | `risk_layer_weights`: `blocked`(crash_flag = 1, 쿨다운, U_t 밖)의 비중을 0으로 |
| Step 2 종목 한도 | `np.minimum(w, stock_cap)` |
| Step 3 업종 한도 | 업종 합 > `sector_cap`이면 그 업종 종목에 `sector_cap ÷ 업종 합`을 곱한다. 업종 정보가 없는 종목은 `unknown_sector_label` 한 그룹 |
| Step 4 노출 상한 | `market_state_table`: sigma_t, E_t = min(1.00, max(0.30, vol_target_annual ÷ sigma_t)) |
| Step 5 국면 대응 | 〃 bear_t, rebound_t, regime_mc_t. regime_mc_t = 1이면 E_t = min(E_t, 0.70) |
| Step 6 적용 | `exposure × w`. 현금 = 1 − Σw |
| 재배분 금지 | 어느 단계도 비중을 늘리지 않는다. 잘린 비중은 모두 현금이다 |
| 출력 | `positions_frame`: 9.3절 열(date, ticker, target_weight, constructor, exposure, regime_mc). 비중이 0보다 큰 종목과 "CASH" 행 |
| 불변식 | `check_invariants`. 어기면 `RiskLayerViolation` (G0) |
| 공개 함수 | `apply_risk_layer`, `compute_regime_mc(date)`. 후자는 `bm_close`, `params`를 넘기지 않으면 `io.read_table("index")`와 `configs/prereg.yaml`에서 읽는다. Step 5만 쓰므로 vol_target_annual이 없어도 된다 |
| vol_target_annual | `compute_vol_target_annual(bm_close, params, vol_target_years(cfg))`: Step 4와 같은 `sigma_series`의 2011~2015년 매 거래일 평균. 2015년보다 늦은 종가는 계산 전에 잘라낸다. 부록 1 값이 `FILL_`인 동안 Step 4는 `PreregValueMissing`으로 멈춘다 (R-08) |
| G7 'M8·M9 미적용' | `mode="off"`: Step 2~6과 쿨다운을 끄고 crash_flag·U_t 필터만 남긴다. 불변식은 비음수, 현금 ≥ 0, 합계만 확인한다 |
| G4 '노출 조절 제외' 진단 | `mode="no_exposure"`: E_t = 1 |

구현 메모:

- 버퍼 규칙과 파이프라인은 pandas 경로(`apply_risk_layer`)를, RL 환경은 배열 경로(`risk_layer_weights`, `check_invariants`)를 쓴다. pandas 경로도 안에서 같은 배열 함수를 부른다.
- R-03: `market_state`는 t일보다 늦은 코스피 200 종가를 잘라낸 뒤 계산한다. `apply_risk_layer`는 signals에 t가 아닌 날짜가 있으면 예외를 낸다.
- 부동소수 오차로 Σw가 1을 1e−9 이내에서 넘으면 현금을 0으로 둔다. 그래서 '현금 ≥ 0'은 정확히, '합계 = 1 ± 1e−9'는 허용 오차 안에서 지킨다.
- 창을 채울 코스피 200 이력이 없으면 `InsufficientHistory`를 낸다 (D4).
- vol_target_annual의 기간: 시작 연도는 `data.first_train_signal_year`(2011), 끝 연도는 walk-forward 첫 테스트 연도의 전년(2015)으로 부록 1에서 만든다. 2011년 첫 거래일의 sigma_t도 6.9절 정의대로 직전 20개 수익률을 쓰므로 2010년 12월 종가 21개가 들어간다. 부록 2 B-02 완료 기준의 '2011~2015년 데이터만'은 6.9절 정의(우선순위가 높다, 0.1절)에 맞춰 '2016년 이후 데이터 미사용과 이 21개 밖의 2010년 데이터 미사용'으로 확인한다.

---

## 5. 다른 역할과의 인터페이스

B는 다른 역할의 파일과 코드를 고치지 않는다 (R-17). 아래는 B가 기대하는 형태이고, `src/portfolio/interfaces.py`에 같은 내용이 있다. 실제 함수가 나오면 B-05에서 어댑터로 연결한다.

C-01(`docs/design_C.md`, PR #3 병합)은 5.1절 형태를 `src.backtest.engine.ExecutionEngine`으로 그대로 받아들였다. 체결 직전 NAV 기준의 a_k, `nav_max`, M9 우선, positions에 없는 종목 = 0도 같다.

### 5.1 M5 체결 엔진 (C 손석범)

`ExecutionEngine.run_period(state, targets, signal_date, next_signal_date) -> PeriodResult`

| 필드 | 뜻 |
|---|---|
| `nav_start` | t+1일 종가, 체결 직전 NAV |
| `nav_end` | t′+1일 종가, 체결 직전 NAV |
| `nav_max` | t+1일 ~ t′+1일 일간 NAV의 최댓값 (D2) |
| `r_net` | nav_end ÷ nav_start − 1. t+1일 체결 비용을 포함한다 |
| `r_bm` | BM(t′+1) ÷ BM(t+1) − 1 |
| `turnover` | t+1일 체결의 회전율 (6.6절) |
| `weights_obs`, `cash_obs` | t′일 종가로 평가한 보유 비중과 현금 비중 |
| `state` | 다음 호출에 넘길 엔진 상태 |

요청 사항:

1. 4.3절 체결 규칙, 4.8절 비용, M9 이벤트 청산을 `run_period` 안에서 처리한다. 버퍼 규칙 백테스트와 같은 함수를 써야 B-05의 '같은 비중이면 같은 체결·비용 결과' 기준을 맞출 수 있다.
2. 호출 한 번에 1 ms 이하를 목표로 한다 (6절).
3. positions에 없는 종목의 목표 비중은 0으로 해석한다.
4. IR 함수는 주간 초과수익 배열을 받아 실수 하나를 돌려준다 (`IRFunction`).

### 5.2 M9 쿨다운 (PM 장지원)

`CooldownFunction(신호일 t) -> 종목코드 목록`: 체결일 t+1에 신규 진입이 금지된 종목이다. 접수 시각이 t일 15:30 이전인 공시만 쓴다 (R-03). 쿨다운 기간은 6.10절대로 접수일 다음 거래일부터 20거래일이다.

### 5.3 신호와 피처 (A 김윤하)

| 입력 | 9.3절 파일 | 비고 |
|---|---|---|
| rank, rank_score, crash_prob, crash_flag | `signals` | 학습·검증 구간 신호의 출처는 D1 |
| vol_20, mkt_vol_20, mkt_ret_20, regime_mc | `features` | regime_mc는 A가 `compute_regime_mc(date)`(B-02, 11/8)를 호출해 만든다 |
| U_t | `universe` (`in_universe` = True) | signals 종목과 같아야 한다 |
| 업종 | `sector` (`sectors_at`로 t일 유효 행) | |
| 코스피 200 종가 | `index` (`index_code` = KOSPI200) | D4 |

### 5.4 입출력과 실행 환경 (D 임소현)

- B는 positions를 `io.write_table("positions", df, date=t)` 또는 `run_id=...`로 쓴다 (R-18). 앞 단계 파일이 아직 없으면 `stub=True`로 읽는다. B가 만드는 `date` 열은 io.py(D-02) 규칙대로 시각 없는 datetime64[ns]다. io.py로 쓰고 읽어도 그대로인지는 `test_positions_round_trip_through_io`가 확인한다.
- 단위 테스트는 `tests/portfolio/`에 두었다 (`tests/__init__.py` 포함). README의 `tests/{영역}/` 규칙과 같다.
- 검증한 환경:
  - B-01: macOS, Python 3.13.7, numpy 2.5.4, pandas 3.1.0, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.14.1, PyYAML 6.0.3, pytest 9.2.0
  - B-02부터: D-01 고정 버전(`requirements-lock.txt`), Windows, Python 3.12.10, torch 2.15.0+cpu, pytest 9.2.0 (D-02에서 lock에 추가)

---

## 6. 성능 예산

전체 학습량은 10폴드 × (λ 탐색 4 × 100,000 + 본 학습 5 × 300,000) = 19.0M 스텝이다. 실제 크기(U_t 500종목)의 합성 데이터와 가짜 엔진으로 잰 값이다 (macOS arm64, 단일 프로세스).

| 항목 | 측정값 |
|---|---|
| 스냅샷 구성 | 신호일 104개에 0.22초 |
| 환경 1스텝 (가짜 엔진 포함) | 0.089 ms |
| PPO 학습 (SB3 기본값, CPU) | 약 2,400스텝/초. 시드 하나(300,000스텝)에 약 2.1분 |
| 전체 19.0M 스텝 | 약 2.2시간 + M5 엔진 시간 |

M5 `run_period`가 호출당 1 ms면 약 5.3시간이 더 든다. 폴드와 시드는 서로 독립이라 병렬로 돌릴 수 있다.

---

## 7. 테스트

실행: `python -m pytest tests/portfolio -p no:cacheprovider` (README 절차로 만든 `.venv`에서)

PPO를 실제로 학습하는 테스트 2개는 `slow`로 표시했다. `-m "not slow"`를 붙이면 뺀다. B의 Windows 환경에서 전체는 약 15초, `slow` 2개가 그중 약 10초다.

| 파일 | 확인 내용 |
|---|---|
| `test_params.py` | 부록 1 값과 키 이름 (R-14), FILL_ 값을 쓰면 멈춤 (R-08), 구현하지 않은 설정값 거부, `configs/prereg.yaml` 커밋 뒤 테스트 입력과 같은지 (커밋 전에는 건너뜀) |
| `test_risk_layer.py` | Step 4~5를 6.9절 식 그대로의 느린 구현과 비교, regime_mc = 1일 때 E_t 상한, t 이후 코스피 200 미사용 (R-03), Step 1~3·6 손계산, '미분류' 한 그룹, mode off, 무작위 입력 200건의 불변식, 위반 시 예외, positions를 io.py로 쓰고 읽기 (io.py가 있을 때) |
| `test_rl_env.py` | SB3·gymnasium 환경 검사, 관측 배치, 보유·U_t 밖 슬롯, 마스킹, 상위 20개와 min_weight, 전액 현금, 보상 식, 52스텝과 구간 경계, 슬롯 초과, 9.3절 표 → 스냅샷 |
| `test_rl_train.py` | **B-01 완료 기준** (`slow`): 합성 입력으로 λ 탐색 → 시드별 학습 → 체크포인트 저장·로드 → 결정적 운용이 끝까지 돈다. 저장한 체크포인트로 검증 IR 재현, 같은 시드면 같은 학습 결과 (`slow`). 연도 경계 모델 교체 |
| `test_vol_target.py` | **B-02 완료 기준**: 2016년 이후 종가를 바꿔도 vol_target_annual이 같다. 6.9절 식 그대로의 느린 구현과 비교, 2010년 종가는 첫 sigma_t 창의 21개만 사용, 기간이 모자라면 예외. compute_regime_mc가 M8 표와 같고 vol_target_annual 없이 계산됨, t 이후 종가 미사용, 2011년 초 이력 부족(D4), io.py로 index를 읽어 계산 (io.py가 없으면 오류 안내). 리포트의 데이터 지문이 2016년 이후 종가와 무관, 리포트 폴더 구성, 같은 run_id 덮어쓰기 거부 (R-11) |

`tests/portfolio/fakes.py`의 가짜 엔진과 합성 시장은 테스트 전용이다. 4.3절 상·하한가, 거래정지, 상장폐지, M9 청산은 구현하지 않았고, 실험과 보고서에 쓰지 않는다.

---

## 8. [결정 필요]

### D1. RL 학습 구간 관측 신호의 출처

```text
[결정 필요]
상황: 6.5절은 RL 에피소드를 폴드 Y의 학습 구간(2011년 ~ Y−3년)에서 뽑는다. 관측에 쓰는
      rank_score, crash_prob, crash_flag를 그 구간에서 어떤 모델로 만드는지는 정하지 않았다.
      A-06·A-07의 완료 기준은 테스트 연도 예측뿐이라, 지금 계획으로는 RL 학습·검증 구간의
      ML 신호가 없다.
관련 조항: 6.3절, 6.4절, 6.5절, 4.7절, A-06, A-07, B-06, B-07, R-04
선택지: A) 폴드 Y 모델(학습 구간으로 적합)이 학습 구간에 낸 표본 내 예측을 쓴다. A는 예측만
           추가하면 되지만, 표본 내 신호는 실제보다 정확해서 RL이 신호를 과신하도록 학습될 수 있다.
        B) 학습 구간 안에서 연도별 교차 적합 예측을 쓴다. 학습 구간의 각 연도를, 그 연도를 뺀
           나머지 학습 연도로 적합한 모델(폴드 Y에서 고른 하이퍼파라미터 그대로, 연도 경계 purge
           적용)로 예측한다. 폴드마다 학습 연도 수만큼 A의 추가 적합이 생긴다.
        C) 학습 구간에서도 베이스라인 신호(B1 순위, vol_60 플래그)를 쓴다. 추가 작업은 없지만
           학습과 테스트의 신호가 달라진다.
        공통: 검증 구간((Y−2)년, (Y−1)년)은 폴드 Y 모델의 표본 외 예측을 쓴다. 폴드별 신호는
           9.3절 열 구성 그대로 폴드마다 run_id를 따로 두어 signals/wf_{run_id}.parquet에
           저장한다 (스키마 변경 없음).
추천: B. 테스트 때와 같은 정확도의 신호로 학습해야 G5 비교(RL 대 버퍼 규칙)가 공정하다.
      A의 일정이 어려우면 A안으로 하고 결과 보고서에 한계로 적는다.
영향: A-06·A-07 산출물 범위와 A 작업량, B-06·B-07 입력, 9.3절 signals 파일 구성.
      A의 walk-forward(11/22) 전에 정해야 한다.
```

### D2. 보상의 DD 측정 기준

```text
[결정 필요]
상황: 6.5절 보상의 DD(t), DD(t′)가 어느 시점의 NAV이고 고점을 무엇으로 잡는지 정하지 않았다.
      R_net은 '체결일 종가 기준'으로 정해져 있다.
관련 조항: 6.5절 보상, 6.6절 MDD, G5
선택지: A) 체결일 종가(체결 직전) NAV로 재고, 고점은 에피소드 시작 이후 일간 NAV의 최댓값
           (6.6절 MDD와 같은 일간 기준). M5가 구간의 일간 NAV 최댓값(nav_max)을 돌려준다.
        B) 체결일 종가 NAV로 재고, 고점은 스텝 끝 NAV들의 최댓값 (주간 기준)
        C) 신호일 종가 NAV로 잰다
추천: A. R_net과 같은 시점이고, 평가 지표(MDD)와 같은 일간 기준이다. 지금 코드는 A이며,
      B로 바꾸려면 rl_env.py 한 줄만 고치면 된다.
영향: rl_env.py 보상, ExecutionEngine의 nav_max 필드(C), G5
```

### D3. RL 환경 세부값 (부록 1에 없음)

6.5절을 코드로 옮기면서 아래 값이 필요해 잠정값으로 두었다. G5 판정은 사전등록 시점의 설계로 하므로 (R-15), 11/9 전에 확정해야 한다.

| 항목 | 잠정값 | 이유 |
|---|---|---|
| (a) 행동(로짓) 범위 | [−10, 10], 로짓 = 행동 | SB3 환경 검사가 유한한 범위를 요구한다. 로짓 차가 20이면 비중 비가 약 5억 배라 현금 100%를 포함한 모든 배분을 표현할 수 있다. [−1, 1]이면 상위 20개 절단을 거쳐도 현금 100%를 만들 수 없다 (후보 40개일 때 현금 최대 약 0.58) |
| (b) U_t 밖 보유 슬롯의 rank_score 백분위, crash_prob, vol_20 | 0 | 신호·피처 행이 없다. 이 슬롯은 항상 마스킹되어 매도 대상이다 |
| (c) 후보가 60개를 넘을 때 | 보유하지 않은 후보를 rank가 나쁜 순서로 뺀다 | 상·하한가나 거래정지로 보유가 20을 넘으면 C_t가 60을 넘을 수 있다. 보유 종목은 슬롯을 받아야 비중을 정할 수 있다 |
| (d) 상위 20개 절단의 동률 | 앞 슬롯(rank가 좋은 종목) 우선 | 결과를 결정적으로 만든다 |
| (e) 52스텝 종료 | truncated(시간 제한)로 표시 | 에피소드 끝은 MDP의 종착 상태가 아니라 표본 추출로 자른 것이다. SB3가 마지막 상태의 가치를 부트스트랩한다 |
| (f) λ 선택 점수 | 100,000스텝 동안 10,000스텝마다 평가한 검증 IR의 최댓값. 동률이면 격자에서 앞선(작은) λ | 6.5절 "λ 선택의 '검증 IR'도 같은 정의다"를 체크포인트 선택과 같은 방식으로 읽었다 |
| (g) 학습·검증 신호일 | 4.7절 purge(h = 20)를 적용한 신호일 | R-04. RL 보상 구간(약 6거래일)은 h = 20 purge 안에 들어간다 |
| (h) 현재 비중 | 신호일 t 종가로 평가한 보유 비중 | t일에 알 수 있는 값이다 (R-03) |

```text
[결정 필요]
상황: 위 표의 값이 부록 1과 6.5절에 없다.
관련 조항: 6.5절, 부록 1 portfolio.rl, R-01, R-09, R-15, PM-07
선택지: A) 잠정값을 그대로 확정하고 부록 1 portfolio.rl에 추가한다
        B) 항목별로 다른 값을 정한다
        C) 확정하되 부록 1 대신 이 문서를 docs/preregistration.md에 포함한다
추천: A. 사전등록 커밋에 들어가야 G5 판정 설계가 고정된다.
영향: rl_env.py 상수, rl_train.select_lambda, 부록 1(PM-07), G5
```

### D4. 2011년 초 regime_mc 계산에 필요한 코스피 200 이력

```text
[결정 필요]
상황: Step 5는 신호일마다 코스피 200 종가 311거래일(252일 고점 창 + 60일 낙폭 창 − 1)이 필요하다.
      data.start_date가 2010-01-04라 2011년 초 약 60거래일(대략 3월까지)은 창을 채우지 못한다.
      regime_mc는 시장 공통 원값 피처이고, 6.2절은 원값 피처의 결측을 [결정 필요]로 보고하게
      한다. 지금 코드는 InsufficientHistory 예외를 낸다.
관련 조항: 6.9절 Step 5, 6.2절 결측 규칙, 부록 1 data.start_date, B-02, A-05
선택지: A) 2010-01-04부터 있는 이력만으로 계산한다 (창 앞부분을 잘라 쓴다)
        B) 결측으로 두고 그 신호일을 학습에서 뺀다
        C) 코스피 200 지수만 2009년부터 받는다 (부록 1 data 절과 A의 수집 범위 변경)
추천: C, 어려우면 A. C는 정의를 그대로 지키고 지수 하나만 더 받으면 된다. A를 택하면 B-02에서
      실제 데이터로 C와 값이 같은지 확인해 보고한다.
영향: compute_regime_mc(B-02), regime_mc 피처(A-05), RL 관측, M8 노출 조절
```

### D5. 4.7절 폴드 분할·purge 공용 함수의 담당

```text
[결정 필요]
상황: 4장은 공용 규약을 src/common/에 한 번만 구현하게 하지만, 5.3절과 9.1절은 4.7절 폴드 분할
      (학습·검증·테스트 신호일)과 purge 함수의 담당을 정하지 않았다. A(M3)와 B(RL)가 같은
      신호일 집합을 써야 한다.
관련 조항: 4장 서문, 4.7절, 5.3절, R-04, A-06, B-06
선택지: A) C가 guards.py 옆에 만든다
        B) A가 만든다 (M3 학습에서 먼저 쓴다)
        C) B가 만든다
추천: A. purge가 라벨 기간 h에 달려 있고, C가 이미 4.7절의 날짜 가드를 맡고 있다.
영향: src/common/ 파일 하나, A-06·A-07·B-06의 입력, 9.1절 담당 표
```

---

## 9. 다음 작업

| 작업 | 기한 | 이 설계에서 이어지는 부분 |
|---|---|---|
| B-02 | 11/8 | 함수와 테스트는 구현함. 남은 것: 실제 코스피 200 데이터로 값 계산 → PM에게 전달(부록 1 Q-12), D4 |
| B-03 | 11/13 | `buffer_rule_weights` |
| B-04 | 11/16 | 경계 사례 테스트(한 업종 20종목, 보유 5종목, regime_mc = 1, sigma_t = 3 × vol_target_annual), 2011~2025년 E_t·regime_mc 그림 |
| B-05 | 11/16 | M5 엔진 어댑터, 같은 비중일 때 버퍼 규칙과 체결·비용 결과 일치 |
| B-06 | 11/23 | 폴드별 스냅샷 구성(D1, D5), R-10 기록(`reports/{run_id}/`), 베이스라인 신호 리허설 |
