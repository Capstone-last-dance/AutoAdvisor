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
| `src/portfolio/risk_layer.py` | M8 Step 1~6, 불변식, positions 행 생성, t일 유효 업종표 | 구현. 경계 사례 테스트 추가 (B-04) |
| 〃 `compute_regime_mc(date)`, `compute_vol_target_annual` | 공개 함수, Q-12 값 계산 | 구현 (B-02). 값 0.157263을 부록 1에 기입했다 (4절) |
| `src/portfolio/vol_target_report.py` | B-02 결과 리포트: `python -m src.portfolio.vol_target_report --run-id {YYYYMMDD}_B_{설명}` → `reports/{run_id}/`에 config.yaml 사본, git_commit.txt, metrics.json (R-10). git_commit.txt에는 HEAD와 커밋되지 않은 파일(새 파일 포함)을 적는다. 세 리포트가 같이 쓴다 | 구현 (B-02) |
| `src/portfolio/market_state_report.py` | B-04 결과 리포트: `python -m src.portfolio.market_state_report --run-id {YYYYMMDD}_B_{설명}` → 위 세 파일과 `figures/e_t_regime_mc.svg`(2011~2025년 일별 E_t·regime_mc 그림), 그 원자료 `figures/e_t_regime_mc.csv` | 구현 (B-04). 실제 실행은 4절 |
| `src/portfolio/buffer_rule.py` | 버퍼 규칙 단계 1~5와 엔진 위 실행 (`run_buffer_rule`) | 구현 (B-03). C 엔진 연결은 `engine_adapter.py` |
| `src/portfolio/engine_adapter.py` | 9.3절 `prices`·`status`·`index`·`delisted`(+ DEC-28 `delisted_reused`)와 부록 1 `costs`·`execution` → C의 `ExecutionEngine` (`engine_inputs`, `load_engine_inputs`, `build_engine`, `cost_rates`) | 구현 (B-05). 규칙은 5.1절 |
| `src/portfolio/env_check.py` | B-05 완료 기준 확인: `run_full_episode`(끝까지 실행), `compare_buffer_and_env`(같은 비중이면 같은 체결·비용), `BufferWeightsEnv`, `ledger_counts` | 구현 (B-05) |
| `src/portfolio/env_check_report.py` | B-05 결과 리포트: `python -m src.portfolio.env_check_report --run-id {YYYYMMDD}_B_{설명}` → `reports/{run_id}/`에 config.yaml, git_commit.txt, metrics.json(완료 기준 (1)·(2), 입력 파일 SHA-256) | 구현 (B-05). 결과 `reports/20261110_B_env_check/` (5.1절) |
| `src/portfolio/interfaces.py` | B가 C(M5)와 PM(M9)에 기대하는 호출 형태 | 초안. C·PM 확인 필요 (5절) |
| `src/portfolio/weekly.py` | run_weekly 4단계 `m4_step`(w_raw), 5단계 `m8_step`(positions). forward 지난 보유 재생 `replay_holdings` | 구현. PM(M9)·D(지난 기록) 함수 연결 대기 (5.4절) |
| `src/portfolio/rl_data.py` | 9.3절 표 → 신호일별 스냅샷 | 구현 |
| `src/portfolio/rl_env.py` | gymnasium 환경: 관측, 행동 해석, M8 호출, 보상 | 구현. C 엔진 위 실행 확인 (B-05) |
| `src/portfolio/rl_train.py` | λ 선택, 본 학습, 검증 IR 체크포인트, 결정적 운용(체결한 신호일도 돌려준다) | 구현. 폴드 데이터 구성과 R-10 기록은 `rl_fold_report.py`·`rl_wf_report.py` |
| `src/portfolio/rl_fold_report.py` | B-06·B-07 폴드 하나의 λ 탐색과 PPO 학습: `python -m src.portfolio.rl_fold_report --run-id {YYYYMMDD}_B_{설명} --fold {Y} [--main-seeds 0] [--signals-run-id ... --label-end-dates ... --m9-alerts-run-id ...]` → `reports/{run_id}/`에 config.yaml, git_commit.txt, metrics.json, lambda_search.csv, λ·시드별 eval_log.csv와 best.zip (best.zip은 팀 드라이브, 지문은 metrics.json) | 구현. B-06 3.4절, B-07 입력 3.5절 |
| `src/portfolio/rl_wf_report.py` | B-07 10폴드 정책의 시험 연도 운용: `python -m src.portfolio.rl_wf_report --run-id ... --fold-runs ..._y{Y} --test-signals-run-id ...` → `positions/wf_{run_id}_s{seed}.parquet` 5개, `reports/{run_id}/`에 fold_seed_ir.csv(ANOVA 2 입력), weekly_excess.csv, metrics.json | 구현 (B-07). 3.5절 |
| `tests/portfolio/` | 단위 테스트와 테스트 전용 가짜 입력(`fakes.py`) | 구현 (7절) |

모듈 의존 방향은 `params` ← `risk_layer` ← `rl_data` ← `rl_env` ← `rl_train`이고, `buffer_rule`은 `rl_data`까지만 쓴다 (torch·gymnasium 불필요). `risk_layer.py`는 numpy와 pandas만 쓴다. A가 피처 `regime_mc`를 만들 때 import해도 torch나 gymnasium이 필요 없다.

---

## 2. 6.5절 대응: 버퍼 규칙

| 6.5절 | 코드 | 상태 |
|---|---|---|
| 단계 1 보유 집합 H | `buffer_rule_weights(holdings, ...)`. `run_buffer_rule`은 직전 `run_period`가 돌려준 실제 보유(`weights_obs` > 0)를 H로 쓴다. M9 청산이나 체결 실패로 목표와 달라도 실제 보유를 따른다 | 구현 |
| 단계 2 유지 조건 | 〃 rank ≤ `exit_rank_greater_than`, crash_flag = 0, U_t 소속, 쿨다운 아님. signals에 없는 보유 종목은 rank가 없으므로 판다 | 구현 |
| 단계 3 추가 | 〃 보유하지 않은 종목 중 rank ≤ `enter_rank_max`, crash_flag = 0, 쿨다운 아님을 rank 오름차순으로 `max_holdings`까지. 같은 rank는 종목코드 순 | 구현 |
| 단계 4 동일가중 | 〃 1 ÷ 종목 수. 종목이 없으면 빈 Series (현금 100%) | 구현 |
| 단계 5 M8 | `buffer_targets`: RL 환경과 같은 배열 경로(`risk_layer_weights`, `check_invariants`). `mode="off"`(G7)이면 M9 쿨다운도 끄고, full이 아니면 E_t = 1 | 구현 |
| 엔진 위 실행 | `run_buffer_rule(snapshots, engine, ...)`: 첫 신호일 현금 100%에서 시작해 신호일마다 단계 1~5 → `engine.run_period`. 마지막 신호일은 비중만 기록하고 체결하지 않는다 (RL `episode="full"`과 같음) | 구현. FakeEngine과 C의 `ExecutionEngine`(PR #8 헤드 a901420을 합친 상태) 위에서 실행 확인. 실제 데이터 연결은 B-05 |

- **입력**: RL과 같은 신호일 스냅샷(`rl_data.build_snapshots`)을 쓴다. 그래서 G5 비교에서 두 구성기의 신호, U_t, 쿨다운, 업종, E_t가 같다.
- **단계 3과 U_t**: 6.5절 단계 3은 U_t 소속을 따로 묻지 않는다. 하지만 rank가 U_t 안의 순위이고(6.4절), 스냅샷은 signals 종목이 U_t와 같은지 확인한다. 그래서 후보가 모두 U_t 종목이고 결과가 같다.

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
| 본 학습 | `train_fold` → `train_one`: 시드 0~4, 시드마다 300,000스텝 | 폴드 실행(`rl_fold_report`)은 `select_lambda`와 `train_one`을 직접 불러 시드를 고를 수 있다 (3.4절) |
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

### 3.4 폴드 하나 실행 (B-06, `rl_fold_report.py`)

- 설정 확인: `RLParams.from_prereg`가 `portfolio.rl.env` 8개(DEC-21)를 구현값(`params.RL_ENV_IMPLEMENTED`)과 대조하고, λ 격자가 오름차순인지 본다. `select_lambda`는 동률이면 격자에서 앞선 λ를 고르므로, 오름차순이어야 '동률이면 작은 λ'(D3(f))가 된다.
- 폴드 분할: A의 `make_folds(signal_dates, trading_days, h=20, test_years=(Y,))` 그대로다 (D5). 신호일은 universe의 2011~2025 날짜(B-05 리포트와 같은 검사)이고 거래일은 코스피 200 종가 날짜다. B-06 리허설의 purge 기준은 명목 t+1+h 거래일이다. B-07은 `--label-end-dates`로 C의 실제 라벨 종료일을 넘겨 A-06·A-07과 같은 `actual_end_dates`로 자른다 (3.5절).
- 환경: 학습은 폴드 학습 신호일의 `episode="random"`, 검증은 검증 신호일의 `episode="full"`이다. 둘 다 2011~2025 신호일 달력을 넘겨 purge 자리를 건너뛰는 스텝이 없다 (D3(g)). 검증 IR은 C의 `validation_ir`이다.
- 신호·피처·엔진: B-05 리포트 함수를 그대로 쓴다. B-06 리허설은 A의 베이스라인 신호를 쓴다. 베이스라인은 규칙 기반이라 학습 구간에서도 교차 적합이 필요 없다. M9 쿨다운·이벤트 청산은 비운다. ML 신호와 M9 경보는 B-07 인자로 넣는다 (3.5절).
- 학습: `select_lambda`(격자 4개, 시드 0, 100,000스텝) → 고른 λ로 `--main-seeds`의 시드마다 `train_one`(300,000스텝). 기본은 사전등록 시드 0~4 전부다.
- 완료 기준 판정(`criteria.lambda_selection_ran_to_the_end`): 격자의 λ가 모두 학습됐고, λ·시드마다 10,000스텝 간격 평가 기록이 빠짐없고 체크포인트가 있으며, 고른 λ가 '최고 검증 IR, 동률이면 작은 λ'를 따로 계산한 값과 같다.
- 기록 (R-10): config.yaml 사본, git_commit.txt(결과 폴더를 만들기 전), metrics.json(폴드 신호일과 purge 수, λ표, 시드별 최고 검증 IR, 입력 파일 SHA-256, 신호·피처 지문, 체크포인트 지문, 라이브러리 버전, 장치 cpu, 시간), lambda_search.csv, λ·시드별 eval_log.csv와 best.zip.
- 체크포인트 보관 (PM 11/10): best.zip은 git에 넣지 않고 팀 드라이브의 run_id 폴더에 올린다. 드라이브 파일이 정본이다. metrics.json의 `checkpoints.files`에 파일마다 크기, `sha256`(드라이브 파일 대조용), `policy_sha256`을 남긴다. `policy_sha256`은 정책 파라미터의 이름·모양·dtype·값만으로 만든 지문이다. zip 파일 SHA-256은 저장 시각 때문에 다시 학습하면 달라져서, 같은 가중치인지는 이 지문으로 비교한다 (6절). 같은 PC에서 다시 학습하면 지문이 같았다 (아래 리허설). 다른 PC에서 같은 가중치가 나오는지는 아직 확인하지 않았다.
- **리허설 결과 `reports/20261110_B_rl_y2016_v2/`** (코드 커밋 54dde55, `configs/prereg.yaml`과 같은 설정, A 11/8 데이터·베이스라인 신호. 입력 파일 SHA-256은 B-05 리포트와 같다. 체크포인트 5개는 팀 드라이브의 같은 이름 폴더에 둔다):
  - 폴드 2016: 학습 152개 신호일(2011-01-07 ~ 2013-11-29), 검증 101개(2014-01-03 ~ 2015-12-04). h = 20 purge로 각각 4개가 빠졌다. U_t는 평균 379종목(321~455)이다.
  - λ 탐색 (시드 0, 100,000스텝): 최고 검증 IR은 λ = 0이 0.856(40,000스텝), 0.5가 0.831(10,000), 1이 0.841(20,000), 2가 0.826(50,000)이다. λ = 0을 골랐다.
  - 본 학습 (λ = 0, 시드 0, 300,000스텝): 최고 검증 IR은 40,000스텝의 0.856이다. 처음 10번의 평가는 λ = 0 탐색과 값이 같다 (같은 λ·시드). 100,000스텝 뒤 최고는 0.776이라 더 학습해도 검증 IR이 오르지 않았다.
  - 완료 기준: λ 4개 모두 학습, 평가 기록(λ마다 10번, 시드마다 30번)과 체크포인트 완전, 고른 λ가 규칙과 같음 → 통과.
  - 시간: 준비 30초, λ 탐색 454초, 본 학습 336초 (합계 약 14분). 체크포인트는 개당 762 KB, 5개 3.7 MB다.
  - 재현: 첫 리허설과 λ표, 평가 기록 전부가 같다. 체크포인트 5개의 `policy_sha256`도 모두 같고, 파일 `sha256`은 모두 다르다 (같은 PC). 첫 리허설은 `20261110_B_rl_y2016`(02ed851)이고, best.zip이 git에 들어가 PR에서 뺐다. 11/10 스모크(스텝 1/5)와 같은 스텝의 평가값도 같았다.
  - 기록 메모: git_commit.txt에는 커밋 아래 `untracked:` 9줄이 있다. 첫 리허설의 결과 파일(설정 사본, 평가 기록, 지표)이 작업 폴더에 남아 있었기 때문이다. 이 실행은 `reports/`를 읽지 않아(자기 결과 폴더가 있는지만 본다) 결과에 영향이 없다. 코드와 설정은 커밋 그대로다.
  - 참고 (설계 변경 아님, R-15): 검증 IR이 10,000~50,000스텝에서 가장 높고 그 뒤 낮아졌다. λ 점수 차이(0.83~0.86)는 같은 λ 안 평가 간 흔들림(0.5~0.86)보다 작다. 검증 101주와 시드 하나로 고르는 사전등록 절차의 성질이다. B-07 보고서에 적을 사항이다.

### 3.5 10폴드 본 학습과 시험 연도 운용 (B-07, `rl_fold_report.py` + `rl_wf_report.py`)

- 입력 (11/10 확인. 인계본은 로컬 `files/`에 있고 git에 넣지 않는다):
  - 학습·검증 신호: A `signals/wf_20261110_A_rlsig_y{Y}.parquet` 10개 (DEC-08 B안). 학습 연도는 그 해를 뺀 교차 적합(`..._cross`)이고, 검증 연도는 폴드 Y 모델과 보정기(`..._fold`)다. 입력은 C 공식 라벨과 PM `ev_*`이고, `ss_regime` 미확정 구간은 0이다 (DEC-31).
  - 시험 신호: A `signals/wf_20261110_A_wfpm.parquet`. 시험 연도 Y의 행은 폴드 Y 모델의 예측이다 (`model_version` `..._y{Y}`). 폴드별 하이퍼파라미터가 rlsig 메타와 10개 폴드 모두 같아 같은 폴드 모델의 짝이다.
  - A 코드 커밋: 11/10판 rlsig·wfpm은 A #44 head `3adfcf2`로 만들었다 (A의 #47 리뷰, 11/11). B가 받은 wfpm metrics.json에는 커밋 키가 없어 A 답으로 기록한다.
  - 라벨 종료일: C `C02_labels_20261109_preholdout/end_dates.parquet` (manifest SHA-256 일치).
  - 신호일 대조 (A의 #47 리뷰 표와 같다, 11/11 다시 계산): 명목 t+1+h purge면 B 학습 신호일 56개가 A 파일에 없다 (폴드 2016 학습 152 vs 147, 폴드 2024 검증 100 vs 85). 실제 종료일 purge(`--label-end-dates`)면 10개 폴드의 학습·검증 신호일이 A 파일과 양방향 모두 차이 0이다. B-06 리허설은 B가 베이스라인 신호를 직접 만들어 명목 purge였다. `--signals-run-id`를 주고 `--label-end-dates`를 빼면 표를 읽기 전에 멈춘다 (PM 11/11 요청). 신호일이 다르면 결과 폴더를 만들기 전에 멈추는 검사도 그대로 있다.
  - 정제 표: A 11/8판. 로컬 `data/processed`와 8개 표의 SHA-256이 같다.
  - 대조: 실제 종료일로 자른 B의 폴드 학습·검증 신호일이 A 파일 날짜와 10개 폴드 모두 같다. 시험 신호일도 같다. 날짜마다 종목이 U_t와 같고 빈 값이 없으며, io 스키마를 통과한다. 실제 종료일 때문에 더 빠진 신호일은 폴드마다 학습 0~10개, 검증 0~15개다 (폴드 2016 학습 152 → 147, 폴드 2024 검증 100 → 85). 오래 정지된 종목 하나가 그 신호일 전체를 빼기 때문이다.
  - io는 9.3절 경로(`signals/wf_{run_id}.parquet`)에서만 읽으므로 인계본을 그 자리에 복사해 쓴다. A는 신호 파일을 git에 넣지 않는다 (#44).
- 폴드 학습 (`rl_fold_report.py`). 폴드마다 한 번 돌리고, 폴드끼리 독립이라 따로 돌려도 된다:
  `python -m src.portfolio.rl_fold_report --run-id {날짜}_B_rlwf_y{Y} --fold {Y} --signals-run-id 20261110_A_rlsig --label-end-dates {C 인계본}/end_dates.parquet [--m9-alerts-run-id {PM run_id}]`
  - 절차는 B-06 그대로다 (λ 탐색 → 시드 0~4 본 학습). 입력만 바뀐다. A 파일의 신호일이 B의 폴드 신호일과 다르면 결과 폴더를 만들기 전에 멈춘다.
  - metrics.json에 더 들어가는 것: 신호 파일 SHA-256과 구간별 `model_version` 행 수, 라벨 종료일 파일 SHA-256, 지연 때문에 더 빠진 수(`fold.purge.delay_purged`, DEC-09 (3) 변경의 보고 항목), M9 기록.
  - M9: PM `load_book(run_id)`의 `cooldown_at`을 스냅샷 쿨다운(행동 해석의 마스킹)에 넣는다. `engine_alerts(첫 신호일, 마지막 신호일)`는 엔진의 체결일 쿨다운·이벤트 청산에 넣는다. 체결일 d의 일정에는 d 전에 시작한 경보만 들어가므로, 한 번에 만들어도 앞 체결이 뒤 경보를 보지 않는다.
- 시험 운용 (`rl_wf_report.py`). 10개 폴드가 끝난 뒤 한 번 돌린다:
  `python -m src.portfolio.rl_wf_report --run-id {날짜}_B_rl_wf --fold-runs {날짜}_B_rlwf_y{Y} --test-signals-run-id 20261110_A_wfpm`
  - 폴드 폴더 10개 확인: task B-07, 완료 기준 통과, 사전등록 시드 전부, 시험 연도, config.yaml이 지금 설정과 같음, 학습 신호 run_id 앞부분·라벨 종료일·M9 경보가 폴드끼리 같음. 하나라도 다르면 멈춘다. 폴드 실행 커밋이 서로 다르면 멈추지 않고 기록하며, 완료 기준을 통과하지 않은 것으로 둔다.
  - 체크포인트: 드라이브에서 받은 `seed_{s}/best.zip`의 `policy_sha256`이 폴드 기록과 다르면 다른 가중치라 멈춘다.
  - 운용: 시드마다 2016년 첫 신호일에 현금 100%로 시작해 2025년 마지막 신호일까지 결정적 행동으로 운용한다. 신호일 t의 정책은 t가 속한 해의 폴드 정책이다. 연도 경계에서는 보유를 유지한 채 정책만 바꾼다 (4.7절). 마지막 신호일은 다음 신호일이 홀드아웃이라 체결하지 않는다. M9는 폴드 학습과 같은 경보를 쓴다.
  - 출력: `positions/wf_{run_id}_s{seed}.parquet` 5개. 9.3절 positions에는 시드 열이 없어 시드마다 파일을 나누고, constructor는 rl이다. `reports/{run_id}/`에는 `fold_seed_ir.csv`(ANOVA 2 입력 fold, seed, ir, 50행), `weekly_excess.csv`(seed, fold, signal_date, week = 체결 시작일, a_k), metrics.json(완료 기준, 폴드·체크포인트 기록, 시드별·평균 IR)을 쓴다.
  - IR: 폴드 Y의 IR은 Y년 신호일로 정한 주의 a_k로 C `validation_ir`를 계산한 B의 확인값이다. 12월 마지막 신호일의 주가 1월에 체결돼도 폴드 Y로 센다. G5 판정 지표는 C-07이 positions로 계산한다.
  - 완료 기준(`criteria.b07_complete`, "10폴드 × 5시드 완료, 설계 변경 없음"): 모든 폴드·시드 학습, 같은 커밋, M9 적용, 실제 종료일 purge, 모든 시험 신호일의 positions.
- 실데이터 스모크 (11/10, 스크래치, 공식 결과 아님. 커밋 f6909cd + 미커밋 B-07 코드, 실행마다 스텝 2,048로 줄임, M9 없음):
  - 폴드 10개 학습과 시험 운용이 끝까지 돌았다. 폴드마다 준비 약 25초다.
  - 시험 운용: 2016-01-08 ~ 2025-12-26 신호일 520개, 시드마다 519주 체결. positions 5개(각 10,920행)가 스키마를 통과했다. fold_seed_ir 50행, 체크포인트 50개 지문 일치, 시험 연도별 model_version 하나씩. 준비 27초, 운용 5초다.
  - 지연 purge 수는 위 대조값과 같다 (폴드 2016 학습 5개 ... 폴드 2024 검증 15개).
  - 공식 실행은 폴드마다 1.9M 스텝이다 (λ 탐색 400,000 + 시드 5 × 300,000). B-06 실측 1.2 ms/스텝이면 폴드당 약 38분, 10폴드를 차례로 돌리면 약 6.4시간이다. 폴드를 동시에 여러 개 돌릴 때의 속도는 재지 않았다.
- **막힌 경우 하나 (C 엔진, 11/10 발견)**: 신호일 t가 마지막 가격일인 종목이 U_t에 있고 목표에 들어가면, C 엔진이 체결일 t+1의 목표 종목 가격을 읽다가 멈춘다 (`_run_interval`의 `_prices(first_i, 보유 ∪ 목표)`, "평가 가격이 없습니다").
  - 엔진은 폐지일 뒤의 매수를 `forbidden`으로 막게 되어 있지만, 그 검사가 가격 읽기보다 뒤에 있다. 보유 종목은 마지막 가격일에 정리매매 가격으로 청산돼서 문제가 없다.
  - 데이터 전체(2011~2025 신호일 × U_t)에서 한 건이다: 2023-08-11 신호일의 048260(2023-08-14 폐지, 마지막 가격일 8/11). 폴드 2025 검증 신호에서 8/4에 4위, 8/11에 18위라 RL 후보다. 순위로는 버퍼 규칙 진입 기준(20위)에도 들지만, 폴드 2025 검증 신호일 92개로 버퍼 규칙을 실제로 돌리면 이 종목을 목표에 넣지 않는다 (91구간 중 0건, 11/11 확인). wfpm(폴드 2023 시험)에서는 42위다. A도 전수 확인에서 같은 결과를 냈다. 신호일이 마지막 가격일인 쌍 444건 중 443건은 데이터 마지막 날(2026-11-06)이고, 과거 구간은 이 한 건뿐이다. A는 universe를 그대로 두고 엔진에서 체결일 기준으로 거르는 데 동의했다 (#47 리뷰).
  - 스모크에서는 폴드 2025가 여기서 멈췄다. 스크래치에서만 엔진을 감싸 '체결일에 이미 폐지된 목표는 주문하지 않고 현금'으로 두자 나머지가 끝까지 돌았다. 막힌 주문은 이 한 건뿐이었다.
  - B는 C 파일을 고치지 않는다 (R-17). C에 엔진 수정을 요청했고, C가 PR #52(11/11, head 0ab3067, 미병합)로 고쳤다.
    - 수정 내용: 첫 체결일에 이미 폐지된(폐지일 < 첫 체결일 < 재상장일) 미보유 목표는 가격을 읽기 전에 빼고 `cooldown_or_delisted`로 기록한다. 그 비중은 현금이고, 회전율은 원래 목표 기준으로 센다. 쿨다운으로 거부된 매수를 회전율에 세는 기존 규칙과 같다.
    - B 확인 (11/11, 스크래치 worktree = #52 head + B 작업): 빠른 테스트 767개와 slow 3개 통과. 재현 호출은 NAV 1.0, 현금 100%, 회전율 0.025, 048260 거부. 감싸지 않은 엔진으로 폴드 2025 스모크가 끝까지 돌았다 (58초, 이 경우 2번 나와 모두 거부). 감싸서 돌린 11/10 스모크와 λ 탐색표가 같고, 검증 IR은 시드 1에서만 1.6e-8 다르다. 거부된 비중이 RL 관측 `turnover`에 들어가서 생기는 차이다.
    - 이 수정은 엔진 `delistings` 날짜가 마지막 가격일이라는 전제에 기댄다. 엔진을 만드는 곳은 B 어댑터뿐이고, 어댑터가 A `delist_date`(다음 거래일)를 마지막 가격일로 바꿔 넘긴다 (5.1절).
- 뉴스 피처 (PM 결정 11/10): 11/10 인계본(rlsig·wfpm)과 A-08 최종 모델에는 뉴스 피처(PM-05)가 없다. 판정 실행은 뉴스 반영판 신호를 받은 뒤 시작한다. 뉴스 피처는 11/16 목표로 A에 넘어간다. A는 같은 형식(파일 이름 `wf_{run_id}_y{Y}`, 9.3절 signals 열, 폴드 구성, `_cross`/`_fold` 접미사)으로 준다 (#47 리뷰). 새 인계본이 오면 위 대조(폴드·시험 신호일, 종목, 하이퍼파라미터)를 다시 하고 스모크를 한 번 더 돌린다.
- 판정 실행 전에 남은 것:
  - 뉴스 반영판 rlsig·wfpm (A, PM 11/16 목표).
  - M9 백테스트 경보(`alerts/wf_{run_id}.parquet`, PM)가 언제 나오는지, 아니면 M9 없이 돌리고 이탈로 기록할지. PM #51에는 forward 날짜별 파일만 있다. 이 파일에도 `generated_at`(= `available_at`)이 필요하다 (DEC-32).
  - 위 엔진 수정 C PR #52 병합 (B 확인 끝).
  - 인계 신호 파일 11개가 `signals/`에 미추적 파일로 있어 git_commit.txt에 `untracked:`로 남는 문제. 파일 SHA-256은 metrics.json에 남는다.

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
- 창을 채울 코스피 200 이력이 없으면 `InsufficientHistory`를 낸다. 단, rebound_t = 0이면 bear_t의 창(종가 311개)이 차지 않아도 regime_mc_t = bear_t × 0 = 0으로 계산한다 (D4).
- vol_target_annual의 기간: 시작 연도는 `data.first_train_signal_year`(2011), 끝 연도는 walk-forward 첫 테스트 연도의 전년(2015)으로 부록 1에서 만든다. 2011년 첫 거래일의 sigma_t도 6.9절 정의대로 직전 20개 수익률을 쓰므로 2010년 12월 종가 21개가 들어간다. 부록 2 B-02 완료 기준의 '2011~2015년 데이터만'은 6.9절 정의(우선순위가 높다, 0.1절)에 맞춰 '2016년 이후 데이터 미사용과 이 21개 밖의 2010년 데이터 미사용'으로 확인한다.
- vol_target_annual 계산 결과 (2026-11-03, `reports/20261103_B_vol_target/`):
  - 값: **0.157263** (0.1572629338). 2011~2015년 1,236거래일의 sigma_t 평균이다. 연도별 평균은 0.249, 0.169, 0.132, 0.110, 0.126
  - 입력: A-02 `index.parquet`(11/3 받은 파일, A PR #1 병합 전). 2010-12-03 ~ 2015-12-30 종가 1,256개를 썼고, 그 sha256을 metrics.json에 남겼다
  - 설정: `configs/prereg.yaml`이 비어 있어(PM-07) 부록 1과 같은 값을 담은 `tests/portfolio/prereg_subset.yaml`로 계산했다. 쓰는 키는 `vol_lookback_days`, `calendar.annualization`, `data.first_train_signal_year`, `splits.walk_forward_test_years`다
  - A-02 지수가 바뀌면 지문이 달라지므로 새 run_id로 다시 계산한다
- 완료 기준 (3) 그림 (B-04): `market_state_report`가 2011년 첫 거래일 ~ 2025년 마지막 거래일의 매 거래일 E_t와 regime_mc를 계산한다.
  - 그림은 SVG로 직접 쓴다. requirements(D 소유)에 그림 라이브러리가 없다.
  - 기간: 시작은 `data.first_train_signal_year`, 끝은 walk-forward 마지막 테스트 연도다. 2025년보다 늦은 종가는 계산 전에 잘라낸다.
  - 멈추는 경우: 하루라도 E_t·regime_mc가 결측이면 `InsufficientHistory`, vol_target_annual이 `FILL_`이면 `PreregValueMissing`.
  - A의 `index.parquet`는 11/3 21:29 파일부터 KOSPI200만 남아 io 검사를 통과한다. KOSDAQ150은 `index_kosdaq.parquet`로 따로 뺐다 (DEC-27). 이 파일로 B-02 값과 입력 지문이 그대로 재현됐다.
- 그림 결과 (2026-11-04, `reports/20261104_B_m8_states/`):
  - 실행: 커밋 f2e8cf3에서 미커밋 파일 없이 돌렸다. 설정은 `configs/prereg.yaml` 초안(de685a8)이고 vol_target_annual은 0.157263이다.
  - 입력: `load_kospi200()`(io 경로). 2010-01-04 ~ 2025-12-30 종가 3,939개, sha256 1607e468…를 metrics.json에 남겼다.
  - 3,688거래일 모두 결측이 없다. bear 창이 차지 않는 2011년 초 59일은 rebound = 0이라 regime_mc = 0이다 (D4).
  - regime_mc = 1은 55일, 14개 구간이다 (2011년 10~11월, 2020년 4~6월, 2025년 5~7월 등).
  - E_t 평균은 0.894다. 하한 0.30에 걸린 날 19일, 국면 상한 0.70에 걸린 날 40일이다.
  - 11/9 공식 사전등록에서 `risk_layer`, `vol_lookback_days`, `calendar.annualization`, `data.first_train_signal_year`, `splits.walk_forward_test_years` 중 하나라도 바뀌면 새 run_id로 다시 만든다.

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

B-05 어댑터(`engine_adapter.py`)가 A의 9.3절 파일을 엔진 입력으로 바꾸는 규칙 (11/4 PM 결정 반영):

| 엔진 입력 | 만드는 법 | 근거 |
|---|---|---|
| `halted` | 일별 `status.is_halted`. v1은 `volume == 0`이다. status 행이 없는 칸은 결측(알 수 없음)이고, 종가가 있는데 행이 없으면 멈춘다 | DEC-24, DEC-26 |
| `closes` | `prices.close` 그대로 넘긴다 | 정지일 평가 가격은 엔진이 마지막 거래일 수정종가로 바꾼다 (DEC-25, PR #8 c82fa1b) |
| `bm_close` | `index`에서 `index_code` = KOSPI200. 가격 표의 거래일이 하나라도 빠지면 멈춘다 | DEC-27 |
| `delistings` | `delisted`와 `delisted_reused`를 합친다. 둘 다 `io.read_table`로 읽는다 (D PR #18). `delist_date`를 마지막 가격일(전 거래일)로 옮기고, 원종가 `last_price`는 그날 close ÷ close_raw로 수정종가 단위로 바꾼다. 결측이면 0으로 청산 | PR #8 docstring, DEC-28, 부록 1 `execution.delisted_without_price_value` |
| 가격 공백 검사 | 첫 가격과 마지막 가격 사이에 빈 칸이 있는 종목은 빈 칸 첫날이 그 종목의 `delist_date`여야 한다. 아니면 멈춘다. 빈 구간이 둘 이상이어도 멈춘다 | DEC-28. `delisted_reused`가 비었거나 일부 기간으로 만들어지면 옛 회사 폐지가 빠진다. 그 종목을 들고 있으면 엔진이 멈추지만, 들고 있지 않으면 오류 없이 결과만 달라진다 (A 리뷰, PR #21) |
| 비용 | 부록 1 `costs`. 기본 실행은 배율 1, 스트레스 실행만 `stress_multiplier` | 4.8절 |
| 쿨다운·이벤트 청산 | M9(PM) 함수가 나오기 전까지 비어 있다 | 5.2절 |

- 엔진은 실행마다 새로 만든다 (`rebalance_count`가 인스턴스에 쌓인다). 표 변환(`EngineInputs`)은 한 번만 한다.
- 부록 1의 `costs.cash_return`이 0이 아니거나, `execution.delisted_without_price_value`가 0이 아니거나, `calendar.execution`이 다음 거래일 종가가 아니면 멈춘다. 엔진이 그 동작만 구현한다.
- 폐지 뒤 가격이 다시 나타나는데 엔진이 새 상장으로 인식하지 않는 코드가 있으면 멈춘다 (DEC-28은 두 건만 다룬다).
- 11/4 실제 데이터(로컬 A 파일): 4,141거래일 × 3,697종목, 빈 칸 39.0%, 정지 칸 288,431, 폐지 930건. 표 읽기·변환 15.5초, 엔진 생성 0.26초.
  - `delisted` 930건 모두 마지막 가격일이 `delist_date`의 전 거래일이고, `last_price` = 그날 원종가 = 수정종가다.
  - DEC-28 옛 회사 두 건(A 154b089의 `build_delisted_reused`로 만듦)은 원종가와 수정종가가 다르다. 101970은 830원 → 18,639.16, 036220은 3,500원 → 20,000이다. 원종가 그대로 청산하면 보유분이 각각 −96%, −83%로 잘못 평가된다. 그래서 단위를 바꾼다.
  - 가격 공백 검사 (11/8): 11/4 가격에서 중간이 비는 종목은 036220(2016-05-09~2024-03-12)과 101970(2015-03-17~2025-03-27) 둘뿐이다. 빈 칸 첫날이 A가 11/8에 넘긴 `delisted_reused`의 `delist_date`와 같아 통과한다. 그 파일이 없거나 비어 있으면 036220에서 멈춘다.
  - B는 종목의 절대 가격 수준을 쓰지 않는다. 엔진 수량은 소수이고, B 입력은 수익률·순위·코스피 200뿐이다. 그래서 A가 PR #21에서 알린 재사용 코드 옛 회사 구간의 수정계수 상수(5.71배, 22.46배)와 11/8 공지의 수정종가 소급 변경(11종목)은 이론상 B 결과를 바꾸지 않는다. 둘 다 한 종목의 한 구간 전체에 같은 배수가 걸려 그 구간의 수익률이 같기 때문이다. 실제 확인은 새 데이터로 B-05 리포트를 만들 때 한다.

- 정지일 평가 가격 (DEC-25):
  - 11/4 A 데이터에서 거래량 0인 288,431행 가운데 수정종가가 마지막 거래일과 다른 행은 19,294행이다 (보통주 1,621).
  - 직전 신호일 U_t에 든 종목은 그중 0행이라, 지금 데이터에서는 결과가 달라지지 않는다.
  - PR #8 c82fa1b부터 엔진이 직접 마지막 거래일 가격으로 평가한다 (`test_halted_valuation_uses_last_traded_close_even_if_source_close_changes`). 그래서 어댑터는 가격을 채우지 않는다.
- DEC-25 재개일 처리는 PR #8(0ba98eb, c82fa1b)에서 합성 시장으로 모두 확인했다:
  - 재개 첫날에는 목표를 넘는 만큼만 판다.
  - 매수는 다시 시도하지 않는다.
  - 재개일이 하한가 근접이면 다음 신호일에 다시 판단한다.
  - 재개 전에 새 신호가 나오면 새 목표를 따른다.
  - 회전율은 원래 체결일 목표로 한 번만 센다 (정지가 없을 때와 같은 값). 매도 비용은 재개일 원장에 남는다.
- DEC-28 재사용 코드(101970, 036220):
  - 옛 회사 보유분은 폐지일에 정리매매 가격으로 청산된다.
  - 0ba98eb에서는 폐지일 뒤에도 같은 코드를 계속 폐지 종목으로 봐서 새 회사 매수가 `cooldown_or_delisted`로 막혔다.
  - C가 c82fa1b에서 고쳤다. 두 코드에 한해, 폐지 뒤 가격 공백 다음의 첫 가격일부터 폐지 제한을 푼다 (`relisting_index`). 그날은 전일 종가가 없으므로 가격 제한 판정을 하지 않는다.
  - B 재현(옛 회사 D4 폐지, 새 회사 D10 상장)에서 새 회사 0.3 매수가 체결됐다. B main과 합친 전체 테스트는 336 통과, 1 건너뜀이다.
  - B 코드(`rl_env.py`, `buffer_rule.py`)는 보유를 엔진 상태에서만 읽어, 종목코드별로 남는 기록이 없다.
  - 어댑터로 넘기면 옛 회사는 폐지일에 청산되고 새 회사는 살 수 있다 (`test_reused_ticker_old_company_liquidated_and_new_company_buyable`). 실제 데이터에서도 엔진이 두 코드를 새 상장으로 인식한다.

B-05 완료 기준 확인 (`env_check.py`):

- (1) `run_full_episode`: RL 환경(`episode="full"`)이 첫 신호일부터 마지막 신호일까지 엔진으로 체결한다.
- (2) `compare_buffer_and_env`: 행동 해석 자리에 버퍼 규칙 단계 1~4의 w_raw를 넣은 RL 환경(`BufferWeightsEnv`)과 `run_buffer_rule`을 각자 새 엔진에서 돌린다. 목표 비중, 일별 체결(수량·금액·비용)과 거부 이유, 일별 NAV·현금, 구간 초과수익·회전율을 비교한다.
- 합성 시장(테스트, 157개 신호일, 정지 칸 1.8%, 보유 중 폐지 1건): 차이는 모두 0이다. 정지 거부 119번, 폐지 청산 1번이 실제로 일어났다.
- 실제 데이터 미리보기 (11/4 스크래치, 공식 결과 아님):
  - 조건: B main과 A PR #1(154b089)을 임시로 합친 폴더에서 돌렸다. 신호는 A의 `baseline_signals`, vol_20은 A의 정의대로 B가 계산했다. 쿨다운은 비어 있다.
  - 2011-01-07 ~ 2025-12-26의 781개 신호일(U_t 평균 419종목)이 행동 0과 무작위 행동 모두 끝까지 돌았다. 스텝당 0.55 ms다.
  - (2)의 차이는 모두 0이다. 체결 12,277건이 있었고, 거부는 정지 71번, 하한가 근접 5번, 상한가 근접 2번이었다. 보유 중 폐지는 없었다.
- 리포트 (`env_check_report.py`, 11/9):
  - 표는 io로 읽는다 (prices, status, index, delisted, delisted_reused, universe, sector).
  - 신호일은 universe의 2011-01-01 ~ 2025-12-31 날짜다. 주마다 마지막 거래일과 같은지 확인한다. 끝날이 휴장일이어도(2025-12-31) 그 주를 잘못 세지 않는다.
  - 신호는 A의 `baseline_signals`를 신호일마다 U_t로 부른다. vol_20은 A의 `compute_stock_features`·`rank_and_fill`, mkt_vol_20·mkt_ret_20은 A의 `compute_market_features`, regime_mc는 B의 `market_state_table`로 만든다.
  - M9 쿨다운·청산은 비운다 (M9 백테스트 경보 전). 무작위 행동 시드는 0이다.
  - 입력 파일 SHA-256과, 만든 신호·피처의 내용 지문을 남긴다.
- 리허설 (11/9, 스크래치. B main 6506579 + A PR #35, A 11/8 데이터, 공식 결과 아님):
  - 781개 신호일(2011-01-07 ~ 2025-12-26, U_t 평균 419, 321~494종목)이 행동 0과 무작위 행동 모두 끝까지 돌았다.
  - (2)의 차이는 모두 0이다. 체결 12,277건, 정지 거부 71번, 하한가 근접 5번, 상한가 근접 2번이다. 보유 중 폐지는 없었다. 11/4 미리보기와 같다.
  - 전체 73.6초 걸렸다 (베이스라인 신호 37초, 스냅샷 5초).
- **공식 결과 `reports/20261110_B_env_check/`** (코드 커밋 5b2f1b3, 미커밋 파일 없음. A PR #35 병합 뒤 main, A 11/8 데이터, `configs/prereg.yaml`과 같은 설정):
  - (1) 781개 신호일(2011-01-07 ~ 2025-12-26, U_t 평균 419, 321~494종목)을 행동 0과 무작위 행동(시드 0) 모두 끝까지 돌았다. 끝 NAV는 행동 0이 1.6557, 무작위가 0.7552다.
  - (2) 목표 비중, 체결, NAV, 초과수익, 회전율 차이가 모두 0이다 (허용 오차 1e-12). 체결 12,277건, 정지 거부 71번, 하한가 근접 5번, 상한가 근접 2번, 보유 중 폐지 0번이다.
  - 입력 7개 표의 파일 SHA-256과 만든 신호·피처의 지문이 리허설과 같다.
  - 첫 실행(be43ba1)은 리포트 코드가 아직 git에 없던 상태라 `git_commit.txt`에 빠졌다. `_git_commit`이 새 파일도 적게 고친 뒤 커밋한 상태에서 같은 run_id로 다시 만들었다. 숫자는 실행 시간만 다르다.

### 5.2 M9 쿨다운 (PM 장지원)

6.10절(PR #16 반영본)을 따른다.

- 쿨다운 기간: 경보 근거 공시의 `available_at`이 속한 KST 날짜의 다음 KRX 거래일부터 20거래일이다. 같은 사건의 후속 공시는 기존 종료일을 유지하고, 별개 사건은 각각 20거래일을 적용한다. 걸린 기간이 하나라도 있으면 진입 금지다.
- 사건 연결과 거래일 계산은 M9(PM)가 한다. B는 결과인 종목 집합만 쓰므로, `rcept_no` 형식(`KIND:<acptno>` 포함)이나 같은 경보가 두 출처에서 온 것은 B 결과를 바꾸지 않는다.

| 쓰는 곳 | 형태 | 포함하는 경보 |
|---|---|---|
| 신호일 t의 목표 (버퍼 규칙 단계 2·3, RL 마스킹, M8 Step 1) | `CooldownFunction(t) -> 종목코드 목록`: 체결일 t+1에 진입 금지인 종목 | `available_at < cutoff_at`(t)인 경보만 (R-03) |
| 체결 (C 엔진 `cooldowns`, `event_liquidations`) | 체결일 → 종목코드 집합 | 그 체결일에 걸린 모든 경보 |

- t일 컷오프 뒤에 공개된 경보(금요일 저녁·주말 공개 포함)도 쿨다운은 t+1부터 걸린다. 신호일에는 쓸 수 없는 정보라 목표에는 남을 수 있다. 그 매수는 엔진이 체결일에 거부하고(현금으로 남음), 보유 종목이면 엔진이 t+1 종가에 청산한다 (6.10절 처리: 보유 종목).

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
- forward test positions (11/20부터, 6.8절):
  - `run_weekly` 4단계(M4)는 `ctx.memory["w_raw"]`(종목코드 → 비중 Series)를 채우고, 5단계(M8)는 `ctx.write_table("positions", df, date=t)`로 쓴다. 입력은 `ctx.read_table`로 읽는다. D가 이 형태에 동의했다 (11/8). D가 B PR 병합 뒤 아래 entry를 STEPS에 넣는다 (11/9).
  - `constructor`는 RL 최종 정책이 나오기 전까지 `buffer`다.
  - 버퍼 규칙 단계 1의 H는 백테스트와 같이 실제 보유다. positions는 목표 비중이라, 정지·상하한가로 체결되지 않으면 실제 보유와 다르다 (PM 지적, 11/27 기록부터 영향).
  - 그래서 태그 `ft-{날짜}`에 기록된 목표 비중을 11/20부터 순서대로 C의 `run_to_signal_close`(PR #22)에 넣고, 현금 100%에서 시작해 직전 신호일 t′ 종가의 보유를 H로 쓴다. `run_period`는 t′+1 가격이 있어야 하는데, forward 실행(t′+1 09:00 KST 마감, DEC-04)에는 그 가격이 없다.
  - 태그에서 지난 positions를 날짜순으로 읽는 함수는 D의 `src.pipeline.forward_test:past_positions(t)`다 (PR #31). 11/20은 첫 기록이라 H가 비어 있다.
  - 기록이 없는 주(스텁 등으로 커밋되지 않은 주)는 리밸런싱 없이 직전 보유를 유지한 것으로 재생하고 경고한다 (11/9 변경). C의 사후 성과(`evaluate_forward_record`, PR #22)도 기록된 주만 이어 재생하므로 둘이 같은 보유를 쓴다. 처음 안은 실패였는데, 그러면 한 주가 빠졌을 때 그 뒤 기록이 모두 멈춘다.
- 구현 (`src/portfolio/weekly.py`, 11/8):
  - entry는 `src.portfolio.weekly:m4_step`(4단계)과 `src.portfolio.weekly:m8_step`(5단계)이다.
  - 버퍼 규칙은 백테스트와 같은 `buffer_rule_weights`를, M8은 `apply_risk_layer`를 쓴다. signals 종목이 U_t와 다르면 멈춘다 (`build_snapshots`와 같은 검사). index에 신호일 t의 코스피 200 종가가 없어도 멈춘다. 그대로 두면 그 주의 신호일이 t 전날로 잡혀 지난 신호일과 M8 입력이 조용히 틀어진다 (A #36 리뷰, 11/10).
  - 지난 신호일은 KOSPI200 거래일에서 주마다 마지막 거래일로 센다 (`calendar.signal_day`). 지난 기록 날짜는 positions_from부터 t 전까지의 신호일이어야 하고, 아니면 멈춘다.
  - 다른 역할 함수는 `weekly.py` 위쪽 세 상수에 "모듈:함수"로 연결한다. 형태는 `interfaces.py`에 있다. 모듈(또는 상위 패키지)이나 함수 자체가 없으면 담당 PR 병합 여부를 적은 `EntryNotFound`를 낸다. 모듈은 있는데 그 안의 import가 실패하면(하위 의존성 없음 등) 원래 오류 그대로 멈춘다 (PM #36 리뷰, 11/10).

    | 상수 | 담당 | 형태 | 없을 때 |
    |---|---|---|---|
    | `M9_COOLDOWN_ENTRY` | PM | `CooldownFunction`: 신호일 t → t+1 진입 금지 종목. `src.events.alerts:cooldown_at` (PR #34) | '없음'은 두 경우뿐이다 (PM #36 리뷰, 11/10): (1) 연결 전이거나 모듈·함수 자체가 없다(PR 미병합) (2) M9가 `FileNotFoundError`를 냈고 `alerts/`에 날짜별 경보 파일이 하나도 없다(`io.list_dates("alerts")`가 빈 목록, M9를 실행하지 않은 상태). PM이 이 범위를 승인했다 (#36, 11/10). 남은 드문 경우가 하나 있다. 목록을 본 직후 마지막 파일이 사라지면 지금은 (2)로 처리된다. PM #51(11/10, 미병합)이 전용 예외 둘을 둔다: `AlertsNotStarted`(날짜별 파일이 하나도 없음)와 그 상위 `AlertsNotCollected`(필요한 날짜 파일 없음, 미수집). 둘 다 `FileNotFoundError` 하위다. #51이 병합되면 B는 (2)를 'M9가 `AlertsNotStarted`를 냈다'로 바꾸고 `AlertsNotCollected`와 다른 `FileNotFoundError`는 잡지 않는다 (PM 11/10 요청). 경보 파일이 없을 때 난 다른 `FileNotFoundError`도 그때부터 멈춘다. 바꾼 코드와 연결 테스트는 #51 위에서 확인해 두었다 (아래 '실행 순서 확인'). PM은 11/20 전에 필요한 구간의 달력일마다 0행 파일까지 채운다. 그 뒤로 날짜 하나가 빠진 경우는 미수집이라 (2)가 아니고 멈춘다. 이때 쿨다운 없이 계산하고 기록한다 (PM 11/9 결정: 기록을 미루지 않는다). 경고 로그와 `ctx.memory["deviations"]`에 이유와 함께 남기고 PM이 deviation log에 적는다. `RECORD_WITHOUT_M9 = False`면 `4(M9 쿨다운 없음)`을 스텁으로 표시해 7단계가 커밋하지 않는다 (6.12절). 경보 파일이 있는데 난 `FileNotFoundError`(목록의 파일을 못 엶, M9가 읽는 index가 없음 등), M9 모듈 안의 import 실패, 그 밖의 M9 오류는 틀린 경보로 기록하지 않도록 그대로 멈춘다 |
    | `M9_ENGINE_ENTRY` | PM | `EngineAlertsFunction`: (첫 기록 신호일, t) → 체결일별 cooldowns, event_liquidations. `src.events.alerts:engine_alerts` (PR #34) | 재생할 때만 쓴다. 없으면 M9 청산·체결일 쿨다운 없이 재생하고 위와 같이 남긴다 |
    | `RECORDED_POSITIONS_ENTRY` | D | `RecordedPositionsFunction`: t → t 전 기록의 positions (신호일 → 표). `src.pipeline.forward_test:past_positions` (PR #31) | 첫 기록(positions_from)에는 필요 없다. 그 뒤에는 실패한다 |

  - **지난 체결 재생의 경보 일정 (PM #36 리뷰 → PM 결정 11/10, B 1안)**: 재생은 그 시점의 경보 파일로 처음 기록부터 체결 지도를 다시 만든다. 그대로 두면 M9 없이 체결한 주나 체결 뒤 보충된 경보(예: 11/21 토요일 경보를 11/23 체결 뒤 보충)가 지난 체결에 소급된다. C 사후 성과(`evaluate_forward_record`)도 같은 `engine_alerts`를 써서 둘은 같지만, 둘 다 당시 적용한 일정과 다르다. ft 태그에는 signals·positions만 커밋되므로 B 코드만으로는 막을 수 없다. PM 결정:
    - alerts에 `generated_at`(KST) 열을 둔다 (D가 io 스키마에 추가). 백테스트 파일은 `generated_at = available_at`이라 결과가 바뀌지 않는다.
    - 같은 경보를 다시 만들어도 첫 생성 시각을 유지하고, 이미 낸 경보는 지우지 않는다.
    - 늦게 생성된 경보는 버리지 않는다. 생성 뒤 첫 체결일에 청산하고 남은 쿨다운을 적용한다. 쿨다운이 끝난 뒤 만든 경보는 청산하지 않는다 (DEC-32, PM #51 d3d28ac에서 11/10 답장대로 고침). 지난 체결일로 되돌아가지는 않는다.
    - 신호 쪽 쿨다운 `cooldown_at(t)`에는 t 15:30 전에 이용 가능하고(`available_at`) 그 전에 만든(`generated_at`) 경보만 들어간다 (#51 d3d28ac). 그 뒤에 만든 경보 종목은 목표에 남고, 엔진의 체결일 쿨다운·청산이 처리한다.
      - 엔진 확인 (11/11, 스크래치, #51 d3d28ac): 금요일 t 10:00 경보를 다음 날 08:05에 만들면 `cooldown_at(t)`에서 빠진다. `engine_alerts`는 다음 거래일에 쿨다운과 청산을 둘 다 건다. 그 종목을 보유했으면 그날 매도되고, 보유하지 않았으면 매수되지 않고 그 비중은 현금이다.
      - 기록 차이: 미보유 목표를 막을 때 같은 날 청산 이벤트가 목표를 0으로 먼저 바꾼다. 그래서 장부 `rejected`에 사유가 남지 않고 회전율에도 들어가지 않는다. 쿨다운만 걸린 경우는 `cooldown_or_delisted`로 남고 회전율에 들어간다. 결과(현금)는 같다. C 엔진 기록 방식이라 C에 알린다.
      - forward와 백테스트의 차이: D-06은 달력일 d의 경보를 d 다음 날 08:00 이후에 만든다. 그래서 신호일 t의 00:00~15:30 경보는 forward에서는 언제나 `cooldown_at(t)`에서 빠진다. 백테스트 파일은 `generated_at = available_at`이라 같은 경보가 신호 쪽에서 걸러지고, 버퍼 규칙이 다음 순위 종목으로 채운다. forward는 그 비중이 한 주 동안 현금이다. 시각 모순(미래 정보 사용)은 아니지만, '같은 신호일이면 forward positions = 백테스트 positions'가 이런 주에는 성립하지 않는다. PM에 알린다. 기록 마감(t+1 거래일 09:00, DEC-04) 전에 만든 경보까지 쓰면 이 차이가 없어지고, 같은 시각 기준이라 다시 계산해도 같다.
    - PM이 `engine_alerts`·`AlertBook` 안에서 처리하므로 B(`_replayed_holdings`)·C 호출부는 그대로다. #51 위에서 B 테스트를 돌리자 B-07 테스트 하나만 실패했다. 백테스트 경보 파일을 직접 쓰는데 `generated_at`이 없었기 때문이다. 그래서 io 스키마에 이 열이 있으면 넣게 고쳤다 (병합 전후 둘 다 통과).
    - 한계: alerts 파일은 자동 커밋되지 않아 `generated_at`의 사후 검증력은 커밋 기록보다 약하다. PM이 최종 보고서에 적는다.
  - 실행 순서 확인 (D-06, 11/10). 시각 사이에 모순은 없다:
    - 일간 경보는 달력일 d마다 d 다음 날 08:00쯤 돈다. OpenDART는 접수일만 줘서 d가 끝나야 0건이 확정된다.
    - forward 주간 실행은 KRX 자료가 나오는 t 다음 날 08:00 뒤부터 t+1 거래일 09:00 전 사이다 (DEC-04).
    - B가 읽는 경보는 `cooldown_at(t)`(t 15:30 전 접수)와 재생의 `engine_alerts(첫 기록, t)`(t 종가까지)뿐이다. 그래서 t일 파일까지만 있으면 된다.
    - t는 그 주 마지막 거래일이라 t+1 거래일은 3일 이상 뒤다. t일 파일은 t 다음 날 08:00에 생겨 마감 전에 있다. 금요일 밤·주말 경보도 월요일 체결 전에 생긴다.
    - 다만 둘 다 수동 실행이다 (DEC-15). t 다음 날 아침에 주간을 먼저 돌리면 t일 파일이 아직 없다. 11/20 뒤에는 그때 M9가 미수집으로 멈추므로 틀린 기록은 남지 않는다. D가 README와 주간 실행 안내에 '일간 경보(t일까지) → 주간 실행' 순서를 적었다 (#50, 11/10). DEC-33 (3)도 같다.
    - PM #51 확인 (11/10, 스크래치 worktree = #51 head 16b4ca0 + B 작업 + 아래 `_call_m9` 변경):
      - `cooldown_at(t)`·`engine_alerts(첫 기록, t)`는 '첫 체결일 − 쿨다운 거래일'의 날(아직 유효할 수 있는 쿨다운의 가장 오래된 근거일)부터 t까지의 달력일 파일만 읽는다. t 다음 날 이후 파일은 읽지 않는다.
      - 실제 PM 함수로 확인했다. 그 구간에 0행 파일만 두고 주간을 돌리면 M9 없이 돈 결과와 같은 positions가 기록되고 deviation이 없다. t일 파일을 빼거나, 시작을 하루 늦추거나, M9가 t 다음 날까지 읽게 바꾸면 셋 다 `AlertsNotCollected`로 멈춘다.
      - `_call_m9` 변경(`AlertsNotStarted`만 잡기): 연결 테스트를 바꾸고 더했다 (빈 alerts/에서 실제 PM 함수 → M9 없이 기록, 필요한 날짜 파일 없음 → 멈춤, 경보 파일 없이 난 다른 `FileNotFoundError` → 멈춤). 잘못된 변형 4개(옛 코드, `AlertsNotCollected` 잡기, `FileNotFoundError` 전부 잡기, 안 잡기)를 모두 잡았다. 전체 빠른 테스트 812 통과, 1 건너뜀.
      - 이 변경은 `AlertsNotStarted`를 import해서 지금 main에서는 돌지 않는다. #51이 병합되면 B 저장소에 넣는다.
  - PR #34 경계 확인 (11/9, 스크래치. 신호일 11/20(금), 체결일 11/23):
    - 15:29:59 경보는 `cooldown_at(11/20)`에 들어간다.
    - 15:30·저녁·토요일 경보는 신호 쪽 쿨다운에는 들어가지 않는다. 엔진이 11/23에 매수를 거부하고 청산한다 (5.2절 '체결' 행).
    - 20거래일 쿨다운은 두 형태가 같은 날 끝난다.
    - 합성 경보 파일 하나로 `weekly`가 두 함수를 실제로 불러 값을 받는 것도 확인했다.

  - 앞 단계가 B 입력 표(signals, universe, sector, index)를 스텁으로 썼으면 w_raw는 비우고 positions는 0행 스텁으로 쓴다. 지난 보유를 재생해야 하는 주에는 엔진 입력(prices, status, index, delisted, delisted_reused)도 같이 본다 (A #36 리뷰, 11/10). 입력이 없는데 스텁 실행도 아니면 멈춘다.
  - 지난 신호일의 기록이 하나도 없으면(첫 기록이 커밋되지 않은 경우) 현금 100%에서 시작한다고 경고한다. 첫 기록일에는 지난 신호일이 없어 이 경로에 오지 않는다.
  - 실제 데이터 리허설 (스크래치, 공식 결과 아님. A 11/8 데이터, A PR #1 베이스라인 신호, C PR #22 엔진):
    - 10/30을 첫 기록으로 가정했다. 10/30은 10종목, 현금 0.70이었다.
    - 11/6에는 10/30 기록을 `run_to_signal_close`로 재생해 보유 10종목을 만들었다. 버퍼 규칙이 9종목을 유지하고 3종목을 새로 사고 1종목을 팔아 12종목, 현금 0.71이 됐다. 표 읽기를 포함해 16.4초 걸렸다.
    - 두 날 모두 E_t는 하한 0.30이다 (코스피 200 변동성이 높다). 업종 한도(전자부품 0.364 → 0.30)가 실제로 걸렸고 불변식을 통과했다.
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

실제 데이터 실측 (11/10 스모크, 스크래치. B-06 `rl_fold_report`, 폴드 2016, A 11/8 데이터·베이스라인 신호, C 엔진, Windows 단일 프로세스, 스텝 수만 1/5로 줄임):

| 항목 | 측정값 |
|---|---|
| 준비 (표 읽기, 폴드 분할, 신호·피처, 스냅샷 253개) | 33초 |
| 학습 100,000스텝 (λ 4 × 20,000 + 시드 0 × 20,000, 10,000스텝마다 검증 평가 포함) | 121초, 약 1.2 ms/스텝 |
| 체크포인트 `best.zip` | 개당 762 KB |

폴드 하나의 공식 리허설(λ 탐색 400,000 + 시드 하나 300,000스텝)은 약 14분, 전체 19.0M 스텝은 단일 프로세스로 약 6.4시간이다 (DEC-06의 6.1시간 추정과 비슷하다). 체크포인트는 B-07 전체(10폴드 × λ 4 + 시드 5 = 90개)면 약 69 MB다. 그래서 git에 넣지 않고 팀 드라이브의 run_id 폴더에 둔다. 드라이브 파일이 정본이고, metrics.json에 파일마다 SHA-256을 남긴다 (PM 11/10). 같은 λ·시드·스텝이면 학습 결과가 같다: 스모크에서 λ = 1 탐색과 시드 0 본 학습의 최고 검증 IR이 0.8410511508509657로 같았다. 공식 리허설에서는 λ = 0 탐색과 시드 0 본 학습의 best.zip(둘 다 40,000스텝)이 정책 파라미터 지문까지 같았다. 리허설을 다시 돌렸을 때도 체크포인트 5개의 지문이 모두 같았다. 파일 SHA-256은 zip에 저장 시각과 시스템 정보가 들어가 달랐다. 그래서 다시 학습한 가중치가 같은지는 파일 해시가 아니라 파라미터 지문(`policy_sha256`)으로 비교한다.

---

## 7. 테스트

실행: `python -m pytest tests/portfolio -p no:cacheprovider` (README 절차로 만든 `.venv`에서)

PPO를 실제로 학습하는 테스트 2개는 `slow`로 표시했다. `-m "not slow"`를 붙이면 뺀다. B의 Windows 환경에서 전체는 약 15초, `slow` 2개가 그중 약 10초다.

| 파일 | 확인 내용 |
|---|---|
| `test_buffer_rule.py` | **B-03 완료 기준**: 단계 2 버퍼(rank 21~40 유지, 41부터 매도, 미보유 rank 21~40은 사지 않음), crash_flag·쿨다운·U_t 밖·signals에 없는 보유 종목 매도, 단계 3 후보 부족·전액 현금, 유지 종목이 보유 상한에 포함됨, 단계 5 M8(업종 한도·E_t)과 G7·G4 모드, H를 엔진의 실제 보유에서 읽음, 합성 시장 전 구간을 엔진 위에서 실행(불변식·보유 20개 이하), 같은 실행을 C의 M5 엔진에서 해 목표 비중이 같음(엔진이 없으면 건너뜀), 버퍼가 회전율을 낮춤 |
| `test_params.py` | 부록 1 값과 키 이름 (R-14), FILL_ 값을 쓰면 멈춤 (R-08), 구현하지 않은 설정값 거부, `portfolio.rl.env` 8개(DEC-21)가 구현과 다르거나 키가 빠지거나 더 있으면 거부, `action_bound`·`out_of_universe_fill`이 `rl_env` 상수와 같음, λ 격자가 오름차순이 아니면 거부, `configs/prereg.yaml` 커밋 뒤 테스트 입력과 같은지 (labels 절 포함) |
| `test_rl_fold_report.py` | **B-06**: 폴드 분할이 A의 `make_folds`와 같고(2011~2025 달력, 폴드 2016), h = 20 purge로 빠진 신호일이 맞음, 검증 2년·h = 20이 아니거나 폴드가 아닌 해면 거부. 학습 환경은 폴드 학습 신호일·`random`, 검증 환경은 검증 신호일·`full`이고, 학습 구간 안 purge 자리를 건너뛰는 에피소드가 없음. 스냅샷이 없거나 52스텝 연속 구간이 없으면 멈춤. λ 동률이면 작은 λ. λ 선택 기록이 완전하면 완료 기준 통과, 고른 λ가 규칙과 다르거나 평가 기록이 빠지면 실패. 체크포인트 지문: 가중치가 같고 파일이 다르면 `sha256`만 다르고 `policy_sha256`은 같음, 가중치가 다르면 `policy_sha256`도 다름. 실제 PPO로 λ 2개 탐색과 시드 하나 학습, 본 학습과 고른 λ 탐색의 가중치 지문이 같음(`slow`). run 폴더 구성과 체크포인트 5개 지문, 덮어쓰기 거부, 사전등록에 없는 시드 거부. **B-07 입력**: A 폴드별 ML 신호 파일을 io로 읽고 SHA-256·구간별 model_version을 기록, 실제 라벨 종료일을 fold_split에 넘겨 지연 purge 수를 기록, PM 백테스트 경보로 스냅샷 쿨다운(다음 신호일까지, 쿨다운 기간 뒤 없음)과 엔진의 체결일 쿨다운·첫날 이벤트 청산. A 파일 신호일이 B 폴드와 다르면 결과 폴더 전에 멈춤 |
| `test_rl_wf_report.py` | **B-07 시험 운용**: 시드마다 시험 신호일 전부의 positions(constructor rl, 비중 합 1), 신호일 t의 정책이 t가 속한 해의 폴드 정책, 마지막 신호일 체결 없음, fold_seed_ir 폴드 × 시드 행, 12월 신호일의 1월 체결 주는 그 해 폴드. 완료 기준: M9 없이 학습한 폴드면 '설계 그대로' 실패, 커밋이 섞이면 기록. 멈추는 경우: 결과 폴더·positions 파일이 이미 있음(R-11), 다른 가중치(policy_sha256), 체크포인트·폴드 폴더 없음, `{Y}` 없는 패턴, task·완료 기준·시드·학습 신호·라벨 종료일·시험 연도·설정이 다른 폴드, 시험 신호일 누락 |
| `test_engine_adapter.py` | B-05 어댑터: 표 → 종가·정지 행렬(상장 전 결측, KOSPI200 행만), status 행이나 지수 거래일이 빠지면 멈춤, 폐지일을 전 거래일로 옮기고 원종가를 수정종가 단위로 바꿔 청산, 가격 없는 폐지는 0, 잘못된 delist_date 거부, DEC-28 옛 회사 청산과 새 회사 매수, 가격 공백에 폐지 기록이 없거나 공백이 둘이면 멈춤, `load_engine_inputs`가 `delisted_reused`를 io로 읽고 파일이 없으면 FileNotFoundError, 엔진이 모르는 재상장은 멈춤, 정지일 평가(DEC-25), 비용 배율, 엔진이 구현하지 않은 부록 1 값 거부 |
| `test_env_check.py` | **B-05 완료 기준 (1)·(2)** (합성 신호): 9.3절 형식 합성 표(정지 칸, 보유 중 폐지 1건)를 어댑터로 C 엔진에 넣는다. RL 환경이 행동 0·무작위 행동으로 끝까지 돌고, 버퍼 비중을 넣은 RL 환경과 버퍼 규칙의 체결·비용·NAV 차이가 0이며 정지 거부와 폐지 청산이 실제로 일어난다. 비용 배율을 다르게 하면 차이가 잡힌다 |
| `test_env_check_report.py` | B-05 리포트: 신호일이 달력과 같음(끝날이 휴장일인 주 포함), universe 날짜가 다르면 멈춤, 합성 시장에서 두 기준 통과, 차이가 있으면 기준 (2) 실패, vol_20·시장 피처가 6.2절 정의와 같고 종목을 나눠 계산해도 같음(A PR #35 병합 전에는 건너뜀), 입력 파일 SHA-256, run 폴더를 한 번만 씀 |
| `test_market_state_report.py` | **B-04 완료 기준 (3)**: 그림 기간의 매 거래일이 결측 없이 계산되고 M8 표와 같다. 기간 뒤 종가를 바꿔도 결과와 데이터 지문이 같다. 이력 부족·해가 끝나지 않음·FILL_이면 멈춘다. metrics와 일별 표가 일치하고, SVG가 올바른 XML이며 같은 입력이면 같다. 리포트 폴더 구성과 덮어쓰기 거부 (R-11) |
| `test_risk_layer.py` | **B-04 완료 기준 (1)·(2)**. (2) 경계 사례는 손으로 계산할 수 있는 결정적 코스피 200 경로로 확인한다: 한 업종 20종목(업종 있음·'미분류', 한 종목 쏠림), 보유 5종목(한도 0.10, 업종 합이 정확히 0.30), regime_mc = 1(상한 0.70이 걸릴 때와 안 걸릴 때), sigma_t = 3배(E_t = 1/3)와 4배(하한 0.30), 네 경계가 겹칠 때. 그 밖에 Step 4~5를 6.9절 식 그대로의 느린 구현과 비교, regime_mc = 1일 때 E_t 상한, t 이후 코스피 200 미사용 (R-03), Step 1~3·6 손계산, '미분류' 한 그룹, mode off, 무작위 입력 200건의 불변식, 위반 시 예외, positions를 io.py로 쓰고 읽기 (io.py가 있을 때) |
| `test_rl_env.py` | SB3·gymnasium 환경 검사, 관측 배치, 보유·U_t 밖 슬롯, 마스킹, 상위 20개와 min_weight, 전액 현금, 보상 식, 52스텝과 구간 경계, purge로 빠진 신호일을 건너뛰지 않음(DEC-21 (g)), 슬롯 초과, 9.3절 표 → 스냅샷 |
| `test_rl_train.py` | **B-01 완료 기준** (`slow`): 합성 입력으로 λ 탐색 → 시드별 학습 → 체크포인트 저장·로드 → 결정적 운용이 끝까지 돈다. 저장한 체크포인트로 검증 IR 재현, 같은 시드면 같은 학습 결과 (`slow`). 연도 경계 모델 교체 |
| `test_weekly.py` | forward 4·5단계: 합성 시장의 7주 동안 매주 positions가 같은 신호일 `run_buffer_rule` positions와 같다. 지난 보유는 그 전 기록을 엔진으로 재생해 만들고, 기록된 목표와 실제 보유가 다른 주가 들어 있다(정지 10%). 재생 엔진은 C의 `run_to_signal_close`가 있으면 그것을, 없으면 같은 보유를 주는 `run_period` 대역을 쓴다. positions_from 전에는 쓰지 않음, index에 신호일 종가가 없으면 멈춤, 첫 기록은 지난 기록 없이, 다음 기록부터 D 함수가 없으면 실패, M9가 없으면(연결 전·모듈 없음·함수 없음·날짜별 경보 파일 0개) M9가 경보를 주지 않을 때와 같은 positions를 스텁 아닌 실행으로 쓰고 이유를 적은 deviation 2건을 남김(`RECORD_WITHOUT_M9 = False`면 스텁 표시), M9의 다른 오류·경보 파일이 있는데 난 FileNotFoundError·M9 모듈 안의 import 실패는 멈춤, `_resolve`가 모듈·함수 없음만 `EntryNotFound`로 바꿈, 스텁 입력이면 스텁 출력, 재생할 주에 엔진 입력만 스텁이어도 스텁 출력(엔진·M9를 부르지 않음), 재생이 상태를 잇고 체결 안 된 목표를 H에서 뺌, 기록이 빠진 주는 직전 보유를 유지하고 다음 주 기록이 이어짐, 기록이 하나도 없으면 엔진 없이 현금 100%에서 시작('직전 보유 유지' 경고 없이 '현금 100%에서 시작' 경고), 신호일이 아닌 기록이면 멈춤, 찾지 못한 entry의 오류 문구, 담당자와 정한 함수 이름 3개와 인자(PR #34·#31 병합 전에는 건너뜀), 휴장 주의 신호일 |
| `test_vol_target.py` | **B-02 완료 기준**: 2016년 이후 종가를 바꿔도 vol_target_annual이 같다. 6.9절 식 그대로의 느린 구현과 비교, 2010년 종가는 첫 sigma_t 창의 21개만 사용, 기간이 모자라면 예외. compute_regime_mc가 M8 표와 같고 vol_target_annual 없이 계산됨, t 이후 종가 미사용, 2011년 초 bear_t 이력이 없을 때 rebound_t = 0이면 0, 1이면 결측(D4), io.py로 index를 읽어 계산 (io.py가 없으면 오류 안내). 리포트의 데이터 지문이 2016년 이후 종가와 무관, 리포트 폴더 구성, 같은 run_id 덮어쓰기 거부 (R-11). `_git_commit`이 바뀐 추적 파일과 git에 없는 새 파일(무시 대상 제외, 공백·한글 경로 그대로)을 적음 |

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

**결정 (DEC-20, 2026-11-07 PM)**: A안(추천안) 그대로 확정. 코드 변경 없음.

### D3. RL 환경 세부값 (부록 1에 없음)

6.5절을 코드로 옮기면서 아래 값이 필요해 잠정값으로 두었다. G5 판정은 사전등록 시점의 설계로 하므로 (R-15), 11/9 전에 확정해야 한다.

**결정 (DEC-21, 2026-11-07 PM)**: A안. 아래 8개 값을 그대로 확정하고 PM이 부록 1 `portfolio.rl`에 넣는다. 들어오면 B는 `RLParams.from_prereg`에서 YAML 값과 코드 상수가 같은지 검사한다 (다르면 멈춤). (g)의 purge 처리 문구는 PM 요청으로 11/7에 추가했다.

| 항목 | 값 | 이유 |
|---|---|---|
| (a) 행동(로짓) 범위 | [−10, 10], 로짓 = 행동 | SB3 환경 검사가 유한한 범위를 요구한다. 로짓 차가 20이면 비중 비가 약 5억 배라 현금 100%를 포함한 모든 배분을 표현할 수 있다. [−1, 1]이면 상위 20개 절단을 거쳐도 현금 100%를 만들 수 없다 (후보 40개일 때 현금 최대 약 0.58) |
| (b) U_t 밖 보유 슬롯의 rank_score 백분위, crash_prob, vol_20 | 0 | 신호·피처 행이 없다. 이 슬롯은 항상 마스킹되어 매도 대상이다 |
| (c) 후보가 60개를 넘을 때 | 보유하지 않은 후보를 rank가 나쁜 순서로 뺀다 | 상·하한가나 거래정지로 보유가 20을 넘으면 C_t가 60을 넘을 수 있다. 보유 종목은 슬롯을 받아야 비중을 정할 수 있다 |
| (d) 상위 20개 절단의 동률 | 앞 슬롯(rank가 좋은 종목) 우선 | 결과를 결정적으로 만든다 |
| (e) 52스텝 종료 | truncated(시간 제한)로 표시 | 에피소드 끝은 MDP의 종착 상태가 아니라 표본 추출로 자른 것이다. SB3가 마지막 상태의 가치를 부트스트랩한다 |
| (f) λ 선택 점수 | 100,000스텝 동안 10,000스텝마다 평가한 검증 IR의 최댓값. 동률이면 격자에서 앞선(작은) λ | 6.5절 "λ 선택의 '검증 IR'도 같은 정의다"를 체크포인트 선택과 같은 방식으로 읽었다 |
| (g) 학습·검증 신호일 | 4.7절 purge(h = 20)를 적용한 신호일. 한 스텝은 신호일 달력에서 바로 이어지는 두 신호일 사이이고, purge로 구간 중간 신호일이 빠지면 거기서 구간을 끊는다 (빠진 날을 건너뛰는 스텝은 없다). random 에피소드는 끊기지 않은 구간 안에서만 시작점을 뽑고, full(검증)은 끊긴 자리에서 체결하지 않고 다음 신호일에 현금 100%로 다시 시작한다 | R-04. RL 보상 구간(약 6거래일)은 h = 20 purge 안에 들어간다. 실제 종료일(`actual_end_dates`)로 purge하면 경계에서 먼 신호일도 빠질 수 있다. 빠진 날을 건너뛰면 그 스텝만 몇 주짜리 보상이 된다 |
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

A-02 데이터(11/3)로 확인했다. 코스피 200이 2010-01-04부터라 bear_t의 창은 2011-03-31에 처음 찬다. universe 신호일 가운데 2011-01-07 ~ 2011-03-25의 12개가 여기에 걸린다.

**B 결정 (2026-11-03, 조윤상)**: A가 Open API에 2009년 코스피 200 원천이 없음을 확인해 C는 할 수 없다. 대신 아래 규칙을 쓴다. 6.9절 식과 부록 1 값은 바뀌지 않고, 학습 신호일도 빠지지 않는다. A 동의와 PM의 decisions.md 기록을 기다린다.

```text
regime_mc_t = bear_t × rebound_t 그대로 계산한다.
rebound_t = 0이면 bear_t의 창이 차지 않아도 regime_mc_t = 0이다 (곱이 bear_t와 상관없이 0).
rebound_t = 1인데 bear_t를 계산할 수 없으면 결측이다 (InsufficientHistory).
```

- **근거**: rebound_t는 종가 21개만 있으면 원래 식대로 계산된다. 2011-01-03 ~ 03-30의 59거래일 모두 계산되고, 최댓값이 0.0866이라 전부 0이다. 그래서 이 기간 regime_mc는 추정 없이 0으로 정해진다
- **확인**: 2011년 이후 3,890거래일에 결측이 없다. bear_t가 계산되는 3,831일은 수정 전 값(bear × rebound)과 같다
- **옵션 B와 비교**: B(초기 신호일 12개를 학습에서 제외)는 값이 이미 정해진 신호일을 버린다
- **참고**: bear_t가 1이 되려면 2009년 고점이 300.14 이상이어야 한다 (2010-10 ~ 2011-03 최저 종가 240.11 ÷ 0.8). 2010-01-04 종가는 223.49였다

### D5. 4.7절 폴드 분할·purge 공용 함수의 담당

정리됨 (11/10 확인): A가 `src/models/splits.py`(`make_folds`, `purge`, `cross_fits`, PR #35)로 만들었다. B-06은 `make_folds`를 그대로 불러 A-06·A-07과 같은 학습·검증 신호일을 쓴다 (3.4절). 아래는 처음 적은 내용이다.

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
| B-02 | 11/8 | 완료: 0.157263을 부록 1에 기입, PR #11 병합. Q-12 결과 칸은 PM이 기입했고 D4는 DEC-19로 기록됐다 |
| B-03 | 11/13 | 단계 1~5와 엔진 위 실행을 구현했다 (2절). C의 M5(PR #8 헤드)를 합친 상태에서 합성 시장 실행을 확인했다: 목표 비중이 FakeEngine과 같고, 초과수익은 목표 합이 1인 체결(현금 부족 배율)에서만 최대 7e-5 다르다. PR #8 병합(b2b32a6) 뒤 main에서 `test_runs_on_c_execution_engine`이 건너뛰지 않고 통과한다 (전체 336 통과). 남은 것: A 코드 리뷰 승인. 실제 신호(A-04) 실행은 완료 기준 밖이라 B-05 완료 기준 (베이스라인 신호로 에피소드 끝까지)에서 확인한다 |
| B-04 | 11/16 | 완료: 완료 기준 (1)·(2) 테스트 통과, (3) 그림 `reports/20261104_B_m8_states/` (4절, prereg.yaml 초안 기준), A 승인 (PR #13). 11/9 공식 사전등록에서 쓰는 키가 바뀌면 새 run_id로 다시 만든다 |
| B-05 | 11/16 | 엔진 어댑터와 완료 기준 확인 함수 구현 (5.1절). 합성 시장에서 (1) 끝까지 실행, (2) 버퍼 규칙과 차이 0. 실제 A 베이스라인 신호 미리보기도 781개 신호일 끝까지, 차이 0. A 승인 (PR #21, 11/6). `delisted_reused`를 io로 읽고 가격 공백 검사를 더했다 (11/8). 리포트 `env_check_report.py`로 실제 A 베이스라인 신호의 공식 결과 `reports/20261110_B_env_check/`를 남겼다 (11/10, 코드 커밋 5b2f1b3): 781개 신호일 끝까지, 버퍼 규칙과 차이 0, 입력 파일 SHA-256 포함. 리포트 코드와 결과 A 승인 (PR #36 재리뷰 11/10, head 5c72fae, 병합 c014969) |
| B-06 | 11/23 | 폴드 실행 `rl_fold_report.py`(3.4절)와 `portfolio.rl.env` 8개 검사를 만들었다. 폴드 2016 베이스라인 신호 리허설 `reports/20261110_B_rl_y2016_v2/`에서 λ 선택 절차가 끝까지 돌았다 (11/10, 코드 커밋 54dde55, λ = 0, 시드 0 최고 검증 IR 0.856). 체크포인트는 팀 드라이브에 두고 metrics.json에 지문을 남긴다 (PM 11/10). 다시 실행해도 체크포인트 5개의 가중치 지문이 같았다. A 승인 (PR #47, 11/11, 병합 8828d6e). 리뷰 지적은 B-06 `main()`이 명목 purge라 A 신호와 신호일이 다르다는 것이었다. B-06 리허설은 베이스라인 신호라 영향이 없고, 실제 종료일 purge는 B-07의 `--label-end-dates`로 넣었다 (3.5절) |
| B-07 | 11/30 | 폴드 학습에 A 폴드별 ML 신호·C 실제 라벨 종료일·M9 백테스트 경보 입력을 더했고, 시험 연도 운용 `rl_wf_report.py`를 만들었다 (3.5절). A·C 인계본(11/10)은 B 폴드·시험 신호일과 10개 폴드 모두 맞는다. 실데이터 스모크(스텝 축소)에서 10폴드 학습 → 시험 운용이 끝까지 돌았다. 다만 C 엔진이 '신호일이 마지막 가격일인 U_t 종목'(2023-08-11 048260 한 건)을 목표로 받으면 멈춘다. 판정 실행은 뉴스 반영판 신호를 받은 뒤 시작한다 (PM 11/10, 뉴스 11/16 목표). 남은 것: 뉴스 반영판 rlsig·wfpm(A), C 엔진 수정 병합(PR #52, B 확인 11/11), M9 백테스트 경보(PM) 또는 M9 없이 돌리는 결정, 판정 실행(10폴드 × 5시드) |
