# 급락 필터 탐색 후보 · 짝 32개

2016~2025년 백테스트에서 방어 후보로 남긴 버전이다. 실적 공시 유형, 공시 후 2~5일 주가·거래대금 반응과 기존 가격·시장 피처를 사용한다. 같은 주의 급락·비급락 종목을 최대 1:32로 짝지으며, 고정 시드 20261119와 시험 직전 검증 연도 1년의 확률 보정을 적용한다. 보정 점수 90%와 별도 급등·거래대금 대비 가격 변동 점수 10%를 섞고, 매주 상위 10%를 차단한다.

| 2016~2025년 탐색 결과 | 값 |
|---|---:|
| 전체 AUC | 0.723046 |
| G1 목표 및 차이 | 0.727460 · 0.004413 남음 |
| 같은 주 AUC | 0.726179 |
| Lift@10% | 2.727021 |
| 최대낙폭 | 46.49% |
| 무필터 대비 상대 낙폭 감소 | 32.18% |
| CAGR | −0.62% |
| 차단 효과 HAC t | −3.834 |
| G2 수치 기준 | 10년 전체 충족 · 5년 1/6 충족 |

G2 결과는 차단 종목의 몫을 현금에 두는 개인 프록시에서 얻었다. 5년 구간에서는 2016~2020년만 세 조건을 충족했다. 기록의 범위는 walk-forward 탐색이다. 2026년 홀드아웃과 운영 신호에는 별도 검증이 필요하다.

## 재현

`git_commit.txt`는 원래 실험에서 사용한 저장소 기준 커밋이다. `config.yaml`은 선택한 후보의 입력 열 순서와 고정 설정을 담는다. `metrics.json`에 원래 실험의 전체·연도별 분류 지표, 전체·기간별 G2 결과, 입력 파일 SHA-256을 기록했다.

점수 재학습에는 아래 다섯 가지 원래 입력 파일이 필요하다. 저장소에는 실행 코드와 요약 결과를 둔다. 입력 스냅샷은 팀 데이터 보관 경로로 별도 전달한다.

| 파일 | 입력 계약 |
|---|---|
| `features.parquet` | 신호일 시점의 전체 유니버스 `(date, ticker)`와 `config.yaml`의 19개 열, 학습·검증·시험 기간 포함 |
| `labels_h20.parquet` | C의 20거래일 라벨 `(date, ticker, h, y_crash)` |
| `label_ends_h20.parquet` | C의 명목·실제 라벨 종료일 `(date, ticker, h, nominal_end, actual_end)` |
| `trading_calendar.parquet` | 당시 거래일 캘린더 `date` |
| `pressure_scores.parquet` | 같은 10개 시험 연도의 표본 외 보조 확률 `(date, ticker, pressure_prob)` |

저장소 루트에서 팀 Python 환경을 활성화하고 실행한다. 입력을 다른 폴더에 보관한다면 경로를 바꾼다.

```powershell
python reports/20261119_PM_gate_pair32/reproduce.py --features inputs/features.parquet --labels inputs/labels_h20.parquet --ends inputs/label_ends_h20.parquet --calendar inputs/trading_calendar.parquet --pressure inputs/pressure_scores.parquet --output pair32_scores.parquet
```

출력은 520주 228,422개 신호의 `date`, `ticker`, `crash_prob`, `crash_flag`다. 실행기는 원래 입력 지문을 확인하고 실제 라벨 종료일을 기준으로 학습·검증 경계를 정리한다. `metrics.json`의 포트폴리오 결과는 M5 개인 프록시를 사용한 별도 재생 기록이다.
