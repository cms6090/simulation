# MC vs Split Conformal Prediction 시뮬레이션

## 1. 작업 목적과 원칙

- 이 파일은 연구 시뮬레이션을 구현하는 프로젝트 지침이며, 기존 코드나 실행 결과가 존재한다는 뜻은 아님
- 사용자의 목표: **MC–CP 차이를 coverage·중심·폭으로 평가 → 각 지표의 차이를 Structural·Fitting·Calibration으로 분해 → marginal·fixed-x·fixed-calibration에서 확인**
- Coverage가 우선이며 중심·폭과 원인 분해도 필수 핵심 분석으로 구현
- Heatmap은 결과 표현 단계로 후순위 배치하고, plotting 없이 수치 평가·분해·저장이 가능하도록 구성
- 구현 전에 기존 저장소·설정·설계 문서를 확인하고 재사용 가능한 코드를 파악
- 이 파일만으로 기본 구현 가능하도록 작성했으며, 존재하지 않는 외부 파일을 필수 의존성으로 가정하지 않음
- 함께 제공된 `시뮬.md`가 있으면 설명 자료로 참고하고, 이 파일의 fixed-x 중심 불변·500회 기본값 설명을 적용
- 이후 사용자가 명시한 수정사항은 반영하되, 연구 목적·DGP·score·평가 대상의 의미를 설명 없이 변경하지 않음
- 별도 언어 지정이나 기존 구현이 없으면 Python, NumPy, SciPy, scikit-learn으로 시작하는 구현 기본안 사용
- 설명·README는 한국어, 변수·함수명은 일관된 영어 사용, `fitted model` 용어 사용
- 관측한 사실과 예상·미정 설정을 구분하고, 실행하지 않은 코드·검증·결과를 실행했다고 보고하지 않음

## 2. 연구 범위와 DGP

- 설명변수 `X.shape == (n, 2)`, 반응 `y.shape == (n,)`, 각 행은 독립 관측치
- $X_1,X_2\overset{iid}{\sim}U(-1,1)$, 입력과 오차 독립, $Y=m_0(X)+\sigma_0(X)\varepsilon$
- `mean=linear`: $m_0(x)=\sqrt2(x_1+x_2)$
- `mean=nonlinear`: $m_0(x)=\sqrt{6/11}\{2\sin(\pi x_1)+x_2+x_1x_2\}$
- 두 mean의 평균은 0, 신호 분산은 $4/3$이며 이번 프로젝트의 구현 기준 함수로 사용
- `scale=homo`: $\sigma_0(x)=1$
- `scale=hetero`: $\sigma_0(x)=(0.4+1.2\sin^2(\pi x_1))/\sqrt{1.18}$
- Hetero의 $E[\sigma_0^2(X)]=1.18$, homo는 1 — 이를 몰래 재정규화하지 않으며 비교 해석에 기록
- `error=gaussian`: $\varepsilon\sim N(0,1)$
- `error=student_t`: $\varepsilon=T/\sqrt3$, $T\sim t_3$
- `error=lognormal`: $\varepsilon=(e^Z-e^{1/2})/\sqrt{e(e-1)}$, $Z\sim N(0,1)$
- 모든 오차는 평균 0·분산 1, error별 `sample`, `cdf`, `ppf`가 동일한 표준화를 사용하도록 구현
- Student-t는 `cdf(e) = t.cdf(sqrt(3)*e, df=3)`, `ppf(p) = t.ppf(p,df=3)/sqrt(3)`
- LogNormal은 $a=e^{1/2}$, $s=\sqrt{e(e-1)}$일 때 $F_\varepsilon(u)=0$ if $su+a\le0$, 아니면 $\Phi(\log(su+a))$; $q_p=(\exp(\Phi^{-1}(p))-a)/s$
- OLS는 절편과 원변수 두 개만 사용, nonlinear·interaction feature를 추가하지 않음
- RF는 squared-error conditional mean 회귀 forest 사용, 1,000 trees 유지
- 2 mean × 2 scale × 3 error = 12개 DGP; OLS·RF 포함 24개 model–DGP 조합
- 목표 coverage `[0.90, 0.95, 0.99]`, alpha `[0.10, 0.05, 0.01]`; 같은 fitted model·residual에서 동시 계산
- 주 분석 `n_train=1000`, `n_cal=1000`, `B=200`; 차원 확장·CQR·normalized CP·Jackknife+는 현재 범위 밖

## 3. 표기와 설정 상태

- `model_id` / 수학의 $j$: OLS 또는 RF이며 평가 관점이 아님
- `regime`: `marginal`, `fixed_x`, `fixed_calibration`; `b`: training ID, `r`: calibration ID, `i`: calibration 관측치
- 출력의 `b,r`는 1부터 시작하는 논리 ID 사용; 배열 인덱스와 명확히 구분
- Fixed-x는 각 DGP의 `b=1..10`에서 `R_cal=500`을 실행 기본값으로 사용; 수학적으로 필수인 횟수는 아님
- RF의 leaf size·depth·feature sampling·bootstrap·세부 버전은 아직 미정: 기존 설정이 없으면 명시적인 pilot 기본안을 정해 기록하고, 같은 설정을 모든 DGP에 적용
- `M_MC`, `M_pop`, `N_eval`, 수치 허용오차, master seed는 config로 관리; 아직 연구 결과로 검증된 확정값은 없음
- 합리적인 pilot 설정으로 작은 실행을 진행하고, 선택 근거·안정성을 기록해 full config를 완성; 라이브러리 기본값을 숨기지 않음
- 설정 제안은 제안으로 표시하고, 사용자 지정값이 있으면 이를 우선 적용

## 4. 다섯 구간의 생성 계약

$C(x)=[L(x),U(x)]$이며 모든 구간이 같은 평가 입력에 대한 endpoint를 반환하도록 구현

| interval_id | 생성 과정과 정의 |
| --- | --- |
| `oracle` | True error 분위수 계산 → $[m_0(x)+\sigma_0(x)q_{\alpha/2},\ m_0(x)+\sigma_0(x)q_{1-\alpha/2}]$ |
| `mc` | True DGP로 $Y\mid X=x$ 표본 생성 → 양쪽 empirical quantile 사용 → Oracle의 MC 근사 |
| `res_true` | True mean residual $R_0=\lvert Y-m_0(X)\rvert$ → 전체 $X$에 대한 $q_{0,\alpha}=q_{1-\alpha}(R_0)$ → $[m_0(x)-q_{0,\alpha},m_0(x)+q_{0,\alpha}]$ |
| `pop` | Training으로 $\widehat m_j$ 학습·고정 → 독립 새 관측치의 $R_j=\lvert Y-\widehat m_j(X)\rvert$ 분포 → $q_{pop,j,\alpha}=q_{1-\alpha}(R_j\mid\widehat m_j)$ → $[\widehat m_j(x)-q_{pop,j,\alpha},\widehat m_j(x)+q_{pop,j,\alpha}]$ |
| `cp` | 같은 fitted model → 실제 독립 calibration $m$개의 absolute residual → 아래 conformal 순위 → $[\widehat m_j(x)-\widehat q_{j,\alpha},\widehat m_j(x)+\widehat q_{j,\alpha}]$ |

- `res_true`, `pop`, `cp`는 **absolute residual** 사용; signed residual의 양쪽 분위수로 바꾸지 않음
- `pop`은 training residual 또는 실제 calibration residual을 재사용하여 만드는 구간이 아님
- `cp`: $k=\lceil(m+1)(1-\alpha)\rceil$, $k\le m$이면 정렬 residual의 `k-1` 원소, $k=m+1$이면 $+\infty$
- 기본 `m=1000`에서 순위는 901·951·991; quantile 기본 보간이나 경험적 90·95·99% 분위수로 대체하지 않음
- 부동소수점 때문에 정수여야 할 순위가 한 단계 바뀌지 않도록 순위 계산을 검증
- MC empirical quantile의 보간 규칙은 명시적으로 고정하고 기록; CP의 순위 규칙과 분리
- Location-scale 구조에서는 동일 error sample의 empirical quantile을 $m_0(x)+\sigma_0(x)\widehat q_p$로 변환해 MC를 모든 위치에서 계산 가능
- MC는 fitted model을 사용하지 않음; error별 benchmark 표본을 pilot에서 검증 후 고정해 모델·반복 간 공유하고 그 의존성을 기록
- True quantile을 MC endpoint에 직접 넣지 않으며, true quantile은 Oracle·검증에만 사용
- Oracle·MC·res_true는 모델별로 중복 생성할 필요 없음; pop은 training·model별, cp는 training·calibration·model별로 계산

## 5. Population quantile 계산

- Calibration 무한대는 정의이며, 실제 구현은 수치 근사
- 고정 중심 함수 $g$에 대해 $p_g(q,x)=F_\varepsilon((g(x)+q-m_0(x))/\sigma_0(x))-F_\varepsilon((g(x)-q-m_0(x))/\sigma_0(x))$
- $H_g(q)=E_X[p_g(q,X)]$; $g=m_0$이면 res_true, $g=\widehat m_j$이면 pop; $H_g(q)=1-\alpha$의 해를 구함
- 독립 reference 입력을 생성 → true·fitted prediction과 scale 캐시 → 동일 입력에서 CDF 평균 → bracket을 확보한 root-finding으로 해 탐색
- Root-finding 중 reference 입력을 다시 뽑지 않음; root 잔차뿐 아니라 별도 reference 입력과 표본 확대에서 quantile 정확도 확인
- Reference 입력은 training·calibration·최종 평가 입력과 독립; 모델 간에는 공유하되 training 반복별 독립 reference를 기본으로 사용
- Reference $Y$ 생성은 불필요; error를 CDF로 평균해 근사 변동을 줄임
- Population 근사에 conformal 순위 보정을 적용하지 않음
- True residual의 입력 의존성은 $X_1$뿐이므로 $q_{0,\alpha}$는 1차원 적분 가능; scale·error·alpha별 캐시 재사용 가능
- Reference 근사값과 exact 정의를 메타데이터에서 구분; 수치 실패를 성공값·0으로 바꾸지 않음

## 6. 평가지표와 필수 3단계 분해

- 고정된 구간의 `conditional_coverage`: $F_\varepsilon((U-m_0)/\sigma_0)-F_\varepsilon((L-m_0)/\sigma_0)$; 현재 연속 오차에 대한 식
- `center=(L+U)/2`, `half_width=(U-L)/2`, `length=U-L=2*half_width`
- `target_coverage=1-alpha`; coverage와 목표의 차이, CP와 MC의 차이는 별도 열로 저장
- `nominal`은 목표 수준; marginal·conditional은 실제 포함 확률의 평균·조건부 대상 구분
- 각 scalar metric $T$에 대해 다음을 같은 위치·반복·alpha에서 계산

| component | 정의 |
| --- | --- |
| `total_mc` | $T(C_{CP,j})-T(C_{MC})$ |
| `structural` | $T(C_{res,0})-T(C_{Oracle})$ |
| `fitting` | $T(C_{pop,j})-T(C_{res,0})$ |
| `calibration` | $T(C_{CP,j})-T(C_{pop,j})$ |
| `mc_approximation` | $T(C_{Oracle})-T(C_{MC})$ |

- `total_mc = structural + fitting + calibration + mc_approximation`; 오차 분해는 coverage·center·half_width·length 모두 필수
- Coverage는 각 구간의 CDF로 직접 계산; endpoint나 길이 차이를 coverage 차이로 선형 변환하지 않음
- $a_\alpha=(q_{\alpha/2}+q_{1-\alpha/2})/2$, $h_\alpha=(q_{1-\alpha/2}-q_{\alpha/2})/2$; Oracle 중심 $m_0+\sigma_0a_\alpha$, 반폭 $\sigma_0h_\alpha$
- 중심 분해는 `(-sigma*a, fitted_mean-true_mean, 0)`; 반폭 분해는 `(q0-sigma*h, qpop-q0, qhat-qpop)`; 길이 분해는 반폭의 2배
- 분해는 **signed difference**이며 절댓값·제곱의 합으로 바꾸지 않음; 항끼리 상쇄 가능, 독립적 인과 기여율이 아님
- Fitting에는 유한 training 오차와 misspecification이 함께 포함; calibration에는 표본 변동과 conformal 순위 보정이 함께 포함
- 평균은 같은 가중치로 계산하면 분해식 유지; 분산의 합에는 공분산이 필요하므로 단순 가산하지 않음
- 작거나 음수인 수치 결과를 이론에 맞추려고 강제로 보정하지 않음

## 7. 세 평가 관점과 반복 절차

세 관점은 동일한 구간 생성법의 평가 방식이며, **모든 calibration 입력은 원래 전체 DGP에서 생성**

1. 각 DGP와 `b=1..200`에서 training 1000개 생성 → OLS·RF 학습 → 독립 calibration 1000개(`r=1`) 생성
2. 같은 training·calibration을 모델 간 공유하고, 세 alpha에서 다섯 구간과 지표·분해 계산
3. 독립 평가 입력에서 marginal 요약, 고정된 평가 위치에서 위치별 결과를 같은 fitted model·calibration로 계산
4. `b=1..10`에서는 기존 `r=1` 재사용 후 `r=2..500`만 추가 → fitted model·population reference·MC benchmark 고정 → fixed-x 평가
5. Primary aggregation에는 각 b의 `r=1`을 한 번씩 사용; 첫 10개 b의 추가 calibration 때문에 가중치가 커지지 않게 분리

| regime | 고정 | 평가·요약 |
| --- | --- | --- |
| `marginal` | DGP·설정 | 각 b에서 독립 $N_{eval}$개 입력의 CDF coverage·구간 지표·분해를 평균 → B개 평균과 MCSE |
| `fixed_x` | Fitted model과 $x_0$ | 각 b별 calibration 500회의 지표·분해 평균·SD·분위수 |
| `fixed_calibration` | Fitted model·calibration | 각 x의 지표·분해 저장; 개별 고정 결과와 반복 평균 결과 구분 |

- Fixed-x의 대표 위치 기본안: 각 좌표 `[-0.9,-0.5,0,0.5,0.9]`의 25개 조합; 평가 위치만 고정, calibration의 모든 $X_i$를 $x_0$로 두지 않음
- Fixed-x에서 calibration 변화에 따라 coverage·폭은 변동 가능하지만 **CP 중심·전체 중심 차이는 고정**, 중심 calibration 항은 항상 0
- 각 b의 500회 결과를 별도 보고; 10×500회를 독립 training 5000회처럼 취급하지 않음; 10·500은 선택한 기본값
- Training·calibration 평균의 위치별 결과는 기본 B개의 `r=1`에서 별도로 계산하고 `conditioning=training_calibration_averaged`로 표시
- `conditioning=fitted_model_and_calibration_fixed`, `calibration_averaged_given_fitted_model` 등 평균 대상을 명시
- 고정 구간 CDF coverage에는 새 Y의 무작위성이 이미 적분됨; 기본 평가를 이진 포함 횟수 추정으로 대체하지 않음
- 같은 평가 입력·가중치를 모든 구간·지표·분해에 사용; 평가 입력은 b별 독립, 모델 간 공유
- MCSE는 b별 요약값의 표본 SD / sqrt(B); 방법 차이도 paired difference의 MCSE 계산
- MCSE와 반복 SD를 구분; 고정 MC benchmark·공통 수치 기준의 오차는 MCSE에 포함되지 않으므로 별도 검증·보고
- 정확한 population 기준에서 marginal coverage의 structural·fitting 평균은 0; 이는 local 차이가 없다는 뜻이 아니며 길이에는 적용되지 않음
- 현재 연속 score에서 CP의 calibration·test 평균 coverage는 $k/(m+1)$; 개별 b·r·x가 목표와 같아야 한다고 assert하지 않음

## 8. 구현 구조·설정·실행

- 기존 구조가 없을 때의 제안: `src/`에 `dgp`, `models`, `intervals`, `population`, `metrics`, `decomposition`, `runners`, `storage`; 별도 `configs/`, `tests/`, `results/`
- 핵심 수학 함수를 RNG·파일 I/O·plotting에서 분리; endpoint shape와 axis 순서를 함수 문서에 명시, float64 기본
- Config에 모든 실험 상수와 model/numerical/seed 설정 저장; 실행 시 최종 resolved config를 기록
- `smoke`, `pilot`, `full` 프로필과 실행 방법을 구현·README에 기록; 여기서 명령이 이미 존재한다고 가정하지 않음
- Smoke는 일부 DGP·작은 반복의 작동 확인용으로 main 결과와 구분; full 설정의 200회·1000 trees를 조용히 줄이지 않음
- Pilot에서 MC의 Oracle 대비 coverage·endpoint 오차, population 분위수 안정성, 평가 입력 평균의 안정성 확인
- Pilot 수치 표본 수는 예를 들어 $10^4,10^5,10^6$ 후보에서 조정 가능하나, 충분하다는 결론은 실제 오차 측정으로 제시
- 허용오차는 config에 명시; 관심 있는 통계적 차이보다 수치 오차가 작도록 정하고 미충족은 명시적 실패로 기록
- Main 구현·검증 → 세 regime와 저장 → pilot → 재현 가능한 full 실행 경로 순서; heatmap·추가 sensitivity로 핵심 구현을 미루지 않음

## 9. RNG·병렬화·수치 처리

- 명시적인 master seed와 안정적인 ID 기반 RNG 분리: scenario, b, r, train, calibration, model, reference, evaluation, MC 단계별 독립 stream
- Python의 매 실행 달라지는 `hash()`나 worker 완료 순서로 seed를 만들지 않음; `SeedSequence` 등에 고정 ID 매핑 사용
- Fixed-x r 반복에서 모델과 reference를 재학습·재생성하지 않음; parallel worker마다 같은 RNG state가 복제되지 않도록 구성
- CPU 기반 벡터화와 chunking 우선; 바깥 b 병렬화와 RF·BLAS 내부 병렬화의 중첩으로 oversubscription을 만들지 않음
- 병렬화 수준·worker 수를 config와 manifest에 기록; GPU 전환을 기본 범위에 넣지 않음
- 예측은 동일 입력에서 캐시하고 모든 alpha에서 재사용; calibration residual 정렬도 한 번만 수행
- CDF 차이의 꼬리 소거오차를 점검하고 필요 시 survival function 사용; 범위를 크게 벗어나는 coverage를 clip하여 오류를 숨기지 않음
- $k=m+1$의 무한 구간은 별도 처리: coverage 1·길이 무한, endpoint 평균의 중심은 정의되지 않을 수 있어 상태 명시; NaN을 유한한 값으로 변조하지 않음

## 10. 결과 저장과 재현성

- 기본 지표 long table: `run_id, scenario_id, model_id, b, r, regime, conditioning, point_id, x1, x2, alpha, interval_id, lower, upper, coverage, center, half_width, length`
- 분해 long table: 동일 key + `metric, component, value`; marginal 행은 위치 null·명시적 집계 표시, 위치별 값과 혼합 금지
- Shared benchmark를 모델별로 복제 저장할 경우 benchmark ID를 유지하여 독립 실험처럼 중복 집계하지 않음
- b별 marginal 요약, fixed-x의 b별 r 요약, fixed-calibration 개별 값과 반복 요약을 분리
- 모든 evaluated point의 기본값·분해 보존; 임의 평가 입력 전체 출력은 선택적으로 저장하고 최소한 seed·표본 수·b별 요약을 보존
- Parquet 또는 동등한 타입 보존 형식 기본, CSV 요약 선택; seed manifest·resolved config·패키지 버전·코드 버전·수치 진단·실패 로그 보존
- Run ID와 checkpoint를 사용해 중단 후 재개 가능하게 구현; 설정·코드 버전 불일치 시 기존 결과에 이어 쓰지 않음
- 기존 결과를 덮어쓰지 않음; stage별 완료 여부와 `b,r`를 명시하고 실패 반복을 조용히 제외하지 않음
- Heatmap은 저장 결과를 읽는 별도 함수; 표현 기본안은 각 축 21점의 grid이며 그 단순 평균을 marginal 적분으로 사용하지 않음

## 11. 검증과 완료 기준

- 순수 수식·계산의 의미 있는 검증부터 구현하고, 검증만으로 전체 과학적 결과가 확정된다고 주장하지 않음
- Error `cdf(ppf(p))≈p`, 표준화·support 확인; 특히 LogNormal support와 Student-t scale 검증
- Mean 평균 0·분산 $4/3$, scale 최소 양수·평균 제곱 1.18의 분석값과 수치 계산 비교
- 알려진 배열로 conformal 순위 901·951·991 및 $k=m+1$ 경계 확인
- Oracle coverage가 각 목표와 일치, 대칭·등분산에서 Oracle=res_true 확인
- 구조가 다른 임의 endpoint 예제로 모든 metric의 signed 분해 검증; 등식 성립만으로 중간 구간의 정확성이 증명되지는 않음
- CP 반폭의 x 불변·fixed-x 중심의 r 불변·calibration 중심 항 0 확인; half_width와 length의 계수 2 검증
- Gaussian·Student-t의 정확한 population fitting 반폭 차이는 비음수, LogNormal에 같은 부호를 강제하지 않음
- Population root는 독립 reference로 확인; 작은 smoke의 noisier 근사에 엄격한 population 성질을 무조건 강제하지 않음
- 작은 전체 흐름에서 5개 interval family·4개 metric·전체 차이 및 4개 component·3개 regime의 결과와 집계 key 확인
- 작은 실행에서 동일 seed 재현성과 serial/parallel 및 재개 시 결과 일치 확인; 허용오차·범위 기록
- 완료 보고에는 실제 구현 파일, 실행 명령, 수행한 검증, 실행 결과 범위, 미정·실패 설정을 명시; full 미실행이면 그대로 표시

참고: 프로젝트 규칙 파일은 저장소 루트의 `CLAUDE.md` 사용 — [Claude Code 공식 문서](https://code.claude.com/docs/en/memory)
