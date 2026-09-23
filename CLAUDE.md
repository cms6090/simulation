# MC vs Split CP — 구현 지침

## 0. 작업 진행 방식 — 계획 제시 후 승인

- 사용자가 구현·수정·실행을 지시하면 바로 코드 작성이나 파일 변경을 하지 않고, 먼저 어떻게 구현할지 제시
- 계획에 포함할 내용: 변경·생성할 파일, 구현 방식(주요 함수·처리 흐름), 관련된 이 문서의 계약 조항, 미정 설정이나 불확실한 부분
- 사용자가 ok로 승인한 뒤에만 진행
- 사용자가 계획을 수정하면 수정한 계획을 다시 제시하고 재승인 후 진행
- 진행 중 승인한 계획과 달라져야 하는 상황이 생기면 멈추고 보고한 뒤 재승인 요청
- 승인받은 범위 밖의 작업은 하지 않음
- 계획 작성을 위한 파일 읽기 등 읽기 전용 확인은 승인 전 허용

## 1. 목적과 우선순위

- 이 파일은 MC·CP 예측구간 비교 시뮬레이션의 구현 지침
- 같은 training data에서 학습한 **동일한 fitted model**을 MC와 CP가 공유
- MC는 training residual로 추정한 정규 오차분포에서 가상 반응값을 생성하는 방법
- CP는 별도 calibration absolute residual과 conformal 순위로 구간을 구성하는 방법
- 주 결과는 true model의 새 반응값에 대한 실제 포함 비율과 구간 길이
- **첫 단계는 Random Forest가 true 함수를 충분히 잘 근사하는지 확인하는 것**
- 함수 근사 정확도를 먼저 보고한 뒤 잔차분산 추정과 MC·CP 비교 진행
- 현재 모델은 Random Forest 하나이며, 후속 deep learning 등 추가 가능
- 최신 사용자 지시 우선, 다음으로 이 문서와 동반 `시뮬.md` 적용
- 두 문서 충돌은 조용히 추정하여 해결하지 말고 구체적으로 보고
- 설명은 한국어, 코드 식별자는 영어, 모델 명칭은 fitted model 사용
- 구현 언어·라이브러리 제안: Python·NumPy·SciPy·scikit-learn

## 2. 확정 설정과 미정 설정

확정값

```yaml
design_version: fitted_gaussian_mc_vs_split_cp_v1
means: [linear, nonlinear]
scales: [homo, hetero]
errors: [gaussian, student_t, lognormal]
models: [random_forest]
d: 2
n_train: 1000
n_cal: 1000
M_MC: 1000
R_MC: 200  # 임시 구간 생성·평가 반복 수
true_y_per_interval: 1
alphas: [0.10, 0.05, 0.01]
evaluation_inputs: training_inputs
mc_refit: false
mc_error_family: gaussian
coverage_estimator: empirical_inclusion
calibration_repeat_policy: fixed_dataset
viz_grid: {x1: [-0.8, -0.4, 0.0, 0.4, 0.8], x2: [-0.8, -0.4, 0.0, 0.4, 0.8]}
```

- True DGP: Y=f(X)+sigma0(X)*epsilon, E(epsilon)=0, Var(epsilon)=1
- X의 두 좌표는 독립 U(-1,1), training 입력은 생성 후 고정, calibration 입력은 별도 생성
- linear: f(x)=sqrt(2)*(x1+x2)
- nonlinear: f(x)=sqrt(6/11)*(2*sin(pi*x1)+x2+x1*x2)
- homo: sigma0(x)=1
- hetero: sigma0(x)=(0.4+1.2*sin(pi*x1)**2)/sqrt(1.18)
- gaussian: epsilon~N(0,1)
- student_t: epsilon=T/sqrt(3), T~t(df=3)
- lognormal: epsilon=(exp(Z)-exp(0.5))/sqrt(e*(e-1)), Z~N(0,1)
- 12개 DGP, Random Forest 기준 12개 model–DGP 조합, alpha별 36개 결과 조건
- Training·calibration·true 평가 반응값은 해당 scenario의 동일 DGP로 생성
- MC는 모든 scenario에서 fitted mean+N(0,sigma2_hat)를 사용
- MC의 분산은 training residual에서 추정한 scalar이며 true sigma0(X)를 사용하지 않음
- 비정규 DGP에서도 MC error family를 true error family로 자동 변경하지 않음
- 조건별 결과에는 MC 정규·등분산 가정의 적합 여부에 따른 영향도 포함
- sigma0는 표준편차 함수, sigma0(X)**2는 조건부 분산
- Error sample·CDF·PPF는 같은 표준화 사용, lognormal support 처리 확인
- CDF: Gaussian은 norm.cdf(u), Student-t는 t.cdf(sqrt(3)*u,df=3)
- LogNormal CDF: a=exp(0.5), c=sqrt(e*(e-1)); c*u+a<=0이면 0, 아니면 norm.cdf(log(c*u+a))
- Error 분산 1, E_X[sigma0(X)**2]=1, 두 mean의 모집단 분산 4/3
- 유한 고정 입력의 평균 true 분산이 정확히 1이어야 한다고 assert하지 않음
- 모델 인터페이스는 fit(X,y), predict(X), model_id, 학습 설정 기록을 공통으로 제공
- Deep learning 추가 시 모델 adapter를 구현하고 MC·CP·평가 모듈은 재사용
- Training·calibration은 1:1이며 전체 1,000개를 나누는 구조가 아님
- Calibration은 별도 입력을 생성하고 true model로 반응값 생성

미정값

```yaml
variance_estimator_by_model: null
rf_hyperparameters: null
fit_accuracy_criteria: null
fit_pilot_tuning_policy: null
B: null
training_design_repeat_policy: null
master_seed: null
mc_quantile_method: null
```

- 사용자 확정값과 구현 제안값을 설정 메타데이터에서 구분
- 미정값은 함수 인자·config로 노출하고 silent fallback 금지
- 모듈 구현과 smoke 실행은 진행 가능하며 임시값은 별도 smoke config에 명시
- Full 실행은 해당 실행에 필요한 필수 설정이 모두 정해져야 가능
- 하나의 fitted model 실험도 입력별 R_MC=200 구간 생성·평가 수행
- B는 추가 training 반복이며 R_MC와 구분. calibration_repeat_policy는 fixed_dataset으로 확정(7절), B와는 별개 사안

## 3. 입력과 데이터 계약

- X_train.shape=(1000,2), y_train.shape=(1000,)
- 현재 노트북의 training_x·training_y는 각각 X_train·y_train에 해당
- Training 데이터는 point_id, x1, x2, y를 포함한 training_set 하나로 관리
- 입력 전용 테이블을 별도로 생성하거나 중복 저장할 필요 없음
- X_eval=X_train 값 복사 또는 read-only 공유, point_id를 통해 대응 보존
- X_cal.shape=(1000,2), y_cal.shape=(1000,)
- Calibration 입력은 별도 생성기를 사용하고 X_train을 그대로 복사하지 않음
- 이산 입력에서 우연히 같은 좌표가 나온다는 이유로 calibration 데이터를 강제로 제거하지 않음
- Training true 오차, calibration true 오차, MC 오차, 평가 true 오차는 독립 stream
- 동일 scenario·실험에서 모델 간 training·calibration·true 평가값 공유
- MC·CP 각각 따로 fitting하지 않음
- Training y·calibration y·MC y_star를 coverage 평가값으로 재사용하지 않음
- Training 입력 위치 재방문 성능이며 새로운 X에 대한 일반화 성능과 구분

## 4. 한 실험의 실행 순서

1. 고정 입력 X_train을 로드하거나 명시된 설계로 구성
2. True model에서 y_train 생성
3. 별도 calibration 생성 규칙 준비, calibration_repeat_policy=fixed_dataset이므로 실험 전체에서 한 세트만 생성
4. 각 모델 j를 training data로 정확히 한 번 학습
5. train_pred=model.predict(X_train)과 true_mean(X_train)을 비교하여 함수 RMSE·MAE·위치별 오차를 먼저 계산·저장하고 근사 상태 확인
6. 함수 근사 확인 결과와 설정을 기록한 뒤 train_residual=y_train-train_pred로 sigma2_hat 추정
7. 입력·모델·sigma2_hat을 고정하고 r=1..R_MC 반복, R_MC=200은 임시값
8. 각 입력에서 MC 반응값 M_MC=1000개를 새로 생성하여 해당 반복 구간 구성
9. calibration_repeat_policy=fixed_dataset에 따라 1번에서 이미 확정된 CP 구간을 재사용 (재계산 없음)
10. 각 입력에서 true DGP의 독립 반응값 하나를 새로 생성하여 두 구간에 공통 적용, 포함 여부 0·1 저장
11. 입력별 200회 포함 여부 합/R_MC 계산, 반복별 길이와 평균·SD 및 함수 진단 저장

모든 alpha에 같은 fitted model·MC 표본·calibration residual을 재사용

### 함수 근사 확인 단계

- 구간 생성 전에 DGP별 true mean·fitted mean·오차·함수 RMSE·MAE·절대오차 분위수를 먼저 출력
- 관측 training y에 대한 residual RMSE를 true 함수 RMSE로 대체하지 않음
- 함수 비교 산점도와 위치별 오차로 평균 지표가 가리는 부정확한 영역 확인
- 적합이 부족한 조건은 모델 설정·적합 상태를 먼저 검토하고 미확인 상태를 성공으로 표시하지 않음
- 충분한 정확도의 기준은 미정이므로 자의적인 threshold·통과 판정 금지
- 설정 탐색은 별도의 fit pilot 단계로 제공하고, 본 실험에서 고정한 RF 설정을 사용
- 잘 맞는 seed만 선택하거나 실패한 반복을 조용히 버리지 않음
- True 함수를 튜닝에 사용하면 simulation-only tuning으로 기록
- 추가 diagnostic 입력의 사용 여부와 평가 범위를 명시
- fit_diagnostics 모듈을 MC·CP 없이 독립 실행 가능하게 구현
- 함수 근사가 좋다는 이유로 분산 추정·정규분포 가정의 정확성까지 보장한다고 해석하지 않음

## 5. 분산 추정 계약

- sigma2_hat은 training residual에서 계산
- 분산 추정에 사용하는 데이터는 training residual이며 true 분산으로 대체하지 않음
- 모델별 추정 규칙·분모·보정·사용 표본 수를 결과에 기록
- RF의 잔차분산 추정 규칙은 미정이며 모델에 근거 없는 자유도 보정을 도입하지 않음
- RF의 SSE/n, OOB 등의 선택은 별도 결정이며 아직 확정되지 않음
- OOB로 변경하는 경우 training in-sample residual 방식과 다르다는 점을 사용자에게 명시
- 추정값이 음수·비유한이면 오류, 0이면 퇴화 구간 상태 기록 후 별도 처리
- 작은 상수로 강제 치환하여 적합 문제를 숨기지 않음

## 6. MC 구간 계약

- 함수 입력: fitted_predictions, sigma2_hat, M_MC, alphas, rng, quantile_method
- sigma_hat=sqrt(sigma2_hat)
- 각 r에서 y_star[i,s]=fitted_predictions[i]+sigma_hat*z_r[i,s], z_r~N(0,1) 새로 생성
- 입력별 구간 R_MC개 구성, 이전 endpoint 복사로 반복을 대체하지 않음
- lower_mc[i,r,alpha], upper_mc[i,r,alpha]로 저장하거나 r별 chunk 처리
- np.random.normal의 scale에 sigma2_hat을 넣지 않음
- 분위수 축은 MC 표본 축, 입력 축으로 pooling 금지
- 입력별 독립 MC 표본을 구현 기본안으로 사용하고 stream 정책 기록
- 같은 MC 표본에서 모든 alpha의 lower·upper 계산
- MC quantile method를 명시하고 CP 순위 계산과 분리
- MC 내부 model.fit 호출 금지
- 가상 표본에서 residual을 다시 계산해 sigma2_hat을 재추정하지 않음
- 실제 endpoint midpoint는 finite MC 때문에 fitted prediction과 다를 수 있음
- true model이나 true quantile로 MC 구간을 대체하지 않음

수치 진단: fitted_prediction+sigma_hat*norm.ppf(p)와 MC endpoint 차이 기록

이 기준은 fitted mean과 추정분산으로 정해지는 정규분포의 정확한 분위수

## 7. CP 구간 계약

- 같은 model 객체·같은 train_pred 사용
- Calibration 세트별 cal_pred=model.predict(X_cal), score=abs(y_cal-cal_pred) 계산
- k=ceil((n_cal+1)*(1-alpha)), k<=n_cal이면 sorted_score[k-1]
- k>n_cal이면 q_hat=inf
- n_cal=1000에서 alpha 0.10·0.05·0.01의 순위 901·951·991
- 정수 경계 부동소수점 처리를 검증하고 CP에 보간 quantile을 사용하지 않음
- cp_lower=train_pred-q_hat, cp_upper=train_pred+q_hat
- 한 fitted model·calibration·alpha에서 CP 반폭은 모든 입력에 동일
- calibration_repeat_policy=fixed_dataset 확정: 모든 r에 동일 CP endpoint 사용
- 확정 근거: MC·CP가 동일 fitted model을 공유하므로 "모델에 따른 차이"를 반복 요인으로 넣지 않는다는 전제와 일관되는 정책. calibration을 r마다 재생성하면 CP 쪽에만 별도의 변동 요인이 추가되어 두 방법의 "고정 대상"이 달라짐
- resample_each_r(각 r에서 입력·반응값 1000개를 새로 생성하고 q_hat 계산)은 채택하지 않음. 재검토가 필요하면 구체적 근거와 함께 별도로 결정
- cal_id는 r과 별도로 저장하되 fixed_dataset에서는 실험 전체에서 단일 값으로 고정
- 새 fitted model을 사용하면 calibration 데이터를 고정해도 residual은 재계산
- 고정 training 위치의 coverage를 distribution-free marginal 보장으로 주장하지 않음

## 8. Coverage와 모델 진단

- true_pred=true_mean(X_eval)
- y_test[i,r]=true_pred[i]+sigma0(X_eval[i])*epsilon_test[i,r]
- epsilon_test는 해당 scenario의 표준화 error family에서 생성
- I_A[i,r]=(lower_A[i,r]<=y_test[i,r]) & (y_test[i,r]<=upper_A[i,r])
- coverage_A[i]=sum(I_A[i,:])/R_MC, A는 mc 또는 cp
- 매 r에서 true 반응값 하나를 생성하고 해당 r의 구간에만 대응
- fixed_dataset CP도 평가를 위해 같은 endpoint를 r에 broadcast 가능
- R_MC=200이면 위치·alpha·방법별 indicator 정확히 200개
- 한 true y 재사용, 구간 하나 재사용으로 MC 반복 대체, 모든 구간×모든 true y 교차평가 금지
- 반복 행에는 covered=0/1 저장, 요약 행에는 n_covered와 n_evaluated=R_MC 저장
- 위치별 길이는 각 r의 값과 200회 평균·SD 저장
- paired difference=mean(I_cp-I_mc), 같은 true 평가값 사용
- 기본 요약=입력별 coverage의 동일 가중 평균
- summary_scope='fixed_design_average', marginal로 자동 라벨링 금지
- 함수 delta=train_pred-true_pred, RMSE=sqrt(mean(delta**2)), MAE=mean(abs(delta))
- Training residual RMSE와 true 함수 RMSE는 다른 변수·다른 열
- true_variance[i]=sigma0(X_eval[i])**2
- design_noise_variance=mean(true_variance)
- variance_gap_design=sigma2_hat-design_noise_variance
- sigma2_hat은 scalar이며 이분산 함수 전체를 추정한다고 해석하지 않음
- 모집단 평균 noise variance=1과 유한 설계 평균을 구분
- center=(lower+upper)/2, half_width=(upper-lower)/2, length=upper-lower
- Coverage와 length의 CP−MC 차이를 직접 보고
- 무한 endpoint는 coverage와 길이를 적절히 처리하고 center 미정 상태 별도 기록

선택적 검증: 해당 DGP의 표준화 error CDF로 F_error((upper-true_pred)/true_scale)-F_error((lower-true_pred)/true_scale) 계산

- CDF 검증값을 주 결과 empirical coverage와 다른 열에 저장
- CDF 진단은 각 r의 구간에서 계산한 뒤 평균, indicator 하나가 해당 확률과 같아야 한다고 assert하지 않음
- 현재 200회는 fitted model·sigma2_hat 조건에서의 평가이며 training 변동까지 포함하지 않음
- 비율 간격은 0.005, 200회로 99% coverage 정밀도가 충분하다고 단정하지 않음
- 여러 입력·alpha·방법의 값을 전부 독립 반복처럼 합쳐 표준오차를 계산하지 않음

### 8.1 5×5 grid 보조 시각화

- 주 평가는 1000개 training 위치 그대로 유지, grid는 이를 대체하지 않고 보조 시각화로만 사용
- viz_grid 좌표: x1, x2 각각 [-0.8, -0.4, 0.0, 0.4, 0.8] (확정값 yaml 참고), 조합 25개
- grid point는 1000개 결과를 근방 평균(binning)하지 않음. 각 grid point에서 4절 7~10단계(MC 반복·true y 생성·포함여부)를 독립적으로 새로 수행
- grid point에서도 fitted model·sigma2_hat·CP q_hat은 재추정하지 않고 이미 확정된 값을 predict에만 사용
- grid 전용 R_MC는 주 실험과 동일하게 200 사용, M_MC도 1000으로 동일
- 계산량은 25*200*1000=500만 개/DGP 수준으로 주 실험(1000개 위치, 2억 개/DGP) 대비 약 1/40이며 별도 예산으로 관리
- 출력은 DGP*alpha*method(MC/CP) 조합별 5x5 coverage heatmap, 필요 시 CP-MC 차이 heatmap과 함수오차(delta) heatmap도 동일 grid로 생성
- grid 결과는 point_summary.csv와 별도 테이블(예: grid_summary.csv)에 저장하고 주 결과 테이블과 혼합하지 않음
- grid 실행은 주 실험(1000개) 완료 후 독립 모듈로 수행 가능하게 구현, 주 실험 결과를 변경하거나 대체하지 않음

## 9. 반복·RNG·계산량

- RNG는 난수 생성기이며 seed로 같은 실험을 재현하기 위한 장치
- master seed와 scenario/model/b/r/cal_id/point_id/stage의 안정적 ID로 substream 분리
- Python hash()와 병렬 완료 순서로 seed 생성 금지
- train_error, calibration_input, calibration_error, model, mc_error, eval_error stream 구분
- 데이터 공유를 의도한 모델 간에는 같은 데이터 ID 사용
- 전체 실험 반복의 fitting과 MC 내부 fitting을 구분
- 반복 정책은 입력 고정 범위, calibration 입력 고정 여부, calibration 반응값 고정 여부를 각각 기록. calibration은 calibration_repeat_policy=fixed_dataset에 따라 입력·반응값 모두 실험 전체에서 고정
- 공유 calibration이 있는 반복들을 모두 독립 실험으로 간주하지 않음
- $B$개의 독립 실험이 실제 확보된 경우에만 반복별 요약의 SD/sqrt(B)를 해당 평균의 SE로 사용
- M_MC=1000은 구간당 가상 표본 수, R_MC=200은 구간·true y 쌍의 반복 수
- DGP·모델 하나당 1000*200*1000=2억 개 MC 값, 1000*200=20만 개 true 평가값
- RF·12개 DGP에서 MC 값 총 24억 개, 모든 alpha는 같은 표본 재사용
- 전체 배열을 한꺼번에 저장하지 않고 r·입력 chunk 단위 처리
- 입력별 prediction 캐시, chunking, 모든 alpha에서 표본 재사용
- 전체 y_star·y_test를 저장하지 않아도 count·endpoint·seed로 결과 재현 가능
- CPU 벡터화 우선, 바깥 병렬화와 RF 내부 병렬화 중첩 방지
- 생성·학습·예측·MC 분위수·평가 단계별 시간을 측정하여 실제 병목 보고
- 실행시간은 실제 측정값과 실행 환경을 함께 보고

## 10. 데이터 저장과 결과 기록

### 현재 단계 — Training dataset 저장

- training_set의 point_id, x1, x2, y를 CSV 하나에 저장
- 현재 저장 경로는 ../data/.csv이며 현재 작업 디렉터리 기준
- 저장 폴더가 없으면 생성하고 to_csv(..., index=False) 사용
- 같은 경로로 재실행하면 현재 데이터로 덮어쓰므로, 여러 DGP·학습 반복을 보관할 때는 서로 구분되는 경로 사용
- 불러온 데이터의 x1·x2 열을 training_x, y 열을 training_y로 사용하며 point_id로 대응 유지
- 같은 training_x를 MC·평가 위치로 사용하되, 저장된 training_y는 coverage 평가값으로 재사용하지 않음
- smoke_config는 DGP·seed·입력 ID·설정 상태를 기록하며 현재 단계에서는 노트북에 유지 가능
- 지금은 입력 전용 CSV나 별도 설정 JSON을 추가로 생성할 필요 없음

### 후속 단계 — 결과 파일 구성 제안

아래 파일명·분할 방식은 후속 구현을 위한 제안이며 현재 training 생성 단계에서 모두 만들 필요 없음
필요한 결과와 설정의 대응을 보존하면 파일 통합 또는 다른 저장 형식 사용 가능


| 파일 | 내용 |
| --- | --- |
| resolved_config.json | 설정값·상태·설계 버전·반복 정책 |
| model_diagnostics.csv | scenario, model, b, point_id, true_mean, true_scale, true_variance, fitted_mean, mean_error |
| variance_diagnostics.csv | scenario, model, b, estimator, denominator, design_noise_variance, population_noise_variance, sigma2_hat, variance_gap_design |
| interval_metrics.csv | scenario, model, b, r, cal_id, mc_id, eval_id, point_id, alpha, method, lower, upper, center, half_width, length, y_true, covered |
| point_summary.csv | scenario, model, b, point_id, alpha, method, n_covered, n_evaluated, coverage, length_mean, length_sd |
| method_comparison.csv | 동일 비교 key, coverage_cp_minus_mc, length_cp_minus_mc |
| fixed_design_summary.csv | 입력 평균 coverage·length, 함수 RMSE·MAE, 평균 대상 |
| grid_summary.csv | scenario, model, alpha, method, grid_x1, grid_x2, coverage, length_mean, delta (8.1절, 보조 시각화용) |
| numerical_checks.csv | MC endpoint 검증, 선택적 CDF 검증 |
| run_manifest.json | seed·버전·실행 범위·시간·상태 |

- r은 1..R_MC의 구간·평가 반복 ID, b는 학습 실험 ID, cal_id는 calibration ID
- 200개 indicator를 서로 다른 fitted model의 독립 반복으로 해석하지 않음
- 모델별 고정 입력 요약과 반복 간 요약을 별도로 저장
- 실행 결과에 run_id를 부여하고 데이터·설정·코드 버전이 일치할 때만 중단된 실행 재개
- Plot은 저장 결과를 읽는 별도 모듈. 5×5 grid heatmap(8.1절)은 확정된 viz_grid 좌표만 사용하고, 그 외 임의의 heatmap grid를 자동 추가하지 않음

## 11. 의미 있는 검증과 완료 보고

1. 함수 근사 진단이 구간 비교보다 먼저 실행되고, MC·CP가 동일 fitted model 예측을 사용하며 MC 내부 학습이 없는지 확인
2. 알려진 residual 배열에서 CP 순위와 무한 구간 경계 확인
3. 주어진 fitted mean·분산에서 MC 분위수가 추정 정규분포 분위수와 수치적으로 일치하는지 확인
4. 입력·R_MC·M_MC 축 구분, r마다 새 MC 표본과 true y 하나 생성, 200개 indicator 합/200 확인
5. 작은 예제의 포함 count와 직접 계산 결과 비교
6. 함수 RMSE가 관측 y가 아닌 true mean을 기준으로 계산되는지 확인
7. CDF 진단을 사용하는 경우 empirical coverage와 표본오차 수준에서 비교
8. Seed 재현성과 chunk 처리 후 결과 집계 확인
9. 12개 DGP의 sample·CDF·PPF 표준화와 hetero의 sqrt(1.18) 정규화 확인
10. 비정규·이분산 조건에서 true 평가 생성과 Gaussian MC 생성이 올바르게 분리되는지 확인
11. 5×5 grid는 1000개 결과의 binning이 아니라 grid point별 독립 MC·CP 계산 결과인지, fitted model·sigma2_hat·q_hat은 grid에서 재추정되지 않았는지 확인

- 작은 smoke 성공을 연구 full 결과 또는 99% MC 정밀도 검증 완료라고 보고하지 않음
- 완료 보고에 구현 파일, 실행 명령, 실제 수행한 검증, 미정 설정, 미실행 범위 명시
- 현재 문서는 구현 지침이며 코드 실행·성능 검증 완료를 뜻하지 않음
