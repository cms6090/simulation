# MC vs Split CP 시뮬레이션 — Claude Code 구현 지침

## 1. 작업 목표와 우선순위

Python Jupyter Notebook `notebooks/MC_CP_simulation.ipynb`를 작성한다. 설계 기준 문서는 프로젝트 루트의 `시뮬.md`이며, 이에 따라 Random Forest 기반 MC·split CP의 **고정 grid 입력별 conditional coverage와 평균 예측구간 길이**를 비교한다.

- **작업은 단계별로 진행한다.** 각 단계에서 무엇을 만들거나 바꿀지(대상 파일, 내용, 설치할 패키지, 실행할 명령) 먼저 사용자에게 계획으로 설명하고, 사용자가 OK한 뒤에 파일 생성·수정·설치·실행을 한다. 승인받은 범위를 넘는 작업은 다음 단계 계획으로 다시 제안한다. 읽기 전용 확인(파일 열람·검색)은 승인 없이 해도 된다.
- 설명과 주석은 한국어, 코드 변수명은 영어로 작성한다. 수식과 코드의 대응을 설명한다.
- 노트북은 새 커널에서 위에서 아래로 실행 가능하게 만든다. 숨겨진 셀 상태나 수동 실행 순서에 의존하지 않는다.
- 주 구현은 노트북에서 이해할 수 있게 작성한다. 불필요한 클래스·프레임워크·여러 모듈로 분산하지 않는다.
- **코드는 사람이 순서대로 짜듯 작성한다.**
  - 노트북은 생성 스크립트로 한 번에 찍어내지 않는다. 셀을 하나씩 추가하고 실행해 결과를 확인한 뒤 다음 셀로 넘어간다.
  - 함수를 모두 먼저 정의하고 나중에 실행하는 구조를 피한다. 한 scenario를 실제로 따라가며(training 생성 → 출력 확인 → 기준 RF → 확인 → CV …) 단계마다 shape·값을 출력한다. 반복 루프는 한 번 따라간 코드를 함수로 묶어 만든다.
  - 딕셔너리 병합(`{**a, ...}`), 중첩 설정 딕셔너리, 함수형 한 줄 표현 대신 `if`/`else`와 평범한 변수를 쓴다. 저장용 메타데이터는 저장하는 단계에서 만든다.
  - 확인은 값을 출력해 눈으로 비교하게 하고, assert는 꼭 필요한 곳에만 짧게 쓴다. assert를 반복문에 몰아넣지 않는다.
- 결과를 보고 coverage가 좋아지도록 seed·설정을 선택하거나 실패한 반복을 제외하지 않는다.
- 노트북은 `notebooks/`에 둔다. 실행 위치(작업 디렉터리)와 무관하게 프로젝트 루트를 찾아 결과 경로를 지정한다. 예: 현재 디렉터리 또는 상위 디렉터리 중 `시뮬.md`가 있는 곳을 `PROJECT_ROOT`로 사용하고, 찾지 못하면 오류를 낸다.
- 이번 산출물은 노트북과 실행에 필요한 간단한 의존성 목록이다. 구현 후 작은 smoke 실행으로 검증하고, 실행한 범위를 정확하게 보고한다.

## 2. 확정 설계와 구현 기본값

### 확정 설계

```python
SCENARIOS = [
    'linear_homo_gaussian',
    'linear_homo_student_t',
    'nonlinear_homo_gaussian',
    'nonlinear_homo_student_t',
]
N_TRAIN = 1000
N_CAL = 1000
GRID_AXIS = [-1.0, -0.5, 0.0, 0.5, 1.0]
ALPHAS = [0.10, 0.05]
R = 200  # 구간 생성·평가 반복
B = 200  # 각 r 안에서 생성하는 가상 training 데이터셋·재학습 모델 수
VARIANCE_ESTIMATOR = 'cv'
MC_ERROR_FAMILY = 'gaussian'
```

Training과 calibration은 각각 1,000개이다. 하나의 1,000개 표본을 둘로 나누지 않는다. Grid는 두 축의 모든 조합 25개이다. 각 scenario에서 원래 training은 한 번 생성하고, 기준 RF도 한 번 학습한다.

### 설계 문서에 수치가 없는 항목의 구현 기본값

아래는 연구에서 확정된 값이 아니라, 실행 가능한 초안을 위한 **변경 가능한 구현 제안값**이다. 첫 설정 셀과 `config.json`에서 이 구분을 명시한다. 사용자가 별도로 지정한 값이 있으면 우선 적용한다.

```python
MASTER_SEED = 0
CV_FOLDS = 10
MC_QUANTILE_METHOD = 'linear'
RF_FIXED_PARAMS = dict(
    n_estimators=1000,   # smoke에서는 30
    max_depth=None,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
)
RF_PARAM_GRID = {        # 24개 조합
    'min_samples_leaf': [5, 10, 20, 40],
    'max_features': [0.5, 1.0],
    'max_samples': [0.5, 0.7, 1.0],
}
```

### RF Fit 절차 = OOB grid search + 학습

이 연구에서 RF의 `Fit(D)`는 **튜닝을 포함한 절차**이다. 기준 RF, CV fold RF, MC 재학습 RF 모두 같은 Fit을 적용한다.

1. `RF_PARAM_GRID`의 모든 조합에 대해 `RF_FIXED_PARAMS`와 합쳐 RF를 데이터 D 전체로 학습한다.
2. 각 RF의 `oob_prediction_`으로 OOB MSE `mean((y - oob_prediction_)**2)`를 계산한다. sklearn `oob_score_`(R²)는 선택 기준으로 쓰지 않는다.
3. OOB MSE가 가장 작은 조합을 선택한다. 동률이면 grid 순서상 먼저 나온 조합을 선택한다.
4. 선택된 조합의 RF가 곧 최종 모델이다(OOB이므로 별도 재학습 없음). 탐색과 최종 모델의 트리 수는 같다.

- OOB 예측이 없는 관측치(NaN)가 생기면 오류를 낸다. 몰래 제외하지 않는다.
- 후보 조합을 하나씩 학습하면서 지금까지 가장 좋은 모델만 메모리에 유지한다.
- 자동 튜닝 외 추가 튜닝(조합 추가, 결과를 본 grid 변경)을 하지 않는다. grid를 바꾸면 사용자 승인 후 config에 기록한다.
- 해석상 한계: MC 구간은 튜닝 선택의 변동까지 반영한다. OOB 표본 수는 `max_samples`에 따라 달라진다.
- `random_state`는 모델마다 재현 가능한 별도 seed를 사용한다(§9.3).
- CV fold 수와 MC 분위수 방식을 결과 메타데이터에 저장한다.
- Full 설정을 줄여 놓고 본 실험인 것처럼 보고하지 않는다. 별도 `RUN_MODE='smoke'/'full'`로 실행 규모를 구분한다.

## 3. DGP와 배열 계약

입력의 두 좌표와 관측치는 독립적으로 U(-1,1)에서 생성한다.

$$
m_L(x)=x_1+x_2,\qquad
m_N(x)=\sqrt{2/3}\sin(\pi x_1)+x_2.
$$

모든 조건에서 표준편차는 1이다.

$$
Y=m(X)+\varepsilon,\qquad
\varepsilon_G\sim N(0,1),\qquad
\varepsilon_T=T/\sqrt3,\quad T\sim t_3.
$$

- Student-t는 반드시 `standard_t(df=3) / sqrt(3)`으로 표준화한다.
- 두 오차의 평균은 0, 분산은 1이다. 두 평균함수의 입력 분포에 대한 평균은 0, 분산은 2/3이다.
- 참평균은 DGP 생성과 진단에만 사용한다. fitted model이나 추정분산을 참함수·참분산으로 대체하지 않는다.
- `X_train.shape == (1000, 2)`, `y_train.shape == (1000,)`.
- `X_grid.shape == (25, 2)`, grid 순서와 `grid_id`를 처음에 정한 뒤 고정한다.
- `X_cal.shape == (1000, 2)`, `y_cal.shape == (1000,)`.
- 각 r의 `y_grid_true.shape == (25,)`.
- 각 r의 `mc_predictive_samples.shape == (B, 25)`: 행은 재학습 모델 b, 열은 grid i이다.
- NumPy의 `X[:, 0]`, `X[:, 1]`은 각각 설명변수 열이고, `X[i, :]`가 하나의 입력 벡터이다.
- 별도의 random test 및 marginal 집계는 만들지 않는다.

## 4. 기준 모델과 CV 잔차분산

각 scenario에서 다음을 한 번만 수행한다.

1. True DGP로 training 생성.
2. 전체 training으로 기준 RF `base_model`을 Fit(OOB 튜닝 + 학습, §2).
3. `base_train_pred`와 `base_grid_pred` 계산.
4. Training 내부 K-fold out-of-fold 예측으로 분산 추정.
5. `X_train`, `y_train`, `base_model`, `sigma2_hat`, `X_grid`를 모든 r에서 고정.

관측값 j가 속한 fold를 k(j)라 할 때,

$$
r_j^{CV}=Y_j^{train}-\widehat f^{(-k(j))}(X_j^{train}),\qquad
\widehat\sigma^2=\frac1{n_{train}}\sum_{j=1}^{n_{train}}(r_j^{CV})^2.
$$

- `KFold(shuffle=True, random_state=...)` 사용. 각 관측값은 자신이 제외된 fold 모델의 예측을 정확히 한 번 받는다.
- 각 fold 모델 $\widehat f^{(-k)}$도 같은 Fit을 적용한다. 즉 fold의 학습 데이터 안에서만 OOB 튜닝을 다시 한다(nested). 기준 RF에서 고른 조합을 fold 모델에 재사용하지 않는다.
- 분산 추정은 OOF 잔차 전체의 **제곱평균**이다. `np.var(residuals)`나 fold별 RMSE의 평균을 제곱한 값으로 대체하지 않는다.
- Calibration·grid 평가 반응값을 CV에 사용하지 않는다.
- 기준 RF를 CV fold 모델로 덮어쓰지 않는다. 최종 기준 RF는 전체 training으로 학습한 모델이다.
- CV는 r 루프 밖에서 한 번 수행한다. MC 가상 데이터마다 CV·분산 재추정을 하지 않는다.
- Gaussian 생성기의 `scale`에는 `sigma_hat = sqrt(sigma2_hat)`을 전달한다.
- 잔차 제곱평균에는 평균함수 추정오차도 포함될 수 있다. 순수 오차분산의 불편추정량이라고 설명하지 않는다.
- 비유한 추정값은 오류 처리한다. 0이면 퇴화 상태를 명시하고 작은 양수로 몰래 바꾸지 않는다.
- 진단으로 `sigma2_hat`, pooled CV RMSE, grid의 true mean 대 fitted mean RMSE·MAE를 출력한다. 진단값으로 임의의 통과 기준을 만들지 않는다.

## 5. 반복 구조 — 반드시 이 순서와 공유 범위를 유지

```text
for scenario:
    training 생성 및 기준 RF 학습
    CV 잔차로 sigma2_hat 추정
    grid와 기준 모델 예측값 준비
    for r in 1..R:
        true DGP에서 grid 반응값 25개 새로 생성
        true DGP에서 calibration X, Y 각 1000개 새로 생성
        for b in 1..B:
            원래 training X에서 가상 Y* 1000개 생성
            새로운 RF를 가상 training으로 학습
            모든 grid 25개를 한 번에 예측
            예측값 각각에 독립적인 새 Gaussian 오차 추가
        동일한 B×25 MC 표본에서 두 alpha의 구간 생성
        기준 RF와 이번 calibration에서 두 alpha의 CP 구간 생성
        동일한 y_grid_true로 두 방법·두 alpha의 포함 여부 평가
        반복별 구간 하한·상한·길이·포함 여부 저장
    입력 벡터·alpha·방법별 집계 및 파일 저장
```

고정 대상: 원래 training, 기준 RF, 추정분산, grid X.
재생성 대상: calibration X·Y, grid Y, MC 가상 training Y, MC 재학습 모델, MC 새 관측오차.

평가용 grid Y는 매 r마다 입력별 하나를 생성한다. MC·CP와 두 alpha가 같은 값을 공유한다. 학습 Y, calibration Y, MC 가상 Y를 평가 Y로 재사용하지 않는다.

## 6. MC 구간

각 r, b에서 원래 training 입력을 그대로 사용한다.

$$
Y_j^{*(r,b)}=\widehat f(X_j^{train})+\epsilon_j^{*(r,b)},
\qquad \epsilon_j^{*(r,b)}\sim N(0,\widehat\sigma^2).
$$

이 가상 데이터로 새로운 RF를 학습한 후,

$$
\widetilde Y_i^{*(r,b)}
=\widehat f^{*(r,b)}(x_i)+\epsilon_{new,i}^{*(r,b)},
\qquad \epsilon_{new,i}^{*(r,b)}\sim N(0,\widehat\sigma^2).
$$

- 가상 training 오차와 새 관측오차는 별도 독립 난수이다. 모든 i, r, b에서 필요한 새 오차를 생성한다.
- Student-t scenario에서도 MC 가상 오차는 Gaussian이다. True DGP 오차와 혼동하여 Student-t로 자동 변경하지 않는다.
- 매 b에 **RF 전체를 하나 학습**한다. RF 안의 개별 트리 예측을 B개 모델 예측으로 대체하지 않는다.
- 매 (r,b)의 RF도 같은 Fit을 적용한다. 가상 training $D^{*(r,b)}$로 OOB 튜닝을 다시 하고, 기준 RF에서 고른 조합을 재사용하지 않는다.
- B는 가상 데이터셋 수이며 training 크기 1000이나 RF 트리 수와 다르다.
- 같은 B개 모델을 25개 입력과 두 alpha에 공유한다. 입력별·alpha별로 다시 학습하지 않는다.
- 기준 예측값에 오차만 더하는 방식으로 MC 재학습을 생략하지 않는다.
- 이전 b 모델에 이어서 학습하는 warm-start를 사용하지 않는다.
- 관측오차 추가를 생략하여 평균에 대한 구간으로 바꾸지 않는다.

각 grid 열에서 B개 가상 반응값의 분위수로 구간을 계산한다.

$$
C_{MC,\alpha}^{(r)}(x_i)
=[Q_{\alpha/2}(\widetilde Y_i^{*(r,1:B)}),
Q_{1-\alpha/2}(\widetilde Y_i^{*(r,1:B)})].
$$

`np.quantile(..., axis=0, method=MC_QUANTILE_METHOD)`와 같이 B 축으로 계산한다. 다른 grid의 값을 섞지 않는다. 실제 평가 Y는 분위수 계산에 사용하지 않는다.

## 7. CP 구간

매 r마다 독립적인 calibration X·Y를 true DGP에서 생성한다. 기준 모델은 고정한다.

$$
S_a^{(r)}=|Y_a^{cal,(r)}-\widehat f(X_a^{cal,(r)})|,
\quad
k_\alpha=\lceil(n_{cal}+1)(1-\alpha)\rceil.
$$

- 정렬한 score에서 0-based index `k-1`을 선택한다.
- n_cal=1000에서 alpha=0.10은 901번째, alpha=0.05는 951번째이다.
- k>n_cal이면 q_hat=inf로 처리한다. CP 분위수에 선형 보간을 적용하지 않는다.
- CP에서 별도 RF를 학습하지 않는다. MC의 재학습 RF도 사용하지 않는다.

$$
C_{CP,\alpha}^{(r)}(x_i)
=[\widehat f(x_i)-\widehat q_\alpha^{(r)},
\widehat f(x_i)+\widehat q_\alpha^{(r)}].
$$

같은 r, alpha에서 q_hat은 모든 grid에 공통이다. CP 길이는 모두 2q_hat이지만 중심은 입력별로 다르다. 다음 r에서는 calibration과 q_hat을 새로 계산한다.

## 8. Coverage와 길이 집계

$$
I_{h,i,r,\alpha}
=\mathbf1\{L_{h,i,r,\alpha}\le Y_i^{grid,(r)}\le U_{h,i,r,\alpha}\},
\quad
\ell_{h,i,r,\alpha}=U_{h,i,r,\alpha}-L_{h,i,r,\alpha}.
$$

$$
\widehat c_{h,\alpha}(x_i)=\frac1R\sum_r I_{h,i,r,\alpha},
\qquad
\overline\ell_{h,\alpha}(x_i)=\frac1R\sum_r\ell_{h,i,r,\alpha}.
$$

- 경계 포함 `lower <= y_true <= upper`로 평가한다.
- 포함 여부와 관계없이 모든 반복의 길이를 평균한다.
- 평균 하한·상한의 포함 여부로 coverage를 대체하지 않는다.
- 참평균 m(x_i)의 포함 여부는 주 평가가 아니다. 새 관측값 Y를 평가한다.
- 집계 key는 `[scenario, grid_id, alpha, method]`이다. alpha나 grid를 평균으로 없애지 않는다.
- 같은 grid·alpha에서 `coverage_cp_minus_mc`, `length_cp_minus_mc`를 계산한다.
- 현재는 training을 고정하고 구간 생성과 새 Y의 무작위성을 평균한 X-conditional coverage이다. Training 반복 실험이나 calibration 고정 실험으로 설명하지 않는다.
- CP가 모든 x에서 목표 conditional coverage를 보장한다고 주장하지 않는다.
- 목표 coverage와의 차이를 먼저 보고 길이를 함께 해석한다. 두 방법의 coverage가 서로 가깝다는 이유만으로 둘 다 정확하다고 판단하지 않는다.

## 9. 난수와 계산량

### 9.1 난수 관리 — master seed와 단계·반복 ID 조합

이 문서를 기준으로 새 노트북을 처음부터 구현한다. 이전 구현의 seed 목록이나 전역 RNG 상태에 의존하지 않는다. `MASTER_SEED=0` 하나를 시작점으로 사용하되, 아래 키로 각 난수 흐름을 구분한다.

```text
[MASTER_SEED, scenario_id, stage_id, r, b, fold]
```

- `scenario_id`는 아래 고정 매핑을 사용한다. 실행 목록을 재정렬하거나 일부 DGP만 실행해도 ID를 바꾸지 않는다.
- `r=1,...,R`, `b=1,...,B`, `fold=1,...,CV_FOLDS`로 사용한다. 해당 단계에 필요 없는 인덱스는 0으로 둔다.
- 아래 숫자는 원시 seed의 연속 범위가 아니라 **단계 식별자**이다. 수만 개 seed를 수동 목록으로 관리하지 않는다.
- `common`은 한 DGP 안에서 MC·CP가 기준 training과 fitted model을 공유한다는 뜻이다. 기본 구현에서는 DGP별 난수 흐름을 scenario ID로 구분한다.

| 단계 이름 | stage_id | 사용하는 인덱스 | DGP당 흐름 수 | 용도 |
| --- | ---: | --- | ---: | --- |
| `common_dataset` | 0 | 없음 | 1 | 원래 training X와 오차를 순서대로 생성 |
| `common_training` | 1 | 없음 | 1 | 기준 RF 학습 |
| `cv_split` | 2 | 없음 | 1 | CV fold 분할 |
| `cv_model` | 3 | fold | CV_FOLDS | 각 fold의 RF 학습 |
| `cp_calibration` | 4 | r | R=200 | True DGP에서 calibration X·Y 생성 |
| `mc_dataset` | 5 | r, b | R×B=40,000 | 고정 training X에서 가상 training 오차 생성 |
| `mc_model` | 6 | r, b | R×B=40,000 | 가상 training으로 RF 재학습 |
| `mc_new_error` | 7 | r, b | R×B=40,000 | MC grid 예측에 추가할 Gaussian 오차 생성 |
| `test_error` | 8 | r | R=200 | True DGP에서 실제 평가용 grid 오차 생성 |

CP는 별도 모델을 학습하지 않으므로 `cp_training` 대신 `cp_calibration`이라는 이름을 사용한다. CV를 다른 분산 추정법으로 교체하면 사용하지 않는 CV 단계 ID는 그대로 예약해 두며, 다른 단계 번호를 당기지 않는다.

### 9.2 구현 예시

```python
import numpy as np

SCENARIO_IDS = {
    'linear_homo_gaussian': 0,
    'linear_homo_student_t': 1,
    'nonlinear_homo_gaussian': 2,
    'nonlinear_homo_student_t': 3,
}
STAGE_IDS = {
    'common_dataset': 0, 'common_training': 1,
    'cv_split': 2, 'cv_model': 3, 'cp_calibration': 4,
    'mc_dataset': 5, 'mc_model': 6,
    'mc_new_error': 7, 'test_error': 8,
}

def make_random_state(scenario, stage, r=0, b=0, fold=0):
    # 모든 난수에 쓰는 생성기. 부를 때마다 새 객체를 만든다.
    key = [MASTER_SEED, SCENARIO_IDS[scenario], STAGE_IDS[stage], r, b, fold]
    words = np.random.SeedSequence(key).generate_state(4)
    return np.random.RandomState(words)
```

- **생성기는 `make_random_state` 하나로 통일한다.** 데이터·오차 생성(`uniform`, `normal`, `standard_t`)과 RF·`KFold`의 `random_state` 모두에 사용한다. `Generator`(`make_rng`)는 쓰지 않는다.
- 통일 이유: 단계마다 생성기 종류를 구분할 필요가 없어 읽기 쉽고, `RandomState`는 NumPy 버전이 달라도 같은 난수를 내도록 보장되어 여러 컴퓨터에서 나눠 실행할 때 안전하다. 시드 재료는 128비트(정수 4개)라 키 간 시드 충돌 걱정이 없다.
- 같은 `RandomState` 객체를 여러 모델에서 재사용하지 않는다. 각 키에서 새 객체를 만든다.
- OOB 튜닝의 후보 조합들은 해당 단계의 같은 키를 공유한다(기준 RF는 `common_training`, fold RF는 `cv_model, fold=k`, MC는 `mc_model, r, b`). 조합마다 같은 키로 새 `RandomState` 객체를 만들어 전달한다. 튜닝 때문에 stage를 추가하지 않는다.

### 9.3 난수 사용과 공유 규칙

- 원래 training은 `make_random_state(scenario, 'common_dataset')` 하나를 만들고, 그 생성기에서 X와 오차를 차례로 뽑는다. X와 오차를 뽑기 전에 같은 키로 각각 재초기화하지 않는다.
- 기준 RF는 `make_random_state(scenario, 'common_training')`을 사용한다.
- CV 분할은 `make_random_state(scenario, 'cv_split')`, 각 fold RF는 `make_random_state(scenario, 'cv_model', fold=k)`를 사용한다. Fold별 random_state를 지정할 수 있도록 명시적 fold 루프를 사용한다.
- 매 r의 calibration은 `make_random_state(scenario, 'cp_calibration', r=r)`에서 X와 오차를 순서대로 생성한다.
- 매 (r,b)의 가상 training 오차는 `make_random_state(scenario, 'mc_dataset', r=r, b=b)`에서 길이 N_TRAIN으로 뽑는다. Training X는 재생성하지 않는다.
- 매 (r,b)의 RF는 `make_random_state(scenario, 'mc_model', r=r, b=b)`로 새로 학습한다.
- MC 새 관측오차는 `make_random_state(scenario, 'mc_new_error', r=r, b=b)`에서 `normal(0, sigma_hat, size=len(X_grid))`로 한 번에 뽑는다.
- 실제 평가 오차는 `make_random_state(scenario, 'test_error', r=r)`에서 true DGP에 따라 길이 len(X_grid)로 뽑는다. Gaussian은 표준정규, Student-t는 t_3/sqrt(3)이며 현재 true 표준편차는 1이다.
- **MC 새 관측오차와 실제 test 오차는 다른 난수 흐름이다.** 전자는 구간 구성, 후자는 포함 여부 평가에만 사용한다.
- Seed는 관측값 하나마다 필요하지 않다. 생성기 하나에서 벡터를 뽑는다. Grid i마다 같은 키로 생성기를 재초기화해서 같은 오차를 반복하지 않는다.
- 같은 r의 실제 test Y 벡터는 MC·CP와 두 alpha에 공통으로 사용한다. Alpha별 난수 흐름을 따로 만들지 않는다.
- 같은 (r,b)의 RF가 모든 grid를 예측한다. Grid별 모델 seed나 재학습을 추가하지 않는다.
- Python `hash()`, 전역 `np.random.seed()`, 실행 순서에 따른 단순 seed 증가, 병렬 작업 완료 순서에 의존하지 않는다.
- 위 키 방식은 난수 흐름을 재현 가능하게 분리하기 위한 것이다. 유한한 의사난수로 수학적 독립성을 증명한다고 설명하지 않는다.
- `config.json`에 MASTER_SEED, SCENARIO_IDS, STAGE_IDS, 키 순서, 인덱스 규칙, RandomState 변환 규칙(`RandomState(SeedSequence(key).generate_state(4))`)을 저장한다. 개별 seed 80,000개 이상의 목록은 저장할 필요 없다.
- 같은 키·같은 설정·같은 실행 환경에서는 재현 가능하게 구현한다. 라이브러리 버전도 저장한다.

### 9.4 계산량과 실행

- MC 표본을 전부 저장하지 않는다. 각 r에서 B×25 배열만 유지하고 개별 RF는 예측 후 해제한다.
- 기본은 바깥 루프 순차 실행과 RF 내부 병렬화이다. 바깥 병렬화를 사용하면 RF 내부 `n_jobs=1`로 중첩 병렬화를 피한다.
- Full MC 선택 모델은 DGP당 B×R=40,000개이고, OOB 튜닝 때문에 후보 RF 학습은 DGP당 40,000×24=960,000회이다. 기준 RF 후보 24회와 CV 후보 CV_FOLDS×24=240회가 별도 추가된다.
- 길이 추가는 모델 학습 횟수나 결과 행 수를 증가시키지 않는다.
- **Scenario별로 서로 다른 컴퓨터에서 실행한다.** 노트북 설정의 `RUN_SCENARIOS`로 이번 컴퓨터가 실행할 scenario를 고른다. `SCENARIO_IDS`는 바꾸지 않으므로 어느 컴퓨터에서 실행해도 난수 키는 같다.
- 모든 컴퓨터는 고정된 `requirements.txt`(`pip freeze` 결과)로 같은 라이브러리 버전을 사용하고, `config.json`에 버전과 컴퓨터 정보를 기록한다.
- 진행률을 표시하고 r 단위로 완료 상태를 기록한다. **재개는 필수이다.** 완료된 r만 재사용하고 설정 일치와 key 중복을 확인한다.
- 각 컴퓨터의 `results/<scenario>/` 폴더를 한 곳에 모아 마지막 표시 셀에서 읽는다.

## 10. 결과 저장과 노트북 표시

노트북 위치(`notebooks/`)가 아니라 프로젝트 루트의 `results/<scenario>/` 아래(`PROJECT_ROOT / 'results'`)에 저장한다. Smoke는 `results/smoke/<scenario>/`로 분리하여 본 결과와 섞이지 않게 한다. CSV는 `index=False`로 저장한다.

| 파일 | 필수 내용 |
| --- | --- |
| `config.json` | 설계 버전, 실행 모드, 확정값/구현 기본값 구분, seed·stage 매핑, R·B·n_train·n_cal·grid·alpha, RF 고정 설정·탐색 grid·선택 규칙(OOB MSE, 동률 시 grid 순서), 기준 RF 튜닝표(조합별 OOB MSE), CV folds, sigma2_hat, MC 분위수 방식, 라이브러리 버전, 컴퓨터 정보, 완료 r 목록 |
| `model_selection.csv` | scenario, stage(`base`/`cv`/`mc`), r, b, fold, min_samples_leaf, max_features, max_samples, oob_mse — 선택된 조합만 기록 (full DGP당 약 40,011행) |
| `test_grid.csv` | scenario, grid_id, x1, x2, true_mean, base_prediction |
| `interval_metrics.csv` | scenario, r, grid_id, x1, x2, alpha, method, y_true, true_mean, lower, upper, covered, length |
| `point_summary.csv` | scenario, grid_id, x1, x2, alpha, method, n_covered, n_evaluated, coverage, length_mean |
| `method_comparison.csv` | scenario, grid_id, x1, x2, alpha, mc_coverage, cp_coverage, coverage_cp_minus_mc, mc_length_mean, cp_length_mean, length_cp_minus_mc |

- method 값은 `mc`, `cp`로 통일한다. r과 grid_id의 표기 범위를 문서화한다.
- Full에서는 DGP당 원시 결과 20,000행, point_summary 100행, method_comparison 50행이다. 전체 4개에서는 각각 80,000행, 400행, 200행이다.
- 원시 결과 key `[scenario,r,grid_id,alpha,method]`는 유일해야 한다.
- 미완료 결과에서는 n_evaluated를 실제 완료 수로 표시하고 완료 여부를 명시한다. R=200으로 나누어 완성된 결과처럼 표시하지 않는다.
- 노트북의 입력별 요약 뷰는 scenario별로 나누고 `x='[x1, x2]'`, alpha, method를 index로 표시한다. 원본 저장 데이터는 수치 좌표와 grid_id를 유지한다.
- 비교표에는 두 방법의 coverage와 평균 길이, 각각의 차이를 나란히 표시한다.
- 기본 결과는 표로 제시한다. 임의의 추가 grid나 복잡한 시각화를 필수 구현에 넣지 않는다.
- `results/`와 `.ipynb_checkpoints/`는 `.gitignore`에 추가한다. 기존 결과를 삭제하거나 Git 이력을 변경하지 않는다.

## 11. 권장 노트북 셀 구성

1. 연구 목적, 평가 대상과 반복 구조 설명
2. Imports, 실행 환경, 설정 셀과 smoke/full 선택
3. DGP 및 grid 생성 함수
4. RF 생성·CV 잔차분산 함수
5. MC 구간·CP 순위 분위수 함수
6. 한 scenario 실행 함수와 r 루프
7. 결과 저장·집계 함수
8. 작은 결정적 예제 기반 검증
9. Smoke 실행 및 진단·결과표
10. Full 실행 셀과 실행 안내
11. 저장된 결과를 읽어 비교표 표시

Full 실행 셀은 사용자가 명시적으로 실행할 수 있게 구성한다. 구현 검증 과정에서 160,000회 RF 학습을 자동 시작하지 않는다. Smoke는 예를 들어 4개 DGP, R=2, B=5, RF 트리 수 30(OOB 예측이 비지 않도록)으로 실행하되 n_train, n_cal, grid는 유지한다. Smoke 설정은 연구 결과가 아닌 구조 검증임을 표시한다. 실제 사용한 smoke 값을 저장한다.

## 12. 필요한 검증과 완료 보고

- 알려진 score 배열로 CP의 순위 선택과 k>n_cal 경계를 확인한다.
- 작은 수치 배열로 MC 분위수 축이 B 축인지 확인한다.
- OOF 예측에서 각 관측값의 학습 제외와 한 번의 예측을 확인한다.
- 작은 실행에서 MC 선택 모델이 정확히 R×B개, MC 후보 학습이 R×B×조합 수인지 확인한다. 기준 RF·CV 호출과 구분한다.
- 작은 예제에서 OOB 선택 규칙(최소 OOB MSE, 동률 시 grid 순서)과 OOB 예측 NaN 오류 처리를 확인한다.
- 각 r에서 calibration과 평가 Y를 재생성하며, 두 방법·두 alpha에 동일한 평가 Y가 대응하는지 확인한다.
- 원시 결과의 행 수·key 유일성·길이=상한−하한·집계 count를 확인한다.
- 같은 r, alpha에서 CP 길이가 grid 전체에 동일한지 확인한다.
- 동일 표본으로 계산한 95% 구간이 90% 구간을 포함하는지 확인한다.
- 동일 seed와 설정의 작은 재실행에서 결과가 재현되는지 확인한다.
- 난수 키의 단계·r·b·fold 배정이 맞는지 확인하고, 작은 예제에서 특정 키를 실행 순서와 무관하게 다시 생성할 수 있는지 확인한다.
- MC 새 관측오차와 실제 test 오차의 단계가 구분되고, 같은 r의 test Y가 두 방법·두 alpha에 공유되는지 확인한다.
- Student-t(df=3)는 두꺼운 꼬리이므로 작은 표본분산이 1과 매우 가깝다는 것을 필수 assertion으로 삼지 않는다.
- 관측 coverage가 정확히 0.90/0.95이거나 CP≥MC여야 한다는 검증을 넣지 않는다.
- RF를 재학습한 MC 표본은 일반적으로 단일 정규분포 표본이 아니다. 구간이 `base_prediction ± z*sigma_hat`와 같아야 한다고 검증하지 않는다.
- 생성한 ipynb는 nbformat 형식과 코드 문법을 확인하고 작은 실행을 수행한다. 수행하지 못한 검증은 그대로 알린다.
- 최종 보고에는 산출물, 실행 방법, 적용한 기본값, 실제 수행한 검증, full 실행 여부를 명시한다. Smoke 결과를 본 연구 결과로 제시하지 않는다.
