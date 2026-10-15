# simulation

MC vs Split CP 시뮬레이션 (설계: `시뮬.md`, 도식: `simulation.pdf`, 구현 지침: `CLAUDE.md`)

## 무엇을 하는 시뮬레이션인가

Random Forest(RF)를 기준 모델로 두고, 새 관측값 $Y$에 대한 예측구간을 만드는 두 방법을 비교한다.

- **MC (Monte Carlo)** : 기준 RF와 추정 잔차분산으로 가상 training 데이터를 여러 개 만들고, 각각으로 RF를 다시 학습한 예측값에 새 관측오차를 더해, 그 분포의 분위수로 구간을 만든다.
- **Split CP (Conformal Prediction)** : 별도의 calibration 데이터에서 기준 RF의 절대잔차 순위 분위수 $\widehat q$를 구해 $\widehat f(x)\pm\widehat q$ 구간을 만든다.

비교 지표는 **고정된 입력 벡터 25개 각각에서의 conditional coverage**와 **평균 구간 길이**이다. 구간 수준은 90%, 95%($\alpha=0.10, 0.05$)이다.

### 데이터 생성 (4개 scenario)

$$
Y=m(\underline X)+\varepsilon,\qquad X_1,X_2\overset{iid}{\sim}U(-1,1),\qquad \operatorname{Var}(\varepsilon)=1\ (\text{등분산})
$$

| Scenario | 평균함수 $m$ | 오차 $\varepsilon$ |
| --- | --- | --- |
| `linear_homo_gaussian` | $x_1+x_2$ | $N(0,1)$ |
| `linear_homo_student_t` | $x_1+x_2$ | $t_3/\sqrt3$ (두꺼운 꼬리) |
| `nonlinear_homo_gaussian` | $\sqrt{2/3}\sin(\pi x_1)+x_2$ | $N(0,1)$ |
| `nonlinear_homo_student_t` | $\sqrt{2/3}\sin(\pi x_1)+x_2$ | $t_3/\sqrt3$ |

### 진행 순서 (scenario 하나 기준)

**하이퍼파라미터 선택 (4개 scenario 공통, 한 번)** — 2026-10-15 랩미팅에서 교수님 지침으로 변경

1. `linear_homo_gaussian`의 training으로 24개 조합(`min_samples_leaf` × `max_features` × `max_samples`) RF를 각각 학습
2. 학습에 쓰지 않은 새 X 1,000개에서 참평균과의 RMSE $\sqrt{\text{mean}(\widehat f(x)-m(x))^2}$ 계산 → 가장 작은 조합 선택 (동률이면 grid 순서상 먼저)
3. 선택한 조합의 $m(x)$ 대 $\widehat f(x)$ 그래프(`hparam_check.png`)가 $y=x$ 꼴인지 확인
4. 이 조합을 4개 scenario의 기준 RF와 모든 MC 재학습 RF에 고정 (MC에서 다시 튜닝하지 않음)

**공통 준비 (scenario마다 한 번, 이후 고정)**

1. True DGP에서 training 1,000행 생성
2. 기준 RF $\widehat f$ 학습 : 고정 조합으로 RF 하나
3. 잔차분산 $\widehat\sigma^2$ : 기준 RF의 OOB 잔차 제곱평균 $\text{mean}(Y_j-\widehat f^{OOB}(X_j))^2$
4. Grid 25개 : 두 축 $\{-1,-0.5,0,0.5,1\}$의 모든 조합 $\underline x_i$

→ 하이퍼파라미터, training, $\widehat f$, $\widehat\sigma^2$, grid 위치는 모든 반복에서 고정

**반복 $r=1,\dots,R$ ($R=200$)**

1. **평가 반응값** : grid 25곳에서 true DGP로 새 $Y_i^{grid,(r)}$ 생성. MC·CP와 두 $\alpha$가 같은 값으로 채점된다.
2. **CP** : 새 calibration 1,000행 생성 → score $S_a=|Y_a-\widehat f(X_a)|$ → 정렬해서 $k_\alpha=\lceil(n_{cal}+1)(1-\alpha)\rceil$번째 값 $\widehat q_\alpha$ (90%: 901번째, 95%: 951번째) → 구간 $\widehat f(\underline x_i)\pm\widehat q_\alpha$
3. **MC** : $b=1,\dots,B$ ($B=200$)마다
   - 가상 training $Y_j^{*}=\widehat f(X_j^{train})+N(0,\widehat\sigma^2)$ (입력은 원래 training 그대로)
   - 가상 데이터로 RF를 **고정 조합으로 새로 학습** (다시 튜닝하지 않음)
   - grid 25곳 예측값에 새 관측오차 $N(0,\widehat\sigma^2)$를 더함

   → grid마다 가상 반응값 $B$개의 $\alpha/2$, $1-\alpha/2$ 분위수로 구간
4. **채점** : $L\le Y_i^{grid,(r)}\le U$이면 1, 아니면 0. 구간 길이 $U-L$도 기록.

**집계**

$$
\widehat c_{h,\alpha}(\underline x_i)=\frac1R\sum_{r}\mathbf 1\{Y_i^{grid,(r)}\in C_{h,\alpha}^{(r)}(\underline x_i)\},\qquad
\overline\ell_{h,\alpha}(\underline x_i)=\frac1R\sum_r(U-L),\qquad h\in\{MC,CP\}
$$

grid × $\alpha$ × 방법마다 coverage와 평균 길이를 계산하고, 같은 grid·$\alpha$에서 CP − MC 차이를 계산한다.

### 해석할 때 주의
- Training을 고정한 상태의 **입력별(conditional) coverage**이다. Training 자체의 변동까지 평균한 실험은 후속 과제이다.
- 목표 coverage(0.90, 0.95)와의 차이를 먼저 보고, 그다음 길이를 본다. Coverage가 부족한데 길이가 짧은 것을 더 좋은 방법으로 보지 않는다.
- Split CP는 marginal coverage를 보장하지만, 모든 입력에서 conditional coverage를 보장하지는 않는다.
- MC의 가상 오차는 모든 scenario에서 Gaussian이다. Student-t scenario에서는 참 오차분포와의 차이가 결과에 반영될 수 있다.
- $\widehat\sigma^2$(OOB 잔차 제곱평균)에는 평균함수 추정오차도 섞여 있어 순수 오차분산의 불편추정량이 아니다.
- 하이퍼파라미터를 고정하므로 MC 구간은 튜닝 선택의 변동을 반영하지 않는다. 조합은 `linear_homo_gaussian`에서 골랐으므로 다른 scenario에서는 최적이 아닐 수 있다.

### 계산량 (scenario 하나, full)

| 항목 | 수 |
| --- | --- |
| MC 재학습 모델 | $R\times B$ = 40,000개 (튜닝 없음) |
| 하이퍼파라미터 선택 | 24개 조합 학습 (`linear_homo_gaussian`, 한 번) |
| 원시 결과 (`interval_metrics.csv`) | 25 grid × 2 $\alpha$ × 2 방법 × 200 = 20,000행 |
| 입력별 요약 (`point_summary.csv`) | 100행 |

## 파일

| 파일 | 용도 |
| --- | --- |
| `시뮬.md`, `simulation.pdf` (`simulation.drawio`) | 시뮬레이션 설계 문서와 도식 |
| `run_simulation.py` | **실제 실행용.** 계산을 고칠 때는 이 파일이 기준 |
| `notebooks/simulation.ipynb` | 같은 계산의 노트북 버전 (이해용) + 저장된 결과 확인·결과표 |
| `notebooks/MC_CP_simulation.ipynb` | 한 scenario를 따라가며 계산을 확인한 노트북 + 검증 셀 (결과는 `results/check/`) |
| `.python-version`, `requirements.txt` | 고정된 Python·라이브러리 버전 |

## 1. 환경 만들기 (컴퓨터마다 한 번)

1. python.org에서 **Python 3.13.15** 설치 (`.python-version`과 같은 버전)
2. 프로젝트 폴더에서 cmd 창을 열고
   ```
   py -3.13 -m venv venv
   venv\Scripts\python -m pip install -r requirements.txt
   venv\Scripts\python --version
   ```
   마지막 줄이 `Python 3.13.15`인지 확인한다.

- `run_simulation.py`는 시작할 때 Python과 `requirements.txt`의 모든 라이브러리 버전을 확인하고, 하나라도 다르면 계산을 시작하지 않는다.
- `requirements.txt`에는 Windows 전용 패키지(`pywinpty`)가 있어 Windows에서만 설치된다.

## 2. 실행 설정 고치기

`run_simulation.py` 맨 위의 두 값을 직접 고친다 (명령줄 인자는 받지 않는다).

```python
SCENARIOS = [
    'linear_homo_gaussian',        # 돌릴 scenario만 넣는다
]
RUN_MODE = 'full'                  # 'smoke' 또는 'full'
```

| RUN_MODE | R | B | 트리 수 | 결과 위치 | 용도 |
| --- | --- | --- | --- | --- | --- |
| `'smoke'` | 2 | 5 | 30 | `results/smoke/<scenario>/` | 구조 확인용 (**연구 결과 아님**) |
| `'full'` | 200 | 200 | 1000 | `results/<scenario>/` | 본 실행 |

- 가능한 scenario: `linear_homo_gaussian`, `linear_homo_student_t`, `nonlinear_homo_gaussian`, `nonlinear_homo_student_t`
- full을 여러 컴퓨터에 나눠 돌릴 때는 컴퓨터마다 `SCENARIOS`에 하나씩 넣는다. scenario ID가 고정이라 어느 컴퓨터에서 돌려도 같은 난수를 쓴다.
- 하이퍼파라미터 선택(`linear_homo_gaussian`의 24개 조합)은 `SCENARIOS`와 상관없이 모든 컴퓨터에서 실행 시작 때 돌린다. 같은 난수 키를 쓰므로 어느 컴퓨터에서나 같은 조합이 나온다. 트리 수는 그 실행 모드의 값을 쓴다.

## 3. 실행

프로젝트 폴더의 cmd 창에서

```
venv\Scripts\python run_simulation.py > results\run_log.txt 2>&1
```

- `> results\run_log.txt 2>&1`은 Python이 아니라 **cmd에게 주는 지시**이다. 출력(`>`)과 오류 메시지(`2>&1`)를 화면 대신 `results\run_log.txt`에 저장한다. 자동으로 붙지 않으므로 매번 직접 입력한다.
  - 붙이면: cmd 창에는 아무것도 안 나오고, 끝나면 프롬프트가 다시 나타난다. 진행 상황은 `results\run_log.txt`를 열어 본다 (줄마다 바로 기록된다). 오류로 멈춰도 오류 내용이 파일에 남는다.
  - 안 붙이면 (`venv\Scripts\python run_simulation.py`): 출력이 cmd 창에만 나오고, 창을 닫으면 기록이 사라진다.
  - 같은 파일 이름으로 다시 실행하면 이전 로그를 덮어쓴다. 남기려면 이름을 바꾼다 (예: `run_log_2.txt`).
- 실행 중에는 **절전 모드를 끈다** (절전에 들어가면 계산이 멈춘다).
- MC의 b는 `N_WORKERS`(= 그 컴퓨터의 논리 코어 수)개씩 동시에 계산한다. 실행 중에는 CPU를 거의 모두 쓴다. 동시에 계산하는 수는 결과에 영향이 없다 (순서대로 하나씩 계산한 것과 비트 단위로 같음).
- 진행 상황은 로그의 `r=○/200 완료` 줄이나 `results/<scenario>/config.json`의 `completed_r`로 확인한다.

### 중간에 멈췄을 때

**같은 명령으로 다시 실행한다.** r 하나가 끝날 때마다 `r_results/`에 저장하므로, 완료된 r은 건너뛰고 이어서 계산한다 (잃는 것은 진행 중이던 r 하나).

- 다시 실행하면 저장된 설정과 지금 설정, 선택된 하이퍼파라미터 조합, σ̂²를 비교하고, 다르면 오류로 멈춘다. 기존 결과는 지우지 않는다.
- 요약 파일(`interval_metrics.csv`, `point_summary.csv` 등)은 **R개 r이 모두 끝난 뒤에만** 만들어진다. 미완료 상태에서는 `r_results/`와 `config.json`만 있다.
- **계산 코드를 바꾸면 해당 `results/` 폴더를 지우고 처음부터 다시 돌린다.** 재개 검사는 설정값만 비교하므로 코드 변경은 잡지 못한다.

## 4. full 소요 시간 측정 (본 실행 전, 컴퓨터마다)

full은 r 하나에 RF를 B = 200개 학습하므로, r 하나만 돌려 전체 시간을 추정한다.
b를 코어 수만큼 동시에 계산하므로 B를 줄여서 재면 안 된다 (b 20개는 한 번에 끝나 시간이 B에 비례하지 않는다). **B는 200 그대로 두고 R만 1로** 줄인다.

1. `run_simulation.py`를 임시로 고친다.
   ```python
   SCENARIOS = ['linear_homo_gaussian']
   RUN_MODE = 'full'
   ...
   elif RUN_MODE == 'full':
       R = 1        # 시간 측정용 (원래 200)
       B = 200
       N_TREES = 1000
   ```
2. 실행한다.
   ```
   venv\Scripts\python run_simulation.py > results\time_test.txt 2>&1
   ```
3. `results\time_test.txt`에서 두 값을 읽는다.
   - **T_r** = `r=1/1 완료 (○○초)`의 ○○ (full의 r 하나)
   - **T_base** = (r=1 완료 시각 − T_r) − 실행 시작 시각 (하이퍼파라미터 선택 24개 조합 + 기준 RF, 실행마다 한 번)
4. 계산한다.
   ```
   full scenario 하나 = T_base + 200 × T_r
   ```
   계산 예시 (실측값 아님): T_r = 20초, T_base = 60초 → scenario 하나 ≈ 4,060초 ≈ 1.1시간.
5. **되돌린다.**
   - 코드: `R = 200`, `SCENARIOS`, `RUN_MODE`를 원래대로
   - 폴더: `results\linear_homo_gaussian` 삭제 (R=1 설정이 저장되어 있어, 남아 있으면 본 실행이 설정 불일치로 멈춘다)

## 5. 결과 보기

- 여러 컴퓨터에서 돌렸다면 각 컴퓨터의 `results/<scenario>/` 폴더를 한 컴퓨터의 `results/` 아래로 모은다.
- `notebooks/simulation.ipynb`를 열어 맨 위 `SCENARIOS`에 볼 scenario를 모두 넣고 `RUN_MODE`를 맞춘 뒤, **메인 for문 셀("실행 — scenario for문")은 건너뛰고** 나머지 셀을 위에서부터 실행한다. "저장 결과 확인"과 "결과 표" 셀에 scenario별 coverage·평균 길이·MC와 CP 비교가 나온다.
- 요약 파일이 없는(미완료) scenario는 "요약 없음 → 건너뜀"으로 표시된다.

| 결과 파일 | 내용 |
| --- | --- |
| `config.json` | 설정, 하이퍼파라미터 선택 결과(24개 조합 RMSE 표, 선택 조합), σ̂², 진단값, 난수 정책, 라이브러리 버전, 컴퓨터 정보, 완료 r 목록 |
| `hparam_check.png` | 선택 조합의 $m(x)$ 대 $\widehat f(x)$ 그래프 ($y=x$ 꼴 확인용) |
| `r_results/interval_rXXX.csv` | r 하나의 채점 결과(100행) |
| `test_grid.csv` | grid 25개의 좌표, 참평균, 기준 RF 예측 |
| `interval_metrics.csv` | r × grid × alpha × method별 구간, 포함 여부, 길이 |
| `point_summary.csv` | grid × alpha × method별 coverage, 평균 길이, 길이 표준편차 |
| `method_comparison.csv` | grid × alpha별 MC·CP 나란히, CP − MC 차이 |
