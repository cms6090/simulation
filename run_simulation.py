# MC vs Split CP 시뮬레이션 — 실제 실행용 스크립트
#
# 계산 내용은 notebooks/simulation.ipynb와 같다 (설계: 시뮬.md, 구현 지침: CLAUDE.md).
# 실행 방법: 아래 SCENARIOS, RUN_MODE를 직접 고친 뒤, 프로젝트 폴더의 cmd 창에서
#     venv\Scripts\python run_simulation.py
# 중간에 멈추면 같은 명령으로 다시 실행한다 → 완료된 r은 건너뛰고 이어서 계산한다.
# 계산 코드를 바꾸면 results/의 기존 결과를 지우고 처음부터 다시 돌린다.

# ===== 이번에 실행할 scenario: 여기에 돌리고 싶은 것만 넣는다 =====
# 가능한 이름: 'linear_homo_gaussian', 'linear_homo_student_t', 'nonlinear_homo_gaussian', 'nonlinear_homo_student_t'
SCENARIOS = [
    'linear_homo_gaussian',
    # 'linear_homo_student_t',
    # 'nonlinear_homo_gaussian',
    # 'nonlinear_homo_student_t',
]

# ===== 실행 모드: 'smoke' (구조 검증용 작은 실행) 또는 'full' (설계값) =====
RUN_MODE = 'full'


import json
import os
import platform
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# 출력은 줄마다 바로 내보낸다 (로그 파일로 받을 때도 진행 상황이 바로 보이도록), 한글·기호는 utf-8로
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


# 이 파일이 있는 폴더 또는 현재 위치에서 위로 올라가며 시뮬.md가 있는 폴더를 찾는다 → 프로젝트 루트
PROJECT_ROOT = None
for start in [Path(__file__).resolve().parent, Path.cwd().resolve()]:
    for folder in [start] + list(start.parents):
        if (folder / '시뮬.md').exists():
            PROJECT_ROOT = folder
            break
    if PROJECT_ROOT is not None:
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError('시뮬.md를 찾지 못했습니다. 프로젝트 폴더 안에서 실행하세요.')

# ----- 실행 환경 확인: Python과 라이브러리 버전이 .python-version, requirements.txt와 같아야 한다 -----
# 컴퓨터마다 나눠 실행해도 같은 결과가 나오도록, 버전이 다르면 계산을 시작하지 않는다
version_errors = []
expected_python = (PROJECT_ROOT / '.python-version').read_text(encoding='utf-8').strip()
current_python = sys.version.split()[0]
if current_python != expected_python:
    version_errors.append(f'  python: 필요 = {expected_python}, 지금 = {current_python}')

for line in (PROJECT_ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if line == '':
        continue
    name, expected_version = line.split('==')
    try:
        installed_version = version(name)
    except Exception:
        installed_version = '설치 안 됨'
    if installed_version != expected_version:
        version_errors.append(f'  {name}: 필요 = {expected_version}, 지금 = {installed_version}')

if version_errors:
    raise RuntimeError('실행 환경이 고정 버전과 다름 → README의 환경 만들기 순서대로 venv를 다시 만드세요\n'
                       + '\n'.join(version_errors))

# OOB 예측이 없는 관측치가 생기면 경고 대신 오류로 멈춘다 (sklearn은 0으로 채우고 경고만 냄)
warnings.filterwarnings('error', message='Some inputs do not have OOB scores')
# RF를 n_jobs=-1로 병렬 학습할 때 나오는 안내성 경고 (결과와 무관) → 끈다
warnings.filterwarnings('ignore', message='`sklearn.utils.parallel.delayed` should be used')

print(now(), '시작')
print('PROJECT_ROOT :', PROJECT_ROOT)
print('python       :', current_python, '(고정 버전과 일치)')
print('numpy        :', np.__version__)
print('pandas       :', pd.__version__)
print('scikit-learn :', sklearn.__version__)


# ----- 확정 설계값 (시뮬.md) -----
N_TRAIN = 1000
N_CAL = 1000
GRID_AXIS = [-1.0, -0.5, 0.0, 0.5, 1.0]
ALPHAS = [0.10, 0.05]

# ----- 구현 기본값 (시뮬.md에 수치가 없어 정한 변경 가능한 제안값) -----
MASTER_SEED = 0
MC_QUANTILE_METHOD = 'linear'
CV_FOLDS = 10
MAX_DEPTH = None
# MC의 b를 동시에 몇 개 계산할지 (이 컴퓨터의 논리 코어 수). 결과에는 영향이 없고 속도만 바뀐다.
# 데이터가 1,000행으로 작아 RF 하나를 여러 스레드로 나누는 것보다 b 여러 개를 동시에 돌리는 것이 훨씬 빠르다
N_WORKERS = os.cpu_count()
# OOB grid search 후보 → 4 x 2 x 3 = 24개 조합
GRID_MIN_SAMPLES_LEAF = [5, 10, 20, 40]
GRID_MAX_FEATURES = [0.5, 1.0]
GRID_MAX_SAMPLES = [0.5, 0.7, 1.0]

# ----- 실행 모드별 규모 -----
if RUN_MODE == 'smoke':
    R = 2
    B = 5
    N_TREES = 30     # RF 하나 안의 결정트리 수
    RESULTS_DIR = PROJECT_ROOT / 'results' / 'smoke'
elif RUN_MODE == 'full':
    R = 200
    # R = 1 # test
    B = 200
    N_TREES = 1000
    RESULTS_DIR = PROJECT_ROOT / 'results'
else:
    raise ValueError("RUN_MODE는 'smoke' 또는 'full'")

print('RUN_MODE      :', RUN_MODE)
print('R, B, 트리 수 :', R, B, N_TREES)
print('SCENARIOS     :', SCENARIOS)
print('RESULTS_DIR   :', RESULTS_DIR)
print('N_WORKERS     :', N_WORKERS, '(MC의 b를 동시에 계산하는 수)')


# 난수 키 = [MASTER_SEED, scenario_id, stage_id, r, b, fold]
# r=1..R, b=1..B, fold=1..CV_FOLDS, 해당 단계에 필요 없는 인덱스는 0
# scenario_id는 고정이라 SCENARIOS에 일부만 넣거나 순서를 바꿔도 난수는 같다
SCENARIO_IDS = {
    'linear_homo_gaussian': 0,
    'linear_homo_student_t': 1,
    'nonlinear_homo_gaussian': 2,
    'nonlinear_homo_student_t': 3,
}
STAGE_IDS = {
    'common_dataset': 0,
    'common_training': 1,
    'cv_split': 2,
    'cv_model': 3,
    'cp_calibration': 4,
    'mc_dataset': 5,
    'mc_training': 6,
    'mc_new_error': 7,
    'test_error': 8,
}


def make_random_state(scenario, stage, r=0, b=0, fold=0):
    # 키 → SeedSequence → 정수 4개 → RandomState. 부를 때마다 새 객체를 만든다.
    key = [MASTER_SEED, SCENARIO_IDS[scenario], STAGE_IDS[stage], r, b, fold]
    words = np.random.SeedSequence(key).generate_state(4)
    return np.random.RandomState(words)


# SCENARIOS에 잘못된 이름이 있으면 여기서 멈춘다
for scenario in SCENARIOS:
    if scenario not in SCENARIO_IDS:
        raise ValueError(f'알 수 없는 scenario: {scenario}')


def mean_function(scenario, X):
    # 참평균 m(x). linear: x1 + x2, nonlinear: sqrt(2/3) sin(pi x1) + x2
    if scenario.startswith('linear'):
        return X[:, 0] + X[:, 1]
    elif scenario.startswith('nonlinear'):
        return np.sqrt(2 / 3) * np.sin(np.pi * X[:, 0]) + X[:, 1]
    else:
        raise ValueError(scenario)


def draw_error(scenario, random_seed, size):
    # 참오차 ε (평균 0, 분산 1). gaussian: N(0,1), student_t: t_3 / sqrt(3)
    if scenario.endswith('gaussian'):
        return random_seed.normal(size=size)
    elif scenario.endswith('student_t'):
        return random_seed.standard_t(3, size=size) / np.sqrt(3)
    else:
        raise ValueError(scenario)


# grid 25개: 바깥 루프 x1, 안쪽 루프 x2, grid_id 0~24로 고정
grid_points = []
for x1 in GRID_AXIS:
    for x2 in GRID_AXIS:
        grid_points.append([x1, x2])
X_grid = np.array(grid_points)
N_GRID = len(X_grid)


def fit_rf_oob(X, y, scenario, stage, r=0, b=0, fold=0, n_jobs=-1):
    # Fit = OOB grid search + 학습. 선택된 RF, 선택 조합, 조합별 OOB MSE 표를 돌려준다.
    # n_jobs: RF 하나의 학습 스레드 수. 기준 RF는 -1(모든 코어), MC는 b를 동시에 돌리므로 1.
    #         RF 결과는 random_state로 정해지므로 n_jobs와 상관없이 같은 모델이 나온다.
    rows = []
    best_model = None
    best_mse = np.inf
    best_params = None

    for leaf in GRID_MIN_SAMPLES_LEAF:
        for feat in GRID_MAX_FEATURES:
            for samp in GRID_MAX_SAMPLES:
                rf = RandomForestRegressor(
                    n_estimators=N_TREES,
                    min_samples_leaf=leaf,
                    max_features=feat,
                    max_samples=samp,
                    max_depth=MAX_DEPTH,
                    bootstrap=True,
                    oob_score=True,
                    n_jobs=n_jobs,
                    random_state=make_random_state(scenario, stage, r=r, b=b, fold=fold),
                )
                rf.fit(X, y)

                oob_mse = np.mean((y - rf.oob_prediction_) ** 2)
                rows.append([leaf, feat, samp, oob_mse])

                if oob_mse < best_mse:   # 같으면 교체하지 않음 → 동률이면 grid 순서상 먼저
                    best_mse = oob_mse
                    best_model = rf
                    best_params = (leaf, feat, samp)

    # 예측은 단일 스레드로: n_jobs=-1로 예측하면 여러 스레드가 트리 예측을 더하는 순서가 실행마다 달라
    # 끝자리(1e-15 수준)가 흔들린다. 학습은 병렬 그대로, 예측만 n_jobs=1로 바꿔 같은 키 → 비트 단위로 같은 결과
    best_model.set_params(n_jobs=1)

    oob_table = pd.DataFrame(rows, columns=['min_samples_leaf', 'max_features', 'max_samples', 'oob_mse'])
    return best_model, best_params, oob_table


def run_one_b(scenario, r, b, best_train_pred, sigma_hat, X_train):
    # MC의 b 하나: 가상 Y* → OOB 튜닝 + 학습 → grid 25개 예측 + 새 오차 (9-3과 같은 계산)
    # 별도 프로세스에서 실행되므로 경고 설정을 여기서 다시 한다 (OOB 예측 누락 → 오류)
    warnings.filterwarnings('error', message='Some inputs do not have OOB scores')
    warnings.filterwarnings('ignore', message='`sklearn.utils.parallel.delayed` should be used')

    random_seed = make_random_state(scenario, "mc_dataset", r = r, b = b)
    mc_Y = best_train_pred + random_seed.normal(0, sigma_hat, N_TRAIN)

    mc_model, mc_params, mc_table = fit_rf_oob(X_train, mc_Y, scenario, "mc_training", r = r, b = b, n_jobs = 1)
    selection_row = [scenario, "mc", r, b, mc_params[0], mc_params[1], mc_params[2], mc_table["oob_mse"].min()]

    random_seed = make_random_state(scenario, "mc_new_error", r = r, b = b)
    mc_row = mc_model.predict(X_grid) + random_seed.normal(0, sigma_hat, N_GRID)
    return mc_row, selection_row


def run_one_r(scenario, r, best_model, best_train_pred, grid_predict, sigma_hat, X_train):
    # 한 번의 반복 r: 평가 Y → CP 구간 → MC 구간 → 채점. (MC_CP_simulation.ipynb 10절과 같은 함수)
    # 돌려주는 것: (채점 결과 100행 = grid 25 × alpha 2 × method 2, MC 선택 조합 B행)
    # 고정값(best_model, best_train_pred, grid_predict, sigma_hat, X_train)은 인자로 받는다.

    # 1. 평가 반응값 (9-1)
    random_seed = make_random_state(scenario, "test_error", r = r)
    true_mean = mean_function(scenario, X_grid)
    Y_test = true_mean + draw_error(scenario, random_seed, N_GRID)

    # 2. CP (9-2): 새 calibration → score → 순위 분위수
    random_seed = make_random_state(scenario, "cp_calibration", r = r)
    CP_X = random_seed.uniform(-1, 1, size=(N_CAL, 2))
    CP_Y = mean_function(scenario, CP_X) + draw_error(scenario, random_seed, N_CAL)
    s = sorted(np.abs(CP_Y - best_model.predict(CP_X)))

    cp_lower = {}
    cp_upper = {}
    for alpha in ALPHAS:
        k_alpha = np.ceil((N_CAL + 1) * (1 - alpha))
        if k_alpha > N_CAL:
            q_alpha = np.inf
        else:
            q_alpha = s[int(k_alpha) - 1]
        cp_lower[alpha] = grid_predict - q_alpha
        cp_upper[alpha] = grid_predict + q_alpha

    # 3. MC (9-3, 9-4): b = 1..B 가상 데이터 → OOB 튜닝 + 학습 → grid 예측 + 새 오차
    #    b = 1..B를 N_WORKERS개씩 동시에 계산한다. 결과는 b 순서대로 돌아오고, 난수 키가 b로 정해지므로
    #    순서대로 하나씩 계산한 것과 비트 단위로 같다.
    #    넘기는 배열이 작아(1,000행) 임시 파일 공유(memmap)는 끈다 (max_nbytes=None)
    b_results = Parallel(n_jobs=N_WORKERS, backend='loky', max_nbytes=None)(
        delayed(run_one_b)(scenario, r, b, best_train_pred, sigma_hat, X_train) for b in range(1, B + 1)
    )

    mc_samples = np.zeros((B, N_GRID))   # 행 = b, 열 = grid
    selection_rows = []                  # (r, b)마다 OOB 튜닝으로 선택된 조합 → model_selection.csv
    for b in range(1, B + 1):
        mc_row, selection_row = b_results[b - 1]
        mc_samples[b - 1] = mc_row
        selection_rows.append(selection_row)

    mc_lower = {}
    mc_upper = {}
    for alpha in ALPHAS:
        mc_lower[alpha] = np.quantile(mc_samples, alpha / 2, axis=0, method=MC_QUANTILE_METHOD)
        mc_upper[alpha] = np.quantile(mc_samples, 1 - alpha / 2, axis=0, method=MC_QUANTILE_METHOD)

    # 4. 채점 (9-5): 같은 Y_test로 두 방법·두 alpha
    tables = []
    for alpha in ALPHAS:
        for method in ["mc", "cp"]:
            if method == "mc":
                lower = mc_lower[alpha]
                upper = mc_upper[alpha]
            else:
                lower = cp_lower[alpha]
                upper = cp_upper[alpha]

            covered = (lower <= Y_test) & (Y_test <= upper)   # 25개 True/False (경계 포함)
            tables.append(pd.DataFrame({
                "scenario" : scenario,
                "r" : r,
                "grid_id" : np.arange(N_GRID),
                "x1" : X_grid[:, 0],
                "x2" : X_grid[:, 1],
                "alpha" : alpha,
                "method" : method,
                "y_true" : Y_test,
                "true_mean" : true_mean,
                "lower" : lower,
                "upper" : upper,
                "covered" : covered.astype(int),
                "length" : upper - lower,
            }))

    r_selection = pd.DataFrame(selection_rows, columns=[
        "scenario", "stage", "r", "b", "min_samples_leaf", "max_features", "max_samples", "oob_mse",
    ])
    return pd.concat(tables, ignore_index=True), r_selection


def write_csv_atomic(df, path):
    # 임시 파일에 쓴 뒤 이름을 바꾼다
    tmp_path = path.with_name(path.name + '.tmp')
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def write_json_atomic(obj, path):
    tmp_path = path.with_name(path.name + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as file:
        # numpy 숫자(np.int64, np.float64)는 .item()으로 파이썬 숫자로 바꿔 저장
        json.dump(obj, file, ensure_ascii=False, indent=2, default=lambda value: value.item())
    os.replace(tmp_path, path)


def aggregate_saved_results(save_dir):
    # 학습 없이 저장된 r 파일을 읽어 요약을 만든다. 모든 r(R개)이 완료된 경우에만 집계한다 (미완료 요약은 만들지 않음).
    save_dir = Path(save_dir)
    config_path = save_dir / 'config.json'
    if not config_path.exists():
        print(f'{save_dir}: config.json 없음 → 건너뜀')
        return False
    with open(config_path, encoding='utf-8') as file:
        config = json.load(file)
    scenario = config['scenario']
    settings = config['run_settings']
    R, B = settings['R'], settings['B']
    ALPHAS = settings['alphas']
    N_GRID = len(settings['grid_axis']) ** 2
    params = config['base_rf']['best_params']
    best_params = (params['min_samples_leaf'], params['max_features'], params['max_samples'])
    oob_table = pd.DataFrame(config['base_rf']['oob_table'])
    r_dir = save_dir / 'r_results'
    completed_r = [r for r in range(1, R + 1)
                   if (r_dir / f'interval_r{r:03d}.csv').exists()
                   and (r_dir / f'selection_r{r:03d}.csv').exists()]
    if len(completed_r) != R:
        print(f'{scenario}: 완료 r {len(completed_r)} / {R} → 미완료라 요약을 만들지 않음')
        return False
    # ----- 10절. 집계: 저장된 r 파일을 모두 읽어 모은다 (메모리 값이 아니라 파일 기준) -----
    r_tables = []
    selection_tables = []
    for r in sorted(completed_r):
        r_result = pd.read_csv(r_dir / f'interval_r{r:03d}.csv', float_precision='round_trip')
        r_selection = pd.read_csv(r_dir / f'selection_r{r:03d}.csv', float_precision='round_trip')
        if len(r_result) != N_GRID * len(ALPHAS) * 2 or set(r_result['r']) != {r} or set(r_result['scenario']) != {scenario}:
            raise ValueError(f'interval_r{r:03d}.csv: 행 수 또는 r·scenario 값이 맞지 않음')
        if len(r_selection) != B or set(r_selection['r']) != {r} or set(r_selection['scenario']) != {scenario}:
            raise ValueError(f'selection_r{r:03d}.csv: 행 수 또는 r·scenario 값이 맞지 않음')
        expected_keys = {(i, a, h) for i in range(N_GRID) for a in ALPHAS for h in ['mc', 'cp']}
        actual_keys = set(r_result[['grid_id', 'alpha', 'method']].itertuples(index=False, name=None))
        if actual_keys != expected_keys:
            raise ValueError(f'r={r}: grid·alpha·method 구성 불일치')
        if set(r_selection['b']) != set(range(1, B + 1)) or set(r_selection['stage']) != {'mc'}:
            raise ValueError(f'r={r}: MC 선택 모델 번호 불일치')
        if not np.isfinite(r_result[['y_true', 'true_mean', 'lower', 'upper', 'length']].to_numpy()).all():
            raise ValueError(f'r={r}: 비유한 결과값 (현재 alpha·n_cal 설정은 유한 구간이어야 함)')
        if not r_result['covered'].isin([0, 1]).all():
            raise ValueError(f'r={r}: covered는 0 또는 1이어야 함')
        expected_covered = ((r_result['lower'] <= r_result['y_true']) &
                            (r_result['y_true'] <= r_result['upper'])).astype(int)
        if not np.array_equal(expected_covered, r_result['covered']):
            raise ValueError(f'r={r}: covered 불일치')
        if (r_result['upper'] < r_result['lower']).any() or not np.allclose(
                r_result['length'], r_result['upper'] - r_result['lower'], rtol=1e-12, atol=1e-12):
            raise ValueError(f'r={r}: 구간 길이 불일치')
        if (r_result.groupby('grid_id')['y_true'].nunique() != 1).any():
            raise ValueError(f'r={r}: 방법·alpha별 평가 Y 공유 불일치')
        r_tables.append(r_result)
        selection_tables.append(r_selection)

    interval_metrics = pd.concat(r_tables, ignore_index=True)

    # 같은 r·grid·alpha 안에서 mc, cp가 번갈아 나오도록 정렬 (alpha는 0.10 → 0.05)
    interval_metrics["method_order"] = interval_metrics["method"].map({"mc": 0, "cp": 1})
    interval_metrics = interval_metrics.sort_values(
        ["r", "grid_id", "alpha", "method_order"], ascending=[True, True, False, True]
    ).drop(columns="method_order").reset_index(drop=True)

    n_duplicated = interval_metrics.duplicated(['scenario', 'r', 'grid_id', 'alpha', 'method']).sum()
    if n_duplicated > 0:
        raise ValueError(f'interval_metrics key 중복 {n_duplicated}개')

    # grid마다 반복 방향(r)으로 평균 → conditional coverage와 평균 길이
    point_summary = interval_metrics.groupby(["scenario", "grid_id", "x1", "x2", "alpha", "method"], as_index=False).agg(
        n_covered = ("covered", "sum"),
        n_evaluated = ("covered", "count"),
        coverage = ("covered", "mean"),
        length_mean = ("length", "mean"),
        length_sd = ("length", "std"),
    )
    point_summary["method_order"] = point_summary["method"].map({"mc": 0, "cp": 1})
    point_summary = point_summary.sort_values(
        ["grid_id", "alpha", "method_order"], ascending=[True, False, True]
    ).drop(columns="method_order").reset_index(drop=True)

    # ----- 11절. 저장 -----
    # model_selection.csv : 기준 RF 1행 (r = b = 0) + MC 선택 조합
    base_selection = pd.DataFrame([[
        scenario, "base", 0, 0, best_params[0], best_params[1], best_params[2], oob_table["oob_mse"].min()
    ]], columns=["scenario", "stage", "r", "b", "min_samples_leaf", "max_features", "max_samples", "oob_mse"])
    model_selection = pd.concat([base_selection] + selection_tables, ignore_index=True)

    # method_comparison.csv : grid × alpha마다 MC·CP를 한 행에 나란히 놓고 CP − MC 차이 계산
    mc_part = point_summary[point_summary["method"] == "mc"]
    cp_part = point_summary[point_summary["method"] == "cp"]
    mc_part = mc_part[["scenario", "grid_id", "x1", "x2", "alpha", "coverage", "length_mean"]].rename(
        columns={"coverage": "mc_coverage", "length_mean": "mc_length_mean"})
    cp_part = cp_part[["scenario", "grid_id", "alpha", "coverage", "length_mean"]].rename(
        columns={"coverage": "cp_coverage", "length_mean": "cp_length_mean"})

    method_comparison = mc_part.merge(cp_part, on=["scenario", "grid_id", "alpha"])
    method_comparison["coverage_cp_minus_mc"] = method_comparison["cp_coverage"] - method_comparison["mc_coverage"]
    method_comparison["length_cp_minus_mc"] = method_comparison["cp_length_mean"] - method_comparison["mc_length_mean"]
    method_comparison = method_comparison[[
        "scenario", "grid_id", "x1", "x2", "alpha",
        "mc_coverage", "cp_coverage", "coverage_cp_minus_mc",
        "mc_length_mean", "cp_length_mean", "length_cp_minus_mc",
    ]]

    write_csv_atomic(interval_metrics, save_dir / "interval_metrics.csv")
    write_csv_atomic(model_selection, save_dir / "model_selection.csv")
    write_csv_atomic(point_summary, save_dir / "point_summary.csv")
    write_csv_atomic(method_comparison, save_dir / "method_comparison.csv")
    config['completed_r'] = sorted(completed_r)
    config['is_complete'] = len(completed_r) == R
    write_json_atomic(config, config_path)

    print(f'  저장: interval_metrics {len(interval_metrics)}행, model_selection {len(model_selection)}행, '
          f'point_summary {len(point_summary)}행, method_comparison {len(method_comparison)}행')
    return True


# [재개] 저장된 결과와 반드시 같아야 하는 설정
run_settings = {
    'run_mode': RUN_MODE,
    'R': R,
    'B': B,
    'n_trees': N_TREES,
    'n_train': N_TRAIN,
    'n_cal': N_CAL,
    'grid_axis': GRID_AXIS,
    'alphas': ALPHAS,
    'master_seed': MASTER_SEED,
    'scenario_ids': SCENARIO_IDS,
    'stage_ids': STAGE_IDS,
    'cv_folds': CV_FOLDS,
    'mc_quantile_method': MC_QUANTILE_METHOD,
    'max_depth': MAX_DEPTH,
    'grid_min_samples_leaf': GRID_MIN_SAMPLES_LEAF,
    'grid_max_features': GRID_MAX_FEATURES,
    'grid_max_samples': GRID_MAX_SAMPLES,
    'python': sys.version.split()[0],
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'scikit-learn': sklearn.__version__,
}
run_settings = json.loads(json.dumps(run_settings))   # config.json에서 읽은 값과 같은 형태로 맞춘다 (비교용)


# ===================== 실행 — scenario for문 =====================
total_start = time.time()

for scenario in SCENARIOS:
    print('=' * 70)
    print(f'{now()}  {scenario}  ({RUN_MODE}, R={R}, B={B}, 트리 {N_TREES}개)')
    scenario_start = time.time()

    save_dir = RESULTS_DIR / scenario
    r_dir = save_dir / 'r_results'
    r_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / 'config.json'

    # ----- 6절. training 데이터 생성: 생성기 하나에서 X → 오차 순서로 -----
    random_seed = make_random_state(scenario, 'common_dataset')
    X_train = random_seed.uniform(-1, 1, size=(N_TRAIN, 2))
    eps_train = draw_error(scenario, random_seed, N_TRAIN)
    Y_train = mean_function(scenario, X_train) + eps_train

    # ----- 7절. 기준 RF: OOB grid search -----
    best_model, best_params, oob_table = fit_rf_oob(X_train, Y_train, scenario, 'common_training')
    print('  기준 RF 조합 (min_samples_leaf, max_features, max_samples) :', best_params)

    # ----- 8절. CV 잔차분산: best_params 고정, fold마다 RF 하나만 학습 -----
    leaf, feat, samp = best_params
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=make_random_state(scenario, 'cv_split'))

    oof_pred = np.full(N_TRAIN, np.nan)        # 각 관측치의 OOF 예측
    oof_count = np.zeros(N_TRAIN, dtype=int)   # 각 관측치가 몇 번 예측됐는지

    fold = 0
    for train_idx, valid_idx in kfold.split(X_train):
        fold += 1   # fold는 1부터
        fold_model = RandomForestRegressor(
            n_estimators=N_TREES,
            min_samples_leaf=leaf,
            max_features=feat,
            max_samples=samp,
            max_depth=MAX_DEPTH,
            bootstrap=True,
            n_jobs=-1,
            random_state=make_random_state(scenario, 'cv_model', fold=fold),
        )
        fold_model.fit(X_train[train_idx], Y_train[train_idx])
        fold_model.set_params(n_jobs=1)   # 예측은 단일 스레드 (합산 순서 고정 → 재현)
        oof_pred[valid_idx] = fold_model.predict(X_train[valid_idx])
        oof_count[valid_idx] += 1

    assert (oof_count == 1).all()   # 모든 관측치가 자기가 빠진 fold 모델로 정확히 한 번 예측됨

    cv_residual = Y_train - oof_pred
    # 전체 training에서 조합을 선택했으므로 선택 편향과 평균함수 추정오차가 포함될 수 있음
    # 영향의 크기는 확인하지 않았으며 순수 오차분산의 불편추정량으로 해석하지 않음
    sigma2_hat = np.mean(cv_residual ** 2)     # 제곱평균 (np.var 아님)
    if not np.isfinite(sigma2_hat):
        raise ValueError('sigma2_hat이 유한하지 않음')
    if sigma2_hat == 0:
        raise ValueError('sigma2_hat = 0 (퇴화)')
    sigma_hat = np.sqrt(sigma2_hat)

    sigma2_oob = np.mean((Y_train - best_model.oob_prediction_) ** 2)   # 진단용 (MC에는 안 씀)

    # ----- 진단과 고정 예측값: grid에서 기준 RF 예측 vs 참평균 -----
    grid_predict = best_model.predict(X_grid)        # CP 구간의 중심 f̂(x_i)
    best_train_pred = best_model.predict(X_train)    # MC 가상 반응값의 중심 f̂(X_train)
    grid_true_mean = mean_function(scenario, X_grid)
    grid_error = grid_predict - grid_true_mean
    grid_rmse = np.sqrt(np.mean(grid_error ** 2))
    grid_mae = np.mean(np.abs(grid_error))

    print(f'  sigma2_hat (CV, MC에 사용) : {sigma2_hat:.6f}   sigma_hat : {sigma_hat:.6f}')
    print(f'  sigma2_oob (진단용)        : {sigma2_oob:.6f}')
    print(f'  grid RMSE / MAE (f̂ vs m)   : {grid_rmse:.6f} / {grid_mae:.6f}')

    # ===== [재개] 이전 실행이 있으면 설정·기준 RF 조합·sigma2_hat이 같은지 확인 =====
    if config_path.exists():
        with open(config_path, encoding='utf-8') as file:
            saved_config = json.load(file)

        saved_settings = saved_config.get('run_settings')
        if saved_settings != run_settings:
            diffs = []
            if saved_settings is None:
                diffs.append('  저장된 config.json에 run_settings가 없음')
            else:
                for key in sorted(set(saved_settings) | set(run_settings)):
                    if saved_settings.get(key) != run_settings.get(key):
                        diffs.append(f'  {key}: 저장 = {saved_settings.get(key)}, 지금 = {run_settings.get(key)}')
            raise ValueError('저장된 설정과 지금 설정이 다름 → 이어서 계산하지 않음\n' + '\n'.join(diffs))

        saved_params = saved_config['base_rf']['best_params']
        saved_params = (saved_params['min_samples_leaf'], saved_params['max_features'], saved_params['max_samples'])
        if saved_params != best_params:
            raise ValueError(f'기준 RF 조합이 다름: 저장 = {saved_params}, 지금 = {best_params}')
        if saved_config['sigma2_hat'] != float(sigma2_hat):
            raise ValueError(f'sigma2_hat이 다름: 저장 = {saved_config["sigma2_hat"]}, 지금 = {sigma2_hat}')
        print('  [재개] 저장된 설정·기준 RF 조합·sigma2_hat 일치 → 완료된 r은 건너뜀')
    elif any(r_dir.iterdir()):
        raise ValueError(f'{r_dir}에 파일이 있는데 config.json이 없음 → 이어서 계산하지 않음')

    # ----- 11절. config.json: r 반복 전에 만들어 두고, r마다 완료 목록만 갱신 -----
    config = {}
    config['design_doc'] = '시뮬.md'
    config['scenario'] = scenario
    config['run_mode'] = RUN_MODE
    if RUN_MODE == 'smoke':
        config['note'] = 'smoke: 구조 검증용 작은 실행 (연구 결과 아님)'
    config['run_settings'] = run_settings

    # 확정 설계값 (시뮬.md)
    config['design'] = {
        'n_train': N_TRAIN,
        'n_cal': N_CAL,
        'grid_axis': GRID_AXIS,
        'n_grid': N_GRID,
        'alphas': ALPHAS,
        'R_full': 200,
        'B_full': 200,
        'variance_estimator': 'cv',
        'mc_error_family': 'gaussian',
    }
    # 이번 실행 규모 (smoke면 줄인 값)
    config['run_size'] = {'R': R, 'B': B, 'n_trees': N_TREES}
    # 구현 기본값 (변경 가능한 제안값)
    config['implementation_defaults'] = {
        'master_seed': MASTER_SEED,
        'cv_folds': CV_FOLDS,
        'mc_quantile_method': MC_QUANTILE_METHOD,
        'n_trees_full': 1000,
        'rf_fixed': {'max_depth': MAX_DEPTH, 'bootstrap': True, 'oob_score': True, 'predict_n_jobs': 1},
        'rf_train_n_jobs': '기준 RF·CV fold RF는 -1, MC 재학습 RF는 1 (b를 N_WORKERS개 동시에 계산). RF 결과는 n_jobs와 무관',
        'n_workers': N_WORKERS,
        'rf_param_grid': {
            'min_samples_leaf': GRID_MIN_SAMPLES_LEAF,
            'max_features': GRID_MAX_FEATURES,
            'max_samples': GRID_MAX_SAMPLES,
        },
        'selection_rule': 'OOB MSE = mean((y - oob_prediction_)^2) 최소, 동률이면 grid 순서상 먼저 나온 조합',
    }
    # 확정 방식 조합
    config['method_choice'] = {
        'base_rf_tuning': 'OOB',
        'sigma2_estimation': 'CV, 기준 RF 조합 고정 (fold 안에서 다시 튜닝하지 않음)',
        'mc_rf_tuning': 'OOB (가상 데이터마다 다시 튜닝)',
    }
    # 기준 RF와 분산 추정 결과 (CV fold RF도 best_params를 그대로 씀)
    config['base_rf'] = {
        'best_params': {
            'min_samples_leaf': best_params[0],
            'max_features': best_params[1],
            'max_samples': best_params[2],
        },
        'oob_table': oob_table.to_dict(orient='records'),
    }
    config['sigma2_hat'] = float(sigma2_hat)
    config['sigma2_oob_diagnostic'] = float(sigma2_oob)
    config['diagnostics'] = {
        'pooled_cv_rmse': float(sigma_hat),
        'grid_rmse_base_vs_true_mean': float(grid_rmse),
        'grid_mae_base_vs_true_mean': float(grid_mae),
    }
    # 난수 정책
    config['rng'] = {
        'master_seed': MASTER_SEED,
        'scenario_ids': SCENARIO_IDS,
        'stage_ids': STAGE_IDS,
        'key_order': ['MASTER_SEED', 'scenario_id', 'stage_id', 'r', 'b', 'fold'],
        'index_rules': 'r=1..R, b=1..B, fold=1..CV_FOLDS; 해당 단계에 필요 없는 인덱스는 0',
        'generator': 'np.random.RandomState(np.random.SeedSequence(key).generate_state(4)); 부를 때마다 새 객체',
    }
    config['index_ranges'] = 'r=1..R, grid_id=0..24 (x1 바깥 루프, x2 안쪽 루프), method=mc/cp'
    # 실행 환경
    config['library_versions'] = {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit-learn': sklearn.__version__,
    }
    config['machine'] = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
    }

    # [재개] 지금까지 완료된 r (두 파일이 모두 있는 r)
    completed_r = []
    for r in range(1, R + 1):
        if (r_dir / f'interval_r{r:03d}.csv').exists() and (r_dir / f'selection_r{r:03d}.csv').exists():
            completed_r.append(r)
    config['completed_r'] = completed_r
    config['is_complete'] = len(completed_r) == R
    write_json_atomic(config, config_path)
    print(f'  이미 완료된 r : {len(completed_r)} / {R}')

    # test_grid.csv : grid 25개의 좌표·참평균·기준 RF 예측
    test_grid = pd.DataFrame({
        "scenario" : scenario,
        "grid_id" : np.arange(N_GRID),
        "x1" : X_grid[:, 0],
        "x2" : X_grid[:, 1],
        "true_mean" : grid_true_mean,
        "base_prediction" : grid_predict,
    })

    write_csv_atomic(test_grid, save_dir / "test_grid.csv")

    # ----- 10절. r 반복 -----
    for r in range(1, R + 1):
        interval_path = r_dir / f'interval_r{r:03d}.csv'
        selection_path = r_dir / f'selection_r{r:03d}.csv'

        # [재개] 이미 완료된 r은 건너뛴다
        if r in completed_r:
            continue

        r_start = time.time()
        r_result, r_selection = run_one_r(scenario, r, best_model, best_train_pred, grid_predict, sigma_hat, X_train)

        # [재개] r 결과 저장 (selection → interval 순서) 후 완료 목록 갱신
        write_csv_atomic(r_selection, selection_path)
        write_csv_atomic(r_result, interval_path)
        completed_r.append(r)
        config['completed_r'] = sorted(completed_r)
        config['is_complete'] = len(completed_r) == R
        write_json_atomic(config, config_path)

        print(f'  {now()}  r={r}/{R} 완료 ({time.time() - r_start:.1f}초)  완료 {len(completed_r)}/{R}')

    aggregate_saved_results(save_dir)
    print(f'  {now()}  {scenario} 끝 ({time.time() - scenario_start:.1f}초)')

print('=' * 70)
print(f'{now()}  전체 {time.time() - total_start:.1f}초')

# 결과 표는 notebooks/simulation.ipynb의 "저장 결과 확인", "결과 표" 셀에서 본다
for scenario in SCENARIOS:
    with open(RESULTS_DIR / scenario / 'config.json', encoding='utf-8') as file:
        saved_config = json.load(file)
    print(f"  {scenario}: 완료 r {len(saved_config['completed_r'])} / {R}, is_complete = {saved_config['is_complete']}")
