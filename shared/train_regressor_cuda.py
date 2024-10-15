import cudf as pd
import cupy as np
from pathlib import Path
import itertools, sys, pickle, warnings
from joblib import Parallel, delayed
from cuml.preprocessing import StandardScaler, MinMaxScaler
from cuml.experimental.preprocessing import KNNImputer
from cuml.linear_model import Ridge as RR
from cuml.svm import SVR
from cuml.neighbors import KNeighborsRegressor as KNN
from cuml.linear_model import Lasso
import tqdm
from cuml.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# Adjust the system path if necessary
sys.path.append(str(Path(Path.home(), 'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(), 'gonza', 'scripts_generales')))

from utils import *
from expected_cost.ec import *
from psrcal import *

project_name = 'GeroApathy'
tasks = ['Fugu']
id_col = 'id'

scaler_name = 'StandardScaler'
imputer = KNNImputer()

l2ocv = False
n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

hyp_opt_list = [True]
feature_selection_list = [True]
bootstrap_list = [True]

boot_test = 100
boot_train = 0

n_seeds_test = 1
scaler = StandardScaler() if scaler_name == 'StandardScaler' else MinMaxScaler()

models_dict = {
    'ridge': RR,
    'lasso': Lasso,
    'knn': KNN,
    'svm': SVR,
}

data_dir = Path(Path.home(), 'data', project_name) if 'Users/gp' in str(Path.home()) else Path('D:', 'CNC_Audio', 'gonza', 'data', project_name)
results_dir = Path(str(data_dir).replace('data', 'results'))

metrics_names = ['r2_score', 'mean_squared_error', 'mean_absolute_error']
random_seeds_test = [0]
n_models = 10

# Load data using pandas and convert to cuDF DataFrame
neuro_data = pd.from_pandas(pd.read_csv(Path(data_dir, 'nps_data_filtered_no_missing.csv')))
dimensions = ['pitch', 'mfcc', 'voice-quality']

sort_by = 'mean_absolute_error'
extremo = 'sup' if 'error' in sort_by else 'inf'
ascending = True if 'error' in sort_by else False

data_features = pd.from_pandas(pd.read_csv(Path(data_dir, 'features_data.csv')))

for task in tasks:
    for dimension in dimensions:
        path = Path(
            results_dir,
            task,
            dimension,
            scaler_name,
            kfold_folder,
            f'{n_seeds_train}_seeds_train',
            f'{n_seeds_test}_seeds_test' if len(random_seeds_test) > 0 else '',
        )

        y_labels = [folder.name for folder in path.iterdir() if folder.is_dir()]
        for y_label, hyp_opt, feature_selection, bootstrap in itertools.product(y_labels, hyp_opt_list, feature_selection_list, bootstrap_list):
            print(task, dimension, y_label)

            # Merge data
            data = data_features.merge(neuro_data, on=id_col, how='inner')
            data = data.dropna(subset=[y_label])
            data = data.reset_index(drop=True)

            path_to_results = Path(path, y_label, 'hyp_opt', 'feature_selection', 'bootstrap')
            path_to_results = Path(str(path_to_results).replace('no_hyp_opt', 'hyp_opt')) if hyp_opt else path_to_results
            path_to_results = Path(str(path_to_results).replace('feature_selection', '')) if not feature_selection else path_to_results
            path_to_results = Path(str(path_to_results).replace('bootstrap', '')) if not bootstrap else path_to_results

            for random_seed_test in random_seeds_test:
                # Load pre-saved data
                X_dev = pd.read_pickle(Path(path_to_results, f'random_seed_{random_seed_test}', 'X_dev.pkl'))
                X_test = pd.read_pickle(Path(path_to_results, f'random_seed_{random_seed_test}', 'X_test.pkl'))
                y_dev = pd.read_pickle(Path(path_to_results, f'random_seed_{random_seed_test}', 'y_dev.pkl'))
                y_test = pd.read_pickle(Path(path_to_results, f'random_seed_{random_seed_test}', 'y_test.pkl'))
                IDs_test = pd.read_pickle(Path(path_to_results, f'random_seed_{random_seed_test}', 'IDs_test.pkl'))
                IDs_dev = pd.read_pickle(Path(path_to_results, f'random_seed_{random_seed_test}', 'IDs_dev.pkl'))

                files = [
                    file
                    for file in Path(path_to_results, f'random_seed_{random_seed_test}').iterdir()
                    if 'all_performances' in file.stem and 'test' not in file.stem
                ]

                for file in files:
                    model_name = file.stem.split('_')[-1]
                    print(model_name)

                    if Path(file.parent, f'best_{n_models}_{model_name}_test.csv').exists():
                        continue

                    # Read results
                    results = pd.read_csv(file)
                    results = results.sort_values(by=f'{extremo}_{sort_by}_bootstrap', ascending=ascending).reset_index(drop=True)

                    if 'Fugu_logit__POS_std' in results.columns:
                        print(f'Fufu_logit__POS_std in {task} {dimension} {y_label}')
                        continue

                    results_test = pd.DataFrame()

                    for r, row in tqdm.tqdm(results.loc[:n_models, ].iterrows(), total=n_models):
                        all_features = [col for col in results.columns if any(x in col for x in dimension.split('_'))]
                        results_r = row.dropna().to_pandas().to_dict()

                        params = {
                            key: value
                            for (key, value) in results_r.items()
                            if all(
                                x not in key for x in ['inf', 'mean', 'sup', id_col] + all_features + y_labels
                            )
                        }

                        if 'gamma' in params.keys():
                            try:
                                params['gamma'] = float(params['gamma'])
                            except:
                                pass
                        if 'random_state' in params.keys():
                            params['random_state'] = int(params['random_state'])

                        features = [col for col in all_features if results_r.get(col, 0) == 1]
                        features_dict = {col: results_r.get(col, 0) for col in all_features}

                        # Initialize the model
                        mod = models_dict[model_name](**params)

                        # Ensure data is in cuDF format
                        X_dev_gpu = X_dev[features]
                        X_test_gpu = X_test[features]
                        y_dev_gpu = y_dev
                        y_test_gpu = y_test

                        # Apply imputation if necessary
                        if imputer is not None:
                            imputer_model = imputer.fit(X_dev_gpu)
                            X_dev_gpu = imputer_model.transform(X_dev_gpu)
                            X_test_gpu = imputer_model.transform(X_test_gpu)

                        # Apply scaling if necessary
                        if scaler is not None:
                            scaler_model = scaler.fit(X_dev_gpu)
                            X_dev_gpu = scaler_model.transform(X_dev_gpu)
                            X_test_gpu = scaler_model.transform(X_test_gpu)

                        # Convert to CuPy arrays
                        X_dev_gpu = X_dev_gpu.values
                        X_test_gpu = X_test_gpu.values
                        y_dev_gpu = y_dev_gpu.values
                        y_test_gpu = y_test_gpu.values

                        # Train the model
                        mod.fit(X_dev_gpu, y_dev_gpu)

                        # Predict on test data
                        y_pred_test = mod.predict(X_test_gpu)
                        y_pred_dev = mod.predict(X_dev_gpu)


                        metrics_test_bootstrap = {}
                        metrics_test_bootstrap['r2_score'] = r2_score(y_test_gpu, y_pred_test)
                        metrics_test_bootstrap['mean_squared_error'] = mean_squared_error(y_test_gpu, y_pred_test)
                        metrics_test_bootstrap['mean_absolute_error'] = mean_absolute_error(y_test_gpu, y_pred_test)

                        result_append = params.copy()
                        result_append.update(features_dict)

                        for metric in metrics_names:
                            # Use computed metrics
                            metric_value = float(metrics_test_bootstrap[metric])
                            result_append[f'mean_{metric}_bootstrap_test'] = metric_value
                            # Use dev metrics from results_r
                            result_append[f'inf_{metric}_bootstrap_dev'] = results_r.get(f'inf_{metric}_bootstrap', np.nan)
                            result_append[f'mean_{metric}_bootstrap_dev'] = results_r.get(f'mean_{metric}_bootstrap', np.nan)
                            result_append[f'sup_{metric}_bootstrap_dev'] = results_r.get(f'sup_{metric}_bootstrap', np.nan)

                        if results_test.empty:
                            results_test = pd.DataFrame(columns=result_append.keys())

                        results_test = results_test.append(result_append, ignore_index=True)

                    results_test.to_csv(Path(file.parent, f'best_{n_models}_{model_name}_test.csv'), index=False)
