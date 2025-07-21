import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings, json, os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from expected_cost.utils import plot_hists
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import bootstrap, percentileofscore

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

warnings.filterwarnings('ignore')

def _calculate_metric_diffs(indices, outputs1, y_dev1, outputs2, y_dev2, metrics, prob_type, cost_matrix):
    """
    Statistic function for bootstrap. Calculates differences for ALL metrics at once.
    """
    # Resample y, ensuring we don't operate on an empty or invalid slice
    resampled_y1 = y_dev1[..., indices].ravel()
    resampled_y2 = y_dev2[..., indices].ravel()

    # If a resample is degenerate (e.g., missing a class), metric calculation is impossible.
    # Return NaNs to signal this. The 'bca' method will fail, triggering our fallback.
    if np.unique(resampled_y1).shape[0] != np.unique(y_dev1).shape[0] or \
       np.unique(resampled_y2).shape[0] != np.unique(y_dev2).shape[0]:
        return np.full(len(metrics), np.nan)

    # Resample model outputs
    resampled_out1 = outputs1[..., indices, :].reshape(-1, outputs1.shape[-1])
    resampled_out2 = outputs2[..., indices, :].reshape(-1, outputs2.shape[-1])

    # Get metrics for both classifiers
    if prob_type == 'clf':
        metrics1, _ = utils.get_metrics_clf(resampled_out1, resampled_y1, metrics, cmatrix=cost_matrix)
        metrics2, _ = utils.get_metrics_clf(resampled_out2, resampled_y2, metrics, cmatrix=cost_matrix)
    else: # 'reg'
        metrics1 = utils.get_metrics_reg(resampled_out1, resampled_y1, metrics)
        metrics2 = utils.get_metrics_reg(resampled_out2, resampled_y2, metrics)

    # Return an array of differences
    return np.array([metrics1[m] - metrics2[m] for m in metrics])

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config['project_name']
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
bayes = bool(config['bayes'])
feature_selection = bool(config['feature_selection'])
filter_outliers = config['filter_outliers']
test_size = float(config['test_size'])
n_boot = int(config['n_boot'])
calibrate = bool(config['calibrate'])
from expected_cost.ec import CostMatrix

home = Path(os.environ.get('HOME', Path.home()))
if 'Users/gp' in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path('D:/CNC_Audio/gonza/results', project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

scoring_metrics = main_config['scoring_metrics'][project_name]
metrics_names = main_config['metrics_names'][main_config['problem_type'][project_name]]
tasks = main_config['tasks'][project_name]
y_labels = main_config['y_labels'][project_name]
cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]

# Set the style for the plots
sns.set(style='whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

if problem_type == 'clf':
    for scoring in scoring_metrics:
        all_results = pd.DataFrame()
        extremo = 'sup' if 'norm' in scoring else 'inf'
        ascending = True if extremo == 'sup' else False

        best_models_filename = f'best_best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_calibrated_bayes.csv'.replace('__','_')
        if not hyp_opt:
            best_models_filename = best_models_filename.replace('_hyp_opt','')
        if not feature_selection:
            best_models_filename = best_models_filename.replace('_feature_selection','')
        if not calibrate:
            best_models_filename = best_models_filename.replace('_calibrated','')

        best_models = pd.read_csv(Path(results_dir,best_models_filename))

        tasks = best_models['task'].unique()
        dimensions = best_models['dimension'].unique()    

        for task,dimension,y_label in itertools.product(tasks,dimensions,y_labels):

            print(task, dimension)
            path_to_results = Path(results_dir, task, dimension, scaler_name, kfold_folder, y_label, stat_folder,'bayes',scoring,'hyp_opt' if hyp_opt else '', 'feature_selection' if feature_selection else '', 'filter_outliers' if filter_outliers and problem_type == 'reg' else '')

            row = best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label)]
            if row.empty:
                continue
                
            model_name = row['model_type'].values[0]
            
            if str(row['random_seed_test']) == 'nan':
                random_seed = ''
            else:
                random_seed = row.random_seed_test
                    
            outputs_filename = f'outputs_{model_name}.pkl'
            try:
                config['shuffle'] = False
                outputs, y_dev = utils._load_data(results_dir,task,dimension,y_label,model_name,'',config, bayes=True, scoring=scoring)
                config['shuffle'] = True
                outputs_shuffle, y_dev_shuffle = utils._load_data(results_dir, task, dimension, y_label,model_name, '', config, bayes=True, scoring=scoring)
            except:
                continue
            # Ensure the datasets are comparable
            try:
                assert y_dev.shape == y_dev_shuffle.shape, "y_dev shapes must match for paired comparison!"
            except:
                print(f"Shape mismatch for {task}, {dimension}")
                continue
            # Prepare data for bootstrap: a tuple of index arrays to resample
            data_indices = (np.arange(y_dev.shape[-1]),)

            # Define the statistic function with data baked in
            stat_func = lambda indices: _calculate_metric_diffs(
                indices, outputs, y_dev, outputs_shuffle, y_dev_shuffle, 
                metrics_names, problem_type, cmatrix
            )

            # 1. Calculate the point estimate (the actual difference on the full dataset)
            point_estimates = stat_func(data_indices[0])

            # 2. Calculate the bootstrap confidence interval
            try:
                # Try the more accurate BCa method first
                res = bootstrap(
                    data_indices,
                    stat_func,
                    n_resamples=n_boot, # Use configured n_boot
                    method=config["bootstrap_method"],
                    vectorized=False,
                    random_state=42
                )
                bootstrap_method = config["bootstrap_method"]

            except ValueError as e:
                # If BCa fails (e.g., due to degenerate samples), fall back to percentile
                print(f"WARNING: {config['bootstrap_method']} method failed for {tasks}/{dimensions}/{y_label}. Falling back to 'percentile'. Error: {e}")
                res = bootstrap(
                    data_indices,
                    stat_func,
                    n_resamples=n_boot,
                    method='percentile',
                    vectorized=False,
                    random_state=42
                )
                bootstrap_method = 'percentile'
            # Store results for this comparison
            result_row = {
                "task": task,
                "dimension": dimension,
                "y_label": y_label,
                "bootstrap method": bootstrap_method
            }
            
            for i, metric in enumerate(metrics_names):
                est = point_estimates[i]
                distribution = res.bootstrap_distribution[i] if est > 0 else -res.bootstrap_distribution[i]
                ci_low, ci_high = res.confidence_interval.low[i], res.confidence_interval.high[i]
                result_row[metric] = f"{est:.3f}, ({ci_low:.3f}, {ci_high:.3f})"
                result_row[f'p_value_{metric}'] = 2*np.round(percentileofscore(distribution,0) / 100.0,3)
            
            if all_results.empty:
                all_results = pd.DataFrame(columns=result_row.keys())
            
            all_results.loc[len(all_results.index),:] = result_row
    
    # --- Save Final Results ---
    output_filename = Path(results_dir, f'ic_diff_{scoring}_shuffle_bayes.csv') # Assumes one scoring metric for filename
    all_results.to_csv(output_filename, index=False)
    print(f"Confidence interval differences saved to {output_filename}")
    
