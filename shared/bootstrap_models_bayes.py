import pandas as pd
import pickle
from pathlib import Path
from expected_cost.utils import *
import itertools
import sys,json
import numpy as np
from scipy.stats import bootstrap

from expected_cost.ec import CostMatrix

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

##---------------------------------PARAMETERS---------------------------------##
config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = True if int(config['n_iter']) > 0 else False
feature_selection = bool(config['feature_selection'])
filter_outliers = config['filter_outliers']
n_boot = int(config["n_boot"])
calibrate = bool(config["calibrate"])
overwrite = bool(config["overwrite"])

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
if not isinstance(scoring_metrics,list):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]
models = main_config["models"][project_name]
metrics_names = main_config["metrics_names"][problem_type]
cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
n_boot = int(config["n_boot"])
problem_type = config["problem_type"]

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]

if not isinstance(scoring_metrics,list):
    scoring_metrics = [scoring_metrics]
    
problem_type = main_config['problem_type'][project_name]
metrics_names = main_config["metrics_names"][main_config["problem_type"][project_name]]
tasks = main_config["tasks"][project_name]
cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None

for scoring in scoring_metrics:
    scoring_col = f'{scoring}_extremo'

    extremo = 1 if any(x in scoring for x in ['error','norm']) else 0
    ascending = any(x in scoring for x in ['error','norm'])

    output_filename = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffled_calibrated_bayes.csv'.replace('__','_')
    if not hyp_opt:
            output_filename = output_filename.replace('_hyp_opt','')
    if not feature_selection:
        output_filename = output_filename.replace('_feature_selection','')
    if not shuffle_labels:
        output_filename = output_filename.replace('_shuffled','')
    if not calibrate:
        output_filename = output_filename.replace('_calibrated','')

    if (Path(results_dir,output_filename).exists()) & (not overwrite):
        all_results = pd.read_csv(Path(results_dir,output_filename))
    else:
        all_results = pd.DataFrame()
    
    for task, model_type in itertools.product(tasks,models):
        if not Path(results_dir,task).exists():
            continue
        
        if model_type == 'lasso':
            continue

        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            
            if isinstance(y_labels,dict):
                y_labels_ = y_labels[task]
            else:
                y_labels_ = y_labels
            
            for y_label in y_labels_:
                print(task,model_type,y_label,dimension)
                
                path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,"bayes",scoring,"hyp_opt" if hyp_opt else "","feature_selection" if feature_selection else "","shuffle" if shuffle_labels else "")

                if not path.exists():
                    continue

                random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name] if config['test_size'] > 0 else []

                if len(random_seeds) == 0:
                    random_seeds = ['']

                for random_seed in random_seeds:
                    
                    if not overwrite and all_results.shape[0] > 0:
                        row = all_results[(all_results['task'] == task) & (all_results['dimension'] == dimension) & (all_results['model_type'] == model_type) & (all_results['y_label'] == y_label)]
                        if len(row) > 0:
                            continue

                    if not utils._build_path(results_dir,task,dimension,y_label,random_seed,f"outputs_{model_type}.pkl",config,bayes=True,scoring=scoring).exists():
                        continue

                    outputs, y_dev = utils._load_data(results_dir,task,dimension,y_label,model_type,random_seed,config,bayes=True,scoring=scoring)

                    if (cmatrix is not None) or (np.unique(y_dev).shape[0] > 2):
                        metrics_names_ = list(set(metrics_names) - set(["roc_auc","f1","precision","recall"]))
                    else:
                        metrics_names_ = metrics_names
                        
                    # Prepare data for bootstrap: a tuple of index arrays to resample
                    data_indices = (np.arange(y_dev.shape[-1]),)

                    # Define the statistic function with data baked in
                    stat_func = lambda indices: utils._calculate_metrics(
                        indices, outputs, y_dev, 
                        metrics_names_, problem_type, cmatrix
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
                        "model_type": model_type,
                        "random_seed_test": random_seed,
                        "bootstrap_method_dev": bootstrap_method
                    }
                    
                    for i, metric in enumerate(metrics_names_):
                        est = point_estimates[i]
                        ci_low, ci_high = res.confidence_interval.low[i], res.confidence_interval.high[i]
                        result_row[metric] = f"{est:.5f}, ({ci_low:.5f}, {ci_high:.5f})"
                    
                    if all_results.empty:
                        all_results = pd.DataFrame(result_row,index=[0])
                    else:
                        all_results.loc[all_results.shape[0],:] = result_row

                    all_results.to_csv(Path(results_dir,output_filename),index=False)
        
    best_best_models = pd.DataFrame(columns=all_results.columns)
    random_seeds_test = all_results['random_seed_test'].unique()

    for task,random_seed_test in itertools.product(tasks,random_seeds_test):

        if not Path(results_dir,task).exists():
            continue
        
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            
            if isinstance(y_labels,dict):
                y_labels_ = y_labels[task]
            else:
                y_labels_ = y_labels
            
            for y_label in y_labels_:
                best_best_models_ = all_results[(all_results['task'] == task) & (all_results['y_label'] == y_label) & (all_results['dimension'] == dimension) & (all_results['random_seed_test'].astype(str) == str(random_seed_test))]

                try:                            
                    best_best_models_[scoring_col] = best_best_models_[scoring].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))
                except:
                    best_best_models_[scoring_col] = best_best_models_[f'{scoring}_score'].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))

                best_best_models_.dropna(subset=[scoring_col], inplace=True)
                
                try:
                    best_best_models_append = best_best_models_.sort_values(by=scoring_col,ascending=ascending).iloc[0]
                except:
                    print(f"WARNING: No valid models found for {task}/{dimension}/{y_label} with random seed {random_seed_test}. Skipping...")
                    continue

                if best_best_models_append['model_type'] =='lasso':
                    best_best_models_append = best_best_models_.sort_values(by=scoring_col,ascending=ascending).iloc[1]
                
                best_best_models.loc[best_best_models.shape[0],:] = best_best_models_append
    
    try:                            
        best_best_models[scoring_col] = best_best_models[scoring].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))
    except:
        best_best_models[scoring_col] = best_best_models[f'{scoring}_score'].apply(lambda x: float(x.split('(')[1].replace(')','').split(', ')[extremo]))

    best_best_models = best_best_models.sort_values(by=['y_label',scoring_col],ascending=ascending)

    best_best_best_models = pd.DataFrame(columns=best_best_models.columns)
    if isinstance(y_labels,dict):
        y_labels_ = sum(y_labels.values(),[])
    else:
        y_labels_ = y_labels
    for y_label,task in itertools.product(y_labels_,tasks):
        best_best_best_models.loc[best_best_best_models.shape[0],:] = best_best_models[(best_best_models['y_label'] == y_label) & (best_best_models['task'] == task)].iloc[0].values
    
    best_best_best_models.to_csv(Path(results_dir,f'best_{output_filename}'),index=False)