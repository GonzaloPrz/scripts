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

conf_int_metrics = pd.DataFrame(columns=['task','dimension','y_label','model_type','random_seed_test'] + metrics_names)

for scoring in scoring_metrics:
    
    #if (Path(results_dir,f'metrics_{kfold_folder}_{scoring}_{stat_folder}_feature_selection_dev.csv'.replace('__','_') if feature_selection else f'metrics_{kfold_folder}_{scoring}_{stat_folder}_dev.csv'.replace('__','_')).exists()) and (not overwrite):
    #    continue

    for task,model,y_label in itertools.product(tasks,models,y_labels):    
        
        if Path(results_dir,task).exists() == False:
            continue
        
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

        for dimension in dimensions:
            print(task,model,dimension,y_label)
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'bayes',scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '')
            
            if not path.exists():  
                continue

            random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name] if config["test_size"] > 0 else []
            if len(random_seeds) == 0:
                random_seeds = ['']
            
            for random_seed in random_seeds:

                if not Path(path,random_seed,f'outputs_{model}.pkl').exists():
                    continue

                outputs = pickle.load(open(Path(path,random_seed,f'outputs_{model}.pkl'),'rb'))
            
                y_dev = pickle.load(open(Path(path,random_seed,'y_dev.pkl'),'rb'))

                IDs_dev = pickle.load(open(Path(path,random_seed,'IDs_dev.pkl'),'rb'))

                if (cmatrix is not None) or (np.unique(y_dev).shape[0] > 2): 
                    metrics_names_ = list(set(metrics_names) - set(["accuracy","roc_auc","f1","precision","recall"])) 
                else:
                    metrics_names_ = metrics_names

                metrics = dict((metric,'') for metric in metrics_names_)
                                     
                n_seeds, n_samples, n_outputs = outputs.shape
                data = (np.arange(n_samples),)  # indices for the sample axis

                def get_metric(metric_name, indices):
                    # indices: shape (n_bootstrap_samples,)
                    resampled_outputs = outputs[:, indices, :].reshape(-1,outputs.shape[-1]) # Preserve all seeds and output dim
                    resampled_y_dev = y_dev[:, indices].ravel()       # Same indices for y_dev

                    try:
                        if problem_type == 'clf':
                            metric, _ = utils.get_metrics_clf(resampled_outputs, resampled_y_dev, [metric_name], cmatrix=cmatrix, priors=None, threshold=None)
                        else:
                            metric = utils.get_metrics_reg(resampled_outputs, resampled_y_dev, [metric_name])
                        return metric[metric_name]
                    except Exception as e:
                        print(f"Error calculating metric {metric_name} for indices {indices}: {e}")
                        return np.nan
                
                conf_int_metrics_append = {'task': task, 'dimension': dimension, 'y_label': y_label, 'model_type': model, 'random_seed_test': random_seed}
                metrics_results = {}

                for metric in metrics_names_:
                    res = bootstrap(
                        data, 
                        lambda idx: get_metric(metric, idx),
                        vectorized=False,         # get_metric works on one index set at a time
                        paired=False,             # Single array of indices
                        n_resamples=1000,
                        confidence_level=0.95,
                        method='bca',
                        random_state=42
                    )
                    ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
                    estimate = get_metric(metric, np.arange(n_samples))  # Full sample
                    metrics_results[metric] = {'estimate': estimate, 'CI': (ci_low, ci_high)}
                    
                    conf_int_metrics_append.update({metric: f'{np.round(estimate,3)}, ({np.round(ci_low,3)}, {np.round(ci_high,3)})'})
                  
                conf_int_metrics.loc[len(conf_int_metrics.index),:] = conf_int_metrics_append

    conf_int_metrics.to_csv(Path(results_dir,f'metrics_{kfold_folder}_{scoring}_{stat_folder}_feature_selection_dev.csv'.replace('__','_') if feature_selection else f'metrics_{kfold_folder}_{scoring}_{stat_folder}_dev.csv'.replace('__','_')),index=False)