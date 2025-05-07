import pandas as pd
import pickle
from pathlib import Path
from expected_cost.utils import *
import itertools
import sys,json
import numpy as np

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
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = config['feature_selection']
filter_outliers = config['filter_outliers']
n_models = int(config["n_models"])
n_boot = int(config["n_boot"])
calibrate = bool(config["calibrate"])
overwrite = bool(config["overwrite"])
parallel = bool(config["parallel"])

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

conf_int_metrics = pd.DataFrame(columns=['task','dimension','y_label','model_type','metric','mean','95_ci'])

for scoring in scoring_metrics:
    for task,model,y_label,scoiring_metric in itertools.product(tasks,models,y_labels,scoring_metrics):    

        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

        for dimension in dimensions:
            print(task,model,dimension,y_label)
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,stat_folder,y_label,'bayes',scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '')
            
            if not path.exists():  
                continue

            random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
            if len(random_seeds) == 0:
                random_seeds = ['']
            
            for random_seed in random_seeds:

                if not Path(path,random_seed,f'outputs_{model}.pkl').exists():
                    continue
                
                outputs = pickle.load(open(Path(path,random_seed,f'outputs_{model}.pkl'),'rb'))
            
                y_dev = pickle.load(open(Path(path,random_seed,'y_dev.pkl'),'rb'))

                IDs_dev = pickle.load(open(Path(path,random_seed,'IDs_dev.pkl'),'rb'))

                metrics_names = main_config["metrics_names"][problem_type]
                metrics_names = metrics_names if cmatrix == None and outputs.shape[-1] == 2 else list(set(metrics_names) - set(['roc_auc','f1','recall']))

                outputs_bootstrap = np.empty((n_boot,outputs.shape[0],outputs.shape[1],outputs.shape[2])) if outputs.ndim == 3 else np.empty((n_boot,outputs.shape[0],outputs.shape[1]))
                y_dev_bootstrap = np.empty((n_boot,y_dev.shape[0],y_dev.shape[1]),dtype=y_dev.dtype)
                y_pred_bootstrap = np.empty((n_boot,y_dev.shape[0],y_dev.shape[1]),dtype=y_dev.dtype)

                metrics = dict((metric,np.empty((outputs.shape[0],n_boot))) for metric in metrics_names)

                for r in range(outputs.shape[0]):

                    metrics_, _ = utils.get_metrics_bootstrap(outputs[r], y_dev[r], IDs_dev[r], metrics_names,n_boot=n_boot,cmatrix=cmatrix,priors=None,threshold=None,problem_type=problem_type)
                    for metric in metrics_names:
                        metrics[metric][r,:] = metrics_[metric]
                for metric in metrics_names:
                    mean, ci = np.nanmean(metrics[metric].squeeze()).round(3), (np.nanpercentile(metrics[metric].squeeze(),2.5).round(3),np.nanpercentile(metrics[metric].squeeze(),97.5).round(3))
                    conf_int_metrics.loc[len(conf_int_metrics.index),:] = [task,dimension,y_label,model,metric,mean,f'[{ci[0]},{ci[1]}]']   

                pickle.dump(outputs_bootstrap,open(Path(path,random_seed,f'outputs_bootstrap_best_{model}.pkl'),'wb'))
                pickle.dump(y_dev_bootstrap,open(Path(path,random_seed,f'y_dev_bootstrap.pkl'),'wb'))
                pickle.dump(y_pred_bootstrap,open(Path(path,random_seed,f'y_pred_bootstrap_{model}.pkl'),'wb'))
                pickle.dump(metrics,open(Path(path,random_seed,f'metrics_bootstrap_{model}.pkl'),'wb'))

    conf_int_metrics.to_csv(Path(results_dir,f'metrics_{scoring}_feature_selection_dev.csv' if feature_selection else f'metrics_{scoring}_dev.csv'),index=False)