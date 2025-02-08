import pandas as pd
import pickle
from pathlib import Path
from expected_cost.utils import *
import itertools
from joblib import Parallel, delayed
import sys,tqdm
from pingouin import compute_bootci
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import logging, sys
import json
import argparse 

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

def compute_metrics(j, model_index, r,outputs, y_dev, metrics_names, n_boot, problem_type, cmatrix=None, priors=None, threshold=None,bayesian=False):
    # Calculate the metrics using the bootstrap method
    results = get_metrics_bootstrap(outputs[j,model_index,r], y_dev[j, r], metrics_names, n_boot=n_boot, cmatrix=cmatrix,priors=priors,threshold=threshold,problem_type=problem_type,bayesian=bayesian)

    metrics_result = {}
    for metric in metrics_names:
        metrics_result[metric] = results[metric]
    return j,model_index,r, metrics_result

def get_metrics_bootstrap(samples, targets, metrics_names, n_boot=2000,cmatrix=None,priors=None,threshold=None,problem_type='clf',bayesian=False):
    all_metrics = dict((metric,np.empty(n_boot)) for metric in metrics_names)
    
    for metric in metrics_names:
        if bayesian:
            weights = np.random.dirichlet(np.ones(samples.shape[0]))
        else:
            weights = None

        for b in range(n_boot):
            indices = np.random.choice(targets.shape[0], targets.shape[0], replace=True)
            if problem_type == 'clf':
                metric_value, y_pred = utils.get_metrics_clf(samples[indices], targets[indices], [metric], cmatrix,priors,threshold,weights)
            else:
                metric_value = utils.get_metrics_reg(samples[indices], targets[indices], [metric])
            all_metrics[metric][b] = metric_value[metric]
        
    return all_metrics
##---------------------------------PARAMETERS---------------------------------##
project_name = 'arequipa'

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

config = json.load(Path(results_dir,'config.json').open())

scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
n_models = int(config["n_models"])
n_boot = int(config["n_boot"])
bayesian = bool(config["bayesian"])
parallel = False 
cmatrix = None

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
problem_type = main_config['problem_type'][project_name]
models = main_config["models"][project_name]
metrics_names = main_config["metrics_names"][problem_type]

##---------------------------------PARAMETERS---------------------------------##
for task,model,y_label,scoring in itertools.product(tasks,models,y_labels,[scoring_metrics]):    
    dimensions = list()

    for ndim in range(1,len(single_dimensions)+1):
        for dimension in itertools.combinations(single_dimensions,ndim):
            dimensions.append('__'.join(dimension))

    if len(dimensions) == 0:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

    for dimension in dimensions:
        print(task,model,dimension,y_label)
        path = Path(results_dir,task,dimension,scaler_name, kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '')
        
        if not path.exists():  
            continue

        random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
        if len(random_seeds) == 0:
            random_seeds = ['']
        
        for random_seed in random_seeds:
            '''        
            if config['n_models'] == 0:

                if Path(path,random_seed,'bayesian' if bayesian else '',f'all_models_{model}_dev_bca.csv').exists():
                    continue
            elif Path(path,random_seed,'bayesian' if bayesian else '',f'best_models_{model}_dev_bca_{scoring}.csv').exists():
                    continue 
            
            if not Path(path,random_seed,f'all_models_{model}.csv').exists():
                continue
            '''
            all_models = pd.read_csv(Path(path,random_seed,f'all_models_{model}.csv'))
            outputs = pickle.load(open(Path(path,random_seed,f'outputs_{model}.pkl'),'rb'))

            y_dev = pickle.load(open(Path(path,random_seed,'y_dev.pkl'),'rb'))
            
            scorings = np.empty(outputs.shape[0])
            
            if config['n_models'] == 0:
                n_models = outputs.shape[0]
                all_models_bool = True
            else:
                all_models_bool = False
                if config['n_models'] < 1:
                    n_models = int(outputs.shape[0]*n_models)

                for i in range(outputs.shape[0]):
                    scorings_i = np.empty((outputs.shape[1],outputs.shape[2]))
                    for j,r in itertools.product(range(outputs.shape[1]),range(outputs.shape[2])):
                        if problem_type[project_name] == 'clf':
                            metrics, _ = get_metrics_clf(outputs[i,j,r], y_dev[j,r], [scoring], cmatrix)
                            scorings_i[j,r] = metrics[scoring]
                        else:
                            metrics = get_metrics_reg(outputs[i,j,r], y_dev[j,r],[scoring])
                            scorings_i[j,r] = metrics[scoring]
                    scorings[i] = np.nanmean(scorings_i.flatten())
                
                scorings = scorings if any(x in scoring for x in ['norm','error']) else -scorings

                best_models = np.argsort(scorings)[:n_models]
            
                all_models = all_models.iloc[best_models].reset_index(drop=True)
                all_models['idx'] = best_models
                outputs = outputs[best_models]
            
            metrics = dict((metric,np.empty((outputs.shape[0],outputs.shape[1],outputs.shape[2],int(config["n_boot"])))) for metric in metrics_names)
            
            all_results = Parallel(n_jobs=-1 if parallel else 1)(delayed(compute_metrics)(j,model_index,r, outputs, y_dev, metrics_names, int(config["n_boot"]), problem_type,cmatrix=None,priors=None,threshold=all_models.loc[model_index,'threshold'],bayesian=True) for j,model_index,r in itertools.product(range(outputs.shape[0]),range(outputs.shape[1]),range(outputs.shape[2])))

            # Update the metrics array with the computed results
            for j,model_index,r, metrics_result in tqdm.tqdm(all_results):
                for metric in metrics_names:
                    metrics[metric][j,model_index,r,:] = metrics_result[metric]

            # Update the summary statistics in all_models
            for model_index in tqdm.tqdm(range(outputs.shape[1])):
                for metric in metrics_names:
                    all_models.loc[model_index, f'{metric}_mean'] = np.nanmean(metrics[metric][:,model_index,:].flatten()).round(5)
                    all_models.loc[model_index, f'{metric}_inf'] = np.nanpercentile(metrics[metric][:,model_index,:].flatten(), 2.5).round(5)
                    all_models.loc[model_index, f'{metric}_sup'] = np.nanpercentile(metrics[metric][:,model_index,:].flatten(), 97.5).round(5)
            if bayesian:
                Path(path,random_seed,'bayesian').mkdir(exist_ok=True)
                
            all_models.to_csv(Path(path,random_seed,'bayesian' if bayesian else '',f'best_models_{model}_dev_bca_{scoring}.csv')) if all_models_bool == False else all_models.to_csv(Path(path,random_seed,'bayesian' if bayesian else '',f'all_models_{model}_dev_bca.csv')) 
    
logging.info("Bootstrap completed.")