import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import itertools
from joblib import Parallel, delayed
import sys,os,json

from expected_cost.ec import CostMatrix

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

late_fusion = False

##---------------------------------PARAMETERS---------------------------------##
config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
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
early_fusion = bool(config["early_fusion"])
bayesian = bool(config["bayesian"])
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
problem_type = main_config['problem_type'][project_name]
models = main_config["models"][project_name]
metrics_names = main_config["metrics_names"][problem_type]
cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None

##---------------------------------PARAMETERS---------------------------------##
for task,model,y_label,scoring in itertools.product(tasks,models,y_labels,[scoring_metrics]):    
    
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

    for dimension in dimensions:
        path = Path(results_dir,task,dimension,scaler_name, kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '',"shuffle" if shuffle_labels else "")
        
        if not path.exists():  
            continue

        random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
        if len(random_seeds) == 0:
            random_seeds = [folder.name for folder in path.parent.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
       
        if len(random_seeds) == 0:
            random_seeds = ['']
        
        for random_seed in random_seeds:
            
            filename_to_save = f'all_models_{model}_dev_bca_calibrated.csv'
            if not calibrate:
                filename_to_save = filename_to_save.replace('_calibrated','')
            if config['n_models'] != 0:
                filename_to_save = filename_to_save.replace('all_models','best_models').replace('.csv',f'_{scoring}.csv')

            if Path(path,random_seed,'bayesian' if bayesian else '',filename_to_save).exists() and overwrite == False:
                print(f"Bootstrapping already done for {task} - {y_label} - {model} - {dimension}. Skipping...")
                continue
              
            if not Path(path,random_seed,f'all_models_{model}.csv').exists():
                continue
            
            print(task,model,dimension,y_label)

            all_models = pd.read_csv(Path(path,random_seed,f'all_models_{model}.csv'))
            outputs = pickle.load(open(Path(path,random_seed,f'cal_outputs_{model}.pkl' if calibrate else f'outputs_{model}.pkl'),'rb')) 
            
            y_dev = pickle.load(open(Path(path,random_seed,'y_dev.pkl'),'rb')).astype(int)
            
            IDs_dev = pickle.load(open(Path(path,random_seed,'IDs_dev.pkl'),'rb'))

            metrics_names = main_config["metrics_names"][problem_type]
            metrics_names = metrics_names if cmatrix == None and outputs.shape[-1] == 2 else list(set(metrics_names) - set(['roc_auc','f1','recall']))

            scorings = np.empty(outputs.shape[1])
            
            if config['n_models'] == 0:
                n_models = outputs.shape[0]
                all_models_bool = True
            else:
                all_models_bool = False
                if config['n_models'] < 1:
                    n_models = int(outputs.shape[1]*n_models)

                for i in range(outputs.shape[1]):
                    scorings_i = np.empty((outputs.shape[0],outputs.shape[2]))
                    for j,r in itertools.product(range(outputs.shape[0]),range(outputs.shape[2])):
                        if problem_type[project_name] == 'clf':
                            metrics, _ = get_metrics_clf(outputs[j,i,r], y_dev[j,r], [scoring], cmatrix)
                            scorings_i[j,r] = metrics[scoring]
                        else:
                            metrics = get_metrics_reg(outputs[j,i,r], y_dev[j,r],[scoring])
                            scorings_i[j,r] = metrics[scoring]
                    scorings[i] = np.nanmean(scorings_i.flatten())
                
                scorings = scorings if any(x in scoring for x in ['norm','error']) else -scorings

                best_models = np.argsort(scorings)[:n_models]
            
                all_models = all_models.iloc[:,best_models].reset_index(drop=True)
                all_models['idx'] = best_models
                outputs = outputs[:,best_models]
            
            metrics = dict((metric,np.zeros((outputs.shape[0],outputs.shape[1],outputs.shape[2],int(config["n_boot"])))) for metric in metrics_names)
            
            all_results = Parallel(n_jobs=-1 if parallel else 1)(delayed(utils.compute_metrics)(j,model_index,r, outputs, y_dev, IDs_dev, metrics_names, int(config["n_boot"]), problem_type,cmatrix=cmatrix,priors=None,threshold=all_models.loc[model_index,'threshold'] if 'threshold' in all_models.columns else None,bayesian=bayesian) for j,model_index,r in itertools.product(range(outputs.shape[0]),range(outputs.shape[1]),range(outputs.shape[2])))
            # Update the metrics array with the computed results
            for j,model_index,r, metrics_result,_ in all_results:
                for metric in metrics_names:
                    metrics[metric][j,model_index,r,:] = metrics_result[metric]

            if len(all_results) == 0:
                continue
            # Update the summary statistics in all_models
            for model_index in range(outputs.shape[1]):
                for metric in metrics_names:
                    all_models.loc[model_index, f'{metric}_mean'] = np.nanmean(metrics[metric][:,model_index,:].flatten()).round(5)
                    all_models.loc[model_index, f'{metric}_inf'] = np.nanpercentile(metrics[metric][:,model_index,:].flatten(), 2.5).round(5)
                    all_models.loc[model_index, f'{metric}_sup'] = np.nanpercentile(metrics[metric][:,model_index,:].flatten(), 97.5).round(5)
            if bayesian:
                Path(path,random_seed,'bayesian').mkdir(exist_ok=True)
                
            all_models.to_csv(Path(path,random_seed,'bayesian' if bayesian else '',filename_to_save))