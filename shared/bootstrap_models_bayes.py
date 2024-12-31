import pandas as pd
import pickle
from pathlib import Path
from expected_cost.utils import *
import itertools
from joblib import Parallel, delayed
import sys,tqdm,json
from pingouin import compute_bootci

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

def get_metrics_bootstrap(samples, targets, metrics_names, random_state=42, n_boot=2000, decimals=5,cmatrix=None, priors=None, problem_type='clf'):
    assert samples.shape[0] == targets.shape[0]
    
    metrics_ci = dict((metric,(np.empty(0),np.empty((0,2)))) for metric in metrics_names)
    all_metrics = dict((metric,np.empty(n_boot)) for metric in metrics_names)
    
    for metric in metrics_names:
        def get_metric(indices):
            if problem_type == 'clf':
                metric_value, y_pred = get_metrics_clf(samples[indices], targets[indices], [metric], cmatrix)
            else:
                metric_value = get_metrics_reg(samples[indices], targets[indices], [metric])
            return metric_value[metric]
                    
        results = compute_bootci(x=np.arange(targets.shape[0]),func=get_metric,n_boot=n_boot,method='cper',seed=random_state,return_dist=True,decimals=decimals)
        metrics_ci[metric] = (np.round(np.nanmean(results[1]),2),results[0])
        all_metrics[metric] = results[1]

    return metrics_ci, all_metrics
##---------------------------------PARAMETERS---------------------------------##
project_name = 'GeroApathy'

feature_selection = True
filter_outliers = True
shuffle_labels = False
n_folds = 5

n_boot = 200
scaler_name = 'StandardScaler'
id_col = 'id'

l2ocv = False
# Check if required arguments are provided
if len(sys.argv) > 1:
    #print("Usage: python bootstrap_models_bca.py <project_name> [hyp_opt] [filter_outliers] [shuffle_labels] [feature_selection] [k]")
    project_name = sys.argv[1]
if len(sys.argv) > 2:
    feature_selection = bool(int(sys.argv[2]))
if len(sys.argv) > 3:
    filter_outliers = bool(int(sys.argv[3]))
if len(sys.argv) > 4:
    shuffle_labels = bool(int(sys.argv[4]))
if len(sys.argv) > 5:
    n_folds = int(sys.argv[5])

models = {'MCI_classifier':['lr','svc','knnc','xgb'],
          'tell_classifier':['lr','svc','knnc','xgb'],
          'Proyecto_Ivo':['lr','svc','knnr','xgb'],
          'GeroApathy':['lr','svc','knnr','xgb'],
            'GERO_Ivo':['elastic','lasso','ridge','knnr','svr']
            }

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['Animales','P','Animales__P','cog','brain','AAL','conn'],
         'GeroApathy':['agradable'],
         'GERO_Ivo':['animales','grandmean','fas__animales','fas']
         }

single_dimensions = {'tell_classifier':['voice-quality','talking-intervals','pitch'],
                     'MCI_classifier':['talking-intervals','psycholinguistic'],
                     'Proyecto_Ivo':[],
                     'GERO_Ivo':[],
                     'GeroApathy':[]}

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GERO_Ivo':'reg',
                'GeroApathy':'clf'}

metrics_names = {'clf':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                'reg':['r2_score','mean_squared_error','mean_absolute_error']}

y_labels = {'MCI_classifier':['target'],
            'tell_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GERO_Ivo':['MMSE_Total_Score'],
            'GeroApathy':['DASS_21_Depression_V_label','AES_Total_Score_label'
                          ,'Depression_Total_Score_label','MiniSea_MiniSea_Total_EkmanFaces_label',
                          'MiniSea_minisea_total_label'],
            }

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

conf_int_metrics = pd.DataFrame(columns=['task','dimension','y_label','model_type','metric','mean','95_ci'])

for task,model,y_label in itertools.product(tasks[project_name],models[project_name],y_labels[project_name]):    
    
    dimensions = list()

    for ndim in range(1,len(single_dimensions[project_name])+1):
        for dimension in itertools.combinations(single_dimensions[project_name],ndim):
            dimensions.append('__'.join(dimension))
    
    if not Path(results_dir,task).exists():
        continue

    if len(dimensions) == 0:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

    for dimension in dimensions:
        print(task,model,dimension,y_label)
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','bayes','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type[project_name] == 'reg' else '','shuffle' if shuffle_labels else '')
        
        if not path.exists():  
            continue

        random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
        if len(random_seeds) == 0:
            random_seeds = ['']
        
        for random_seed in random_seeds:

            if not Path(path,random_seed,f'outputs_best_{model}.pkl').exists():
                continue
            
            outputs = pickle.load(open(Path(path,random_seed,f'outputs_best_{model}.pkl'),'rb'))
            
            try:
                y_dev = pickle.load(open(Path(path,random_seed,'y_true_dev.pkl'),'rb'))
            except:
                y_dev = pickle.load(open(Path(path,random_seed,'y_true.pkl'),'rb'))

            outputs_bootstrap = np.empty((n_boot,outputs.shape[0],outputs.shape[1],outputs.shape[2])) if outputs.ndim == 3 else np.empty((n_boot,outputs.shape[0],outputs.shape[1]))
            y_dev_bootstrap = np.empty((n_boot,y_dev.shape[0],y_dev.shape[1]),dtype=y_dev.dtype)
            y_pred_bootstrap = np.empty((n_boot,y_dev.shape[0],y_dev.shape[1]),dtype=y_dev.dtype)
            
            metrics = dict((metric,np.empty((outputs.shape[0],n_boot))) for metric in metrics_names[problem_type[project_name]])

            for r in range(outputs.shape[0]):
                _,metrics_ = get_metrics_bootstrap(outputs[r], y_dev[r], metrics_names[problem_type[project_name]],n_boot=n_boot,problem_type=problem_type[project_name])
                for metric in metrics_names[problem_type[project_name]]:
                    metrics[metric][r,:] = metrics_[metric]
            for metric in metrics_names[problem_type[project_name]]:
                mean, ci = np.nanmean(metrics[metric].squeeze()).round(5), (np.nanpercentile(metrics[metric].squeeze(),2.5).round(5),np.nanpercentile(metrics[metric].squeeze(),97.5).round(5))
                conf_int_metrics.loc[len(conf_int_metrics.index),:] = [task,dimension,y_label,model,metric,mean,f'[{ci[0]},{ci[1]}]']   

            pickle.dump(outputs_bootstrap,open(Path(path,random_seed,f'outputs_bootstrap_best_{model}.pkl'),'wb'))
            pickle.dump(y_dev_bootstrap,open(Path(path,random_seed,f'y_dev_bootstrap.pkl'),'wb'))
            pickle.dump(y_pred_bootstrap,open(Path(path,random_seed,f'y_pred_bootstrap_{model}.pkl'),'wb'))
            pickle.dump(metrics,open(Path(path,random_seed,f'metrics_bootstrap_{model}.pkl'),'wb'))

conf_int_metrics.to_csv(Path(results_dir,'metrics_feature_selection_dev.csv' if feature_selection else 'metrics_dev.csv'),index=False)