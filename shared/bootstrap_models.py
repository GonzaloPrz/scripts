import pandas as pd
import pickle
from pathlib import Path
from expected_cost.utils import *
import itertools
from joblib import Parallel, delayed
import sys,tqdm
from pingouin import compute_bootci
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

def compute_metrics(model_index, r, outputs, y_dev, metrics_names, n_boot, problem_type, project_name):
    # Calculate the metrics using the bootstrap method
    results = get_metrics_bootstrap(outputs[model_index, r], y_dev[r], metrics_names[project_name], n_boot=n_boot, problem_type=problem_type[project_name])
    
    metrics_result = {}
    for metric in metrics_names[project_name]:
        metrics_result[metric] = results[1][metric]
    return model_index, r, metrics_result

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
parallel = True

project_name = 'tell_classifier'
l2ocv = False

n_boot = 200
n_folds = 5
n_models = 100

cmatrix = None
shuffle_labels = False
hyp_opt_list = [True]
feature_selection_list = [True]

id_col = 'id'
scaler_name = 'StandardScaler'

models = {'MCI_classifier':['lr','svc','knn','xgb'],
          'tell_classifier':['lr','svc','knn','xgb'],
          'Proyecto_Ivo':['lr','svc','knn','xgb'],
          'GeroApathy':['lasso','ridge','knnr','svr','xgb','elastic'],
          'GERO_Ivo':['lasso','ridge','knnr','svr','xgb','elastic']
            }

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean' ],
         'Proyecto_Ivo':['Animales','P','Animales__P','cog','brain','AAL','conn'],
         'GeroApathy':['Fugu'],
         'GERO_Ivo':['animales','fas','grandmean','fas__animales']}

single_dimensions = {'tell_classifier':['voice-quality','talking-intervals','pitch'],
                     'MCI_classifier':['talking-intervals','psycholinguistic'],
                     'Proyecto_Ivo':[],
                     'GeroApathy':[],
                     'GERO_Ivo':[]}

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GeroApathy':'reg',
                'GERO_Ivo':'reg'}	

metrics_names = {'MCI_classifier':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'tell_classifier':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'Proyecto_Ivo':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'GeroApathy':['r2_score','mean_squared_error','mean_absolute_error'],
                    'GERO_Ivo':['r2_score','mean_squared_error','mean_absolute_error']}

y_labels = {'MCI_classifier':['target'],
            'tell_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GeroApathy':['DASS_21_Depression','DASS_21_Anxiety','DASS_21_Stress','AES_Total_Score','MiniSea_MiniSea_Total_FauxPas','Depression_Total_Score','MiniSea_emf_total','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total'],
            'GERO_Ivo':['GM_norm','WM_norm','norm_vol_bilateral_HIP','norm_vol_mask_AD']}

scoring = {'MCI_classifier':'norm_cross_entropy',
           'tell_classifier':'norm_cross_entropy',
           'Proyecto_Ivo':'roc_auc',
           'GeroApathy':'r2_score',
           'GERO_Ivo':'r2_score'}
##---------------------------------PARAMETERS---------------------------------##
if l2ocv:
    kfold_folder = 'l2ocv'
else:
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for task,model,y_label,hyp_opt,feature_selection in itertools.product(tasks[project_name],models[project_name],y_labels[project_name],hyp_opt_list,feature_selection_list):    
    
    dimensions = list()

    for ndim in range(1,len(single_dimensions[project_name])+1):
        for dimension in itertools.combinations(single_dimensions[project_name],ndim):
            dimensions.append('__'.join(dimension))

    if len(dimensions) == 0:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

    for dimension in dimensions:
        print(task,model,dimension,y_label)
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
        
        if not path.exists():  
            continue

        random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
        if len(random_seeds) == 0:
            random_seeds = ['']
        
        for random_seed in random_seeds:

            #if Path(path,random_seed,f'all_models_{model}_dev.csv').exists() or Path(path,random_seed,f'all_models_{model}.csv').exists() == False:
            #    continue
            
            all_models = pd.read_csv(Path(path,random_seed,f'all_models_{model}.csv'))
            outputs = pickle.load(open(Path(path,random_seed,f'outputs_{model}.pkl'),'rb'))
            y_dev = pickle.load(open(Path(path,random_seed,'y_true_dev.pkl'),'rb'))
            
            scorings = np.empty(outputs.shape[0])

            for i in range(outputs.shape[0]):
                scorings_i = np.empty(outputs.shape[1])
                for r in range(outputs.shape[1]):
                    if problem_type[project_name] == 'clf':
                        metrics, y_pred = get_metrics_clf(outputs[i,r], y_dev[r], [scoring[project_name]], cmatrix)
                        scorings_i[r] = metrics[scoring[project_name]]
                    else:
                        metrics = get_metrics_reg(outputs[i,r], y_dev[r],[scoring[project_name]])
                        scorings_i[r] = metrics[scoring[project_name]]
                scorings[i] = np.nanmean(scorings_i)
            
            scorings = scorings if any(x in scoring[project_name] for x in ['norm','error']) else -scorings

            best_models = np.argsort(scorings)[:n_models]
            all_models = all_models.iloc[best_models].reset_index(drop=True)
            all_models['idx'] = best_models
            outputs = outputs[best_models]
            
            outputs_bootstrap = np.empty((n_boot,) + outputs.shape)
            y_dev_bootstrap = np.empty((n_boot,) + y_dev.shape)
            y_pred_bootstrap = np.empty((n_boot,)+outputs.shape) if problem_type[project_name] == 'reg' else np.empty((n_boot,)+outputs.shape[:-1])
            
            metrics = dict((metric,np.empty((len(all_models),outputs.shape[1],n_boot))) for metric in metrics_names[project_name])
            
            all_results = Parallel(n_jobs=-1)(delayed(compute_metrics)(model_index, r, outputs, y_dev, metrics_names, n_boot, problem_type, project_name) for model_index in tqdm.tqdm(range(outputs.shape[0])) for r in range(outputs.shape[1]))

            # Update the metrics array with the computed results
            for model_index, r, metrics_result in all_results:
                for metric in metrics_names[project_name]:
                    metrics[metric][model_index, r, :] = metrics_result[metric]

            # Update the summary statistics in all_models
            for model_index in tqdm.tqdm(range(outputs.shape[0])):
                for metric in metrics_names[project_name]:
                    all_models.loc[model_index, f'{metric}_mean'] = np.nanmean(metrics[metric][model_index].flatten()).round(2)
                    all_models.loc[model_index, f'{metric}_inf'] = np.nanpercentile(metrics[metric][model_index].flatten(), 2.5).round(2)
                    all_models.loc[model_index, f'{metric}_sup'] = np.nanpercentile(metrics[metric][model_index].flatten(), 97.5).round(2)
            all_models.to_csv(Path(path,random_seed,f'all_models_{model}_dev_bca.csv'))

            #pickle.dump(outputs_bootstrap,open(Path(path,random_seed,f'outputs_bootstrap_{model}.pkl'),'wb'))
            #pickle.dump(y_dev_bootstrap,open(Path(path,random_seed,f'y_dev_bootstrap_{model}.pkl'),'wb'))
            #pickle.dump(y_pred_bootstrap,open(Path(path,random_seed,f'y_pred_bootstrap_{model}.pkl'),'wb'))
            pickle.dump(metrics,open(Path(path,random_seed,f'metrics_bootstrap_{model}.pkl'),'wb'))