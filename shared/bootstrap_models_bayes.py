import pandas as pd
import pickle
from pathlib import Path
from expected_cost.utils import *
import itertools
from joblib import Parallel, delayed
import sys,tqdm,json

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

def get_metrics_bootstrap(samples, targets, metrics_names, random_state, stratify=None, cmatrix=None, priors=None, problem_type='clf'):
    assert samples.shape[0] == targets.shape[0]
    indices = np.arange(targets.shape[0])

    sel_indices = resample(indices, replace=True, n_samples=len(samples), stratify=stratify,random_state=random_state)
    
    if problem_type == 'clf':
        metrics, y_pred = get_metrics_clf(samples[sel_indices], targets[sel_indices], metrics_names, cmatrix, priors)
    else:
        metrics = get_metrics_reg(samples[sel_indices], targets[sel_indices], metrics_names)
        y_pred = samples[sel_indices]
    
    return samples[sel_indices],targets[sel_indices],y_pred,metrics

##---------------------------------PARAMETERS---------------------------------##
project_name = 'GeroApathy'
l2ocv = False

n_boot = 1000

cmatrix = None
shuffle_labels = False
hyp_opt_list = [True]
feature_selection_list = [True]

id_col = 'id'
scaler_name = 'StandardScaler'

models = {'MCI_classifier':['lr','svc','knnc','xgb'],
          'tell_classifier':['lr','svc','knnc','xgb'],
          'Proyecto_Ivo':['lr','svc','knnr','xgb'],
          'GeroApathy':['elastic','lasso','ridge','knnr','svr','xgb'],
            'GERO_Ivo':['elastic','lasso','ridge','knnr','svr']
            }

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['Animales','P','Animales__P','cog','brain','AAL','conn'],
         'GeroApathy':['Fugu'],
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
                'GeroApathy':'reg'}

metrics_names = {'MCI_classifier':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'tell_classifier':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'Proyecto_Ivo':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'GeroApathy':['r2_score','mean_squared_error','mean_absolute_error'],
                    'GERO_Ivo':['r2_score','mean_squared_error','mean_absolute_error']}

y_labels = {'MCI_classifier':['target'],
            'tell_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GERO_Ivo':['MMSE_Total_Score'],
            'GeroApathy':['DASS_21_Depression','DASS_21_Anxiety','DASS_21_Stress','AES_Total_Score','MiniSea_MiniSea_Total_FauxPas','Depression_Total_Score','MiniSea_emf_total','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total'],
            }

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

conf_int_metrics = pd.DataFrame(columns=['task','dimension','y_label','model_type','metric','inf','mean','sup'])

for feature_selection in feature_selection_list:
    for task,model,y_label,hyp_opt in itertools.product(tasks[project_name],models[project_name],y_labels[project_name],hyp_opt_list):    
        
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
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','bayes','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
            
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
                
                metrics = dict((metric,np.empty((n_boot,outputs.shape[0]))) for metric in metrics_names[project_name])

                for b,r in itertools.product(range(n_boot),range(outputs.shape[0])):
                    outputs_bootstrap[b,r,:], y_dev_bootstrap[b,r,:],y_pred_bootstrap[b,r,:],metrics_ = get_metrics_bootstrap(outputs[r], y_dev[r], metrics_names[project_name],b,stratify=y_dev[r],problem_type=problem_type[project_name])
                    for metric in metrics_names[project_name]:
                        metrics[metric][b,r] = metrics_[metric]
                for metric in metrics_names[project_name]:
                    mean, inf, sup = conf_int_95(metrics[metric].squeeze())
                    conf_int_metrics.loc[len(conf_int_metrics.index),:] = [task,dimension,y_label,model,metric,np.round(inf,3),np.round(mean,3),np.round(sup,3)]

                pickle.dump(outputs_bootstrap,open(Path(path,random_seed,f'outputs_bootstrap_best_{model}.pkl'),'wb'))
                pickle.dump(y_dev_bootstrap,open(Path(path,random_seed,f'y_dev_bootstrap.pkl'),'wb'))
                pickle.dump(y_pred_bootstrap,open(Path(path,random_seed,f'y_pred_bootstrap_{model}.pkl'),'wb'))
                pickle.dump(metrics,open(Path(path,random_seed,f'metrics_bootstrap_{model}.pkl'),'wb'))

    conf_int_metrics.to_csv(Path(results_dir,'metrics_feature_selection_dev.csv' if feature_selection else 'metrics_dev.csv'),index=False)