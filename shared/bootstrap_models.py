import pandas as pd
import pickle
from pathlib import Path
from expected_cost.utils import *
import itertools
from joblib import Parallel, delayed
import sys,tqdm

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
parallel = True

project_name = 'GeroApathy'
l2ocv = False

n_boot = 10

cmatrix = None
shuffle_labels = False
held_out_default = False
hyp_opt_list = [True]
feature_selection_list = [True]

id_col = 'id'
scaler_name = 'StandardScaler'

models = {'MCI_classifier':['lr','svc','knn','xgb'],
          'tell_classifier':['lr','svc','knn','xgb'],
          'Proyecto_Ivo':['lr','svc','knn','xgb'],
          'GeroApathy':['lasso','ridge','knn']
            }

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean' ],
         'Proyecto_Ivo':['Animales','P',
                         'Animales__P',
                         'cog','brain','AAL','conn'
                         ],
         'GeroApathy':['Fugu']}

single_dimensions = {'tell_classifier':['voice-quality','talking-intervals','pitch'],
                     'MCI_classifier':['talking-intervals','psycholinguistic'],
                     'Proyecto_Ivo':[],
                     'GeroApathy':[]}

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GeroApathy':'reg'}

metrics_names = {'MCI_classifier':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'tell_classifier':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'Proyecto_Ivo':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'GeroApathy':['r2_score','mean_squared_error','mean_absolute_error']}

y_labels = {'MCI_classifier':['target'],
            'tell_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GeroApathy':['DASS_21_Depression','DASS_21_Anxiety','DASS_21_Stress','AES_Total_Score','MiniSea_MiniSea_Total_FauxPas','Depression_Total_Score','MiniSea_emf_total','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total']}

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
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
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,'mean_std' if project_name=='GeroApathy' else '',y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
        
        if not path.exists():  
            continue

        random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
        if len(random_seeds) == 0:
            random_seeds = ['']
        
        for random_seed in random_seeds:

            try:
                all_models = pd.read_csv(Path(path,random_seed,f'all_models_{model}.csv'))

                if Path(path,random_seed,f'all_models_{model}_dev.csv').exists():
                    continue
                outputs = pickle.load(open(Path(path,random_seed,f'outputs_{model}.pkl'),'rb'))
                y_dev = pickle.load(open(Path(path,random_seed,'y_true_dev.pkl'),'rb'))
                outputs_bootstrap = np.expand_dims(np.empty(outputs.shape),axis=0)
                y_dev_bootstrap = np.empty((n_boot,outputs.shape[0],outputs.shape[1],outputs.shape[2]),dtype=y_dev.dtype)
                y_pred_bootstrap = np.empty((n_boot,outputs.shape[0],outputs.shape[1],outputs.shape[2]),dtype=y_dev.dtype)

                metrics = dict((metric,np.empty((n_boot,len(all_models),outputs.shape[1]))) for metric in metrics_names[project_name])

                if parallel:
                    results = Parallel(n_jobs=-1)(
                        delayed(lambda b, model_index,r: (
                            b, 
                            model_index, r,
                            get_metrics_bootstrap(outputs[model_index,r], y_dev[r], metrics_names[project_name],b,stratify=y_dev[r],problem_type=problem_type[project_name])
                        ))(b, model_index,r)
                        for b, model_index,r in itertools.product(range(n_boot), all_models.index,range(outputs.shape[1]))
                    )          
                    for b,model_index, r, result in results:
                        for metric in metrics_names[project_name]:
                            metrics[metric][b,model_index,r] = result[3][metric]
                        y_pred_bootstrap[b,model_index,r,:] = result[2]
                else:
                    for b,model_index,r in itertools.product(range(n_boot),all_models.index,range(outputs.shape[1])):
                        outputs_bootstrap[b,model_index,r], y_dev_bootstrap[b,model_index,r],y_pred,metrics_ = get_metrics_bootstrap(outputs[model_index,r],y_dev[r],metrics_names[project_name],b,stratify=y_dev[r])

                        for metric in metrics_names[project_name]:
                            metrics[metric][b,model_index,r] = metrics_[metric]
                        
                        y_pred_bootstrap[b,model_index,r,:] = y_pred

                for model_index in all_models.index:
                    for metric in metrics_names[project_name]:
                        mean, inf, sup = conf_int_95(metrics[metric][:,model_index,:].squeeze())
                        all_models.loc[model_index,f'inf_{metric}'] = inf
                        all_models.loc[model_index,f'mean_{metric}'] = mean
                        all_models.loc[model_index,f'sup_{metric}'] = sup
                all_models.to_csv(Path(path,random_seed,f'all_models_{model}_dev.csv'))

                pickle.dump(outputs_bootstrap,open(Path(path,random_seed,f'outputs_bootstrap_{model}.pkl'),'wb'))
                pickle.dump(y_dev_bootstrap,open(Path(path,random_seed,f'y_dev_bootstrap_{model}.pkl'),'wb'))
                pickle.dump(y_pred_bootstrap,open(Path(path,random_seed,f'y_pred_bootstrap_{model}.pkl'),'wb'))
                pickle.dump(metrics,open(Path(path,random_seed,f'metrics_bootstrap_{model}.pkl'),'wb'))
                
            except Exception as e:
                print(e)
                continue
