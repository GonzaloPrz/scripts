import pandas as pd
import pickle
from pathlib import Path
from expected_cost.utils import *
import itertools
from joblib import Parallel, delayed
import sys,tqdm

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

parallel = True

project_name = 'MCI_classifier'

tasks = [#'fas',
    'animales','fas__animales','grandmean'] 

single_dimensions = [
                     'talking-intervals',
                     'psycholinguistic'
                     ]

scaler_name = 'StandardScaler'

y_labels = ['target']

id_col = 'id'

dimensions = list()

n_boot = 100

best = .05

scoring = 'roc_auc'
extremo = 'sup' if any(x in scoring for x in ['norm','error']) else 'inf'
ascending = True if extremo == 'sup' else False

for ndim in range(1,len(single_dimensions)+1):
    for dimension in itertools.combinations(single_dimensions,ndim):
        dimensions.append('__'.join(dimension))

cmatrix = None
shuffle_labels = False
held_out_default = False
hyp_opt_list = [True]
feature_selection_list = [True]
metrics_names = ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy']

l2ocv = False

n_boot = 100

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

models = ['lr','svc','knn','xgb']

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for y_label,task,dimension,model,hyp_opt,feature_selection in itertools.product(y_labels,tasks,dimensions,models,hyp_opt_list,feature_selection_list):
    print(task,dimension,model)
    path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '')
    random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and folder.name != 'bootstrap']
    if len(random_seeds) == 0:
        random_seeds = ['']
    
    for random_seed in random_seeds:
        all_models = pd.read_csv(Path(path,random_seed,f'all_models_{model}.csv'))
       
        if Path(path,random_seed,f'best_performances_{model}.csv').exists() or not Path(path,random_seed,f'outputs_{model}.pkl').exists():
            continue
       
        metrics = dict((metric,np.empty((n_boot,len(all_models)))) for metric in metrics_names)

        outputs = pickle.load(open(Path(path,random_seed,f'outputs_{model}.pkl'),'rb'))
        y_dev = pickle.load(open(Path(path,random_seed,'y_dev.pkl'),'rb'))
        outputs_bootstrap = np.empty((n_boot,outputs.shape[0],outputs.shape[1],outputs.shape[2]))
        y_dev_bootstrap = np.empty((n_boot,outputs.shape[0],outputs.shape[1]),dtype=y_dev.dtype)

        scorings_dev = pd.DataFrame(columns=[scoring])

        for model_index in all_models.index:
            score, _ = get_metrics_clf(outputs[model_index],y_dev,[scoring],cmatrix,None)

            scorings_dev.loc[model_index,scoring] = score[scoring]

        scorings_dev = scorings_dev.sort_values(by=scoring,ascending=ascending)
        best_models_index = scorings_dev.index[:int(best*len(all_models))]
        
        best_models = all_models.loc[best_models_index]

        if parallel:
            results = Parallel(n_jobs=-1)(
                delayed(lambda b, model_index: (
                    b, 
                    model_index, 
                    create_bootstrap_set(outputs[model_index], y_dev, b,stratify=y_dev)
                ))(b, model_index)
                for b, model_index in itertools.product(range(n_boot), best_models.index)
            )          
            for b,model_index, result in tqdm.tqdm(results):
                #outputs_bootstrap[b,model_index,:,:] = result[0]
                #y_dev_bootstrap[b,model_index,:] = result[1]

                metrics_,y_pred = get_metrics_clf(result[0], result[1], metrics_names, cmatrix, None)

                for metric in metrics_names:
                    metrics[metric][b,model_index] = metrics_[metric]
            
        else:
            for model_index,b in itertools.product(best_models.index,range(n_boot)):
                outputs_bootstrap[b,model_index,:,:], y_dev_bootstrap[b,model_index,:],_ = create_bootstrap_set(outputs[model_index],y_dev,b,stratify=y_dev)

                metrics_,y_pred = get_metrics_clf(outputs_bootstrap[b,model_index,:,:], y_dev_bootstrap[b,model_index,:], metrics_names, cmatrix, None)

                for metric in metrics_names:
                    metrics[metric][b,model_index] = metrics_[metric]
                
        for model_index in best_models.index:
            for metric in metrics_names:
                mean, inf, sup = conf_int_95(metrics[metric][:,model_index])
                best_models.loc[model_index,f'inf_{metric}'] = inf
                best_models.loc[model_index,f'mean_{metric}'] = mean
                best_models.loc[model_index,f'sup_{metric}'] = sup
        best_models.to_csv(Path(path,random_seed,f'best_performances_{model}.csv'))