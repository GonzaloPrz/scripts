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

project_name = 'Proyecto_Ivo'
l2ocv = False

models = ['lr','svc','knn','xgb']

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['Animales','P','Animales_P','cog','brain','AAL','conn']}

single_dimensions = {'tell_classifier':['voice-quality',
                                        'talking-intervals','pitch'
                                        ],
                     'MCI_classifier':['talking-intervals','psycholinguistic'],
                     'Proyecto_Ivo':[]}

n_boot = 10

cmatrix = None
shuffle_labels = False
held_out_default = False
hyp_opt_list = [True]
feature_selection_list = [True]
metrics_names = ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy']

y_labels = ['target']
id_col = 'id'
scaler_name = 'StandardScaler'

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for y_label,task,model,hyp_opt,feature_selection in itertools.product(y_labels,tasks[project_name],models,hyp_opt_list,feature_selection_list):    
    
    dimensions = list()

    for ndim in range(1,len(single_dimensions[project_name])+1):
        for dimension in itertools.combinations(single_dimensions[project_name],ndim):
            dimensions.append('__'.join(dimension))

    if len(dimensions) == 0:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

    for dimension in dimensions:
        print(task,model,dimension)
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '')
        random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir()]
        if len(random_seeds) == 0:
            random_seeds = ['']
        
        for random_seed in random_seeds:
            all_models = pd.read_csv(Path(path,random_seed,f'all_models_{model}.csv'))
        
            if Path(path,random_seed,f'all_models_{model}_dev.csv').exists() or not Path(path,random_seed,f'outputs_{model}.pkl').exists():
                continue
        
            outputs = pickle.load(open(Path(path,random_seed,f'outputs_{model}.pkl'),'rb'))
            y_dev = pickle.load(open(Path(path,random_seed,'y_true_dev.pkl'),'rb'))
            outputs_bootstrap = np.expand_dims(np.empty(outputs.shape),axis=0)
            y_dev_bootstrap = np.expand_dims(np.empty(y_dev.shape,dtype=y_dev.dtype),axis=0)

            metrics = dict((metric,np.empty((n_boot,len(all_models),outputs.shape[1]))) for metric in metrics_names)

            if parallel:
                results = Parallel(n_jobs=-1)(
                    delayed(lambda b, model_index,r: (
                        b, 
                        model_index, r,
                        create_bootstrap_set(outputs[model_index,r], y_dev[r], b,stratify=y_dev[r])
                    ))(b, model_index,r)
                    for b, model_index,r in itertools.product(range(n_boot), all_models.index,range(outputs.shape[1]))
                )          
                for b,model_index, r, result in tqdm.tqdm(results):

                    metrics_,y_pred = get_metrics_clf(result[0], result[1], metrics_names, cmatrix, None)

                    for metric in metrics_names:
                        metrics[metric][b,model_index,r] = metrics_[metric]
                
            else:
                for b,model_index,r in itertools.product(range(n_boot),all_models.index,range(outputs.shape[1])):
                    outputs_bootstrap[b,model_index,r], y_dev_bootstrap[b,model_index,r],_ = create_bootstrap_set(outputs[model_index,r],y_dev[r],b,stratify=y_dev[r])

                    metrics_,y_pred = get_metrics_clf(outputs_bootstrap[b,model_index,r], y_dev_bootstrap[b,model_index,r], metrics_names, cmatrix, None)

                    for metric in metrics_names:
                        metrics[metric][b,model_index,r] = metrics_[metric]
                    
            for model_index in all_models.index:
                for metric in metrics_names:
                    mean, inf, sup = conf_int_95(metrics[metric][:,model_index,:].squeeze())
                    all_models.loc[model_index,f'inf_{metric}'] = inf
                    all_models.loc[model_index,f'mean_{metric}'] = mean
                    all_models.loc[model_index,f'sup_{metric}'] = sup
            all_models.to_csv(Path(path,random_seed,f'all_models_{model}_dev.csv'))