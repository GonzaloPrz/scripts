import pandas as pd
import numpy as np
from pathlib import Path
import sys

import itertools
from sklearn.linear_model import Ridge as RR
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.preprocessing import StandardScaler
import pickle

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import * 

def new_best(current_best,value,ascending):
    if ascending:
        return value < current_best
    else:
        return value > current_best
    
project_name = 'GeroApathy'
scaler_name = 'StandardScaler'
shuffle_labels = False

metrics_names = ['r2_score','mean_absolute_error']

tasks = ['Fugu']

best_regressors = pd.DataFrame(columns=['task','dimension','target','model_type',
                                        'r2_score_bootstrap_dev',
                                        'r2_score_bootstrap_holdout',
                                         'mean_absolute_error_bootstrap_dev',
                                         'mean_absolute_error_bootstrap_holdout',
                                         'root_mean_squared_error_bootstrap_dev',
                                         'root_mean_squared_error_bootstrap_holdout'
                                         ])

pd.options.mode.copy_on_write = True 

l2ocv = False

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

hyp_opt = True
mean_std = False

feature_selection_list = [True]

scoring = 'r2_score'

extremo = 'sup' if 'error' in scoring else 'inf'

ascending = True if 'error' in scoring else False

models_dict = {'ridge':RR,
               'lasso':Lasso,
               'knn':KNN,
               'svm':SVR
               }

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for feature_selection in feature_selection_list:
    for task in tasks:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            print(task,dimension)
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,'mean_std' if mean_std else '')

            if not path.exists():
                continue
            y_labels = [folder.name for folder in path.iterdir() if folder.is_dir() if 'mean_std' not in folder.name]
            
            for y_label in y_labels:
                path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt' if hyp_opt else 'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
            
                random_seeds_test = [folder.name for folder in path.iterdir() if folder.is_dir() if 'random_seed' in folder.name]

                if len(random_seeds_test) == 0:
                    random_seeds_test = ['']
                for random_seed_test in random_seeds_test:
                    files = [file for file in Path(path,random_seed_test).iterdir() if 'all_models_r2_score' in file.stem and 'test' in file.stem]
                    
                    if len(files) == 0:
                        files = [file for file in Path(path,random_seed_test).iterdir() if f'all_models_' in file.stem and 'dev' in file.stem]

                    best = None
                    for file in files:
                        df = pd.read_csv(file)
                        
                        if f'{extremo}_{scoring}' in df.columns:
                            scoring_col = f'{extremo}_{scoring}'
                        else:
                            scoring_col = f'{extremo}_{scoring}_dev'

                        df = df.sort_values(by=scoring_col,ascending=ascending)
                        
                        print(f'{file.stem.split("_")[-2]}:{df.loc[0,scoring_col]}')
                        if best is None:
                            best = df.loc[0,:]
                            
                            model_type = file.stem.split('_')[-2]
                            best['model_type'] = model_type
                            best['model_index'] = df.index[0]
                            best_file = file
                        else:
                            if new_best(best[scoring_col],df.loc[0,scoring_col],ascending):
                                best = df.loc[0,:]
                                model_type = file.stem.split('_')[-2]
                                best['model_type'] = model_type
                                best['model_index'] = df.index[0]
                                best_file = file
                    if best is None:
                            continue
            
                    print(best['model_type'])
                    for metric in metrics_names:
                        try:
                            best[f'{metric}_dev'] = f'[{np.round(best[f"inf_{metric}_dev"],2)}, {np.round(best[f"mean_{metric}_dev"],2)}, {np.round(best[f"sup_{metric}_dev"],2)}]'
                        except:
                            best[f'{metric}_dev'] = f'[{np.round(best[f"inf_{metric}"],2)}, {np.round(best[f"mean_{metric}"],2)}, {np.round(best[f"sup_{metric}"],2)}]'

                        best[f'{metric}_holdout'] = np.nan
                        try:
                            mean = np.round(best[f'mean_{metric}_test'],2)
                            inf = np.round(best[f'inf_{metric}_test'],2)
                            sup = np.round(best[f'sup_{metric}_test'],2)
                            best[f'{metric}_holdout'] = f'[ {inf}, {mean}, {sup}]'
                        except:
                            continue

                    model_type = file
                    
                    dict_append = {'task':task,'dimension':dimension,'y_label':y_label,'model_type':best['model_type'],'model_index':best['model_index'],'random_seed_test':random_seed_test}
                    dict_append.update(dict((f'{metric}_dev',best[f'{metric}_dev']) for metric in metrics_names))
                    dict_append.update(dict((f'{metric}_holdout',best[f'{metric}_holdout']) for metric in metrics_names))

                best_config = pd.read_csv(best_file).iloc[0,:].dropna()
                best_params = [col for col in best_config.index if all([x not in col for x in task.split('__') + ['mean','inf','sup']])]
                best_features = [col for col in best_config.index if any([x in col for x in task.split('__')]) and best_config[col] == 1]

                params_dict = {param:best_config[param] for param in best_params}

                if 'gamma' in params_dict.keys():
                    try:
                        params_dict['gamma'] = float(params_dict['gamma'])
                    except:
                        pass

                best_regressors.loc[len(best_regressors),:] = pd.Series(dict_append)

    filename_to_save = f'best_regressors_{scoring}_{kfold_folder}_{scaler_name}_no_hyp_opt_feature_selection_shuffled.csv'

    if hyp_opt:
        filename_to_save = filename_to_save.replace('no_hyp_opt','hyp_opt')
    if not feature_selection:
        filename_to_save = filename_to_save.replace('_feature_selection','')
    if not shuffle_labels:
        filename_to_save = filename_to_save.replace('_shuffled','')

    best_regressors.dropna(axis=1,inplace=True)
    best_regressors.to_csv(Path(results_dir,filename_to_save),index=False)
