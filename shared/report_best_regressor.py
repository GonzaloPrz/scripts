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
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

hyp_opt = True

feature_selection_list = [True]

scoring = 'r2_score'

extremo = 'sup' if 'error' in scoring else 'inf'

ascending = True if 'error' in scoring else False

models_dict = {'ridge':RR,
               'lasso':Lasso,
               'knn':KNN,
               'svm':SVR
               }

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','results',project_name)
for feature_selection in feature_selection_list:
    for task in tasks[project_name]:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            print(task,dimension)
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt' if hyp_opt else 'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
            
            if not path.exists():
                continue

            random_seeds_test = [folder.name for folder in path.iterdir() if folder.is_dir() if 'random_seed' in folder.name]

            if len(random_seeds_test) == 0:
                random_seeds_test = ['']
            for random_seed_test in random_seeds_test:
                    files = [file for file in Path(path_to_data,f'random_seed_{random_seed_test}').iterdir() if 'all_models_r2_score' in file.stem and 'test' in file.stem]
                    
                    if len(files) == 0:
                        files = [file for file in Path(path,random_seed_test).iterdir() if f'all_models_' in file.stem and 'dev' in file.stem]

                    for file in files:
                        df = pd.read_csv(file)
                        
                        if f'{extremo}_{scoring}' in df.columns:
                            scoring_col = f'{extremo}_{scoring}'
                        else:
                            scoring_col = f'{extremo}_{scoring}'

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
                    r2_score_bootstrap = f'[{np.round(best["inf_r2_score_bootstrap"],2)}, {np.round(best["mean_r2_score_bootstrap"],2)}, {np.round(best["sup_r2_score_bootstrap"],2)}]'
                    mean_squared_error_bootstrap = f'[{np.round(np.sqrt(best[f'inf_mean_squared_error_bootstrap']),2)}, {np.round(np.sqrt(best[f"mean_mean_squared_error_bootstrap"]),2)}, {np.round(np.sqrt(best[f"sup_mean_squared_error_bootstrap"]),2)}]'
                    mean_absolute_error_bootstrap = f'[{np.round(best["inf_mean_absolute_error_bootstrap"],2)}, {np.round(best["mean_mean_absolute_error_bootstrap"],2)}, {np.round(best["sup_mean_absolute_error_bootstrap"],2)}]'

                    r2_score_bootstrap_test = 'NA'
                    mean_squared_error_bootstrap_test = 'NA'
                    mean_absolute_error_bootstrap_test = 'NA'

                    if Path(best_file.parent,f'best_10_{best['model_type']}_test.csv').exists():
                        
                        best_test = pd.read_csv(Path(best_file.parent,f'best_10_{best['model_type']}_test.csv')).loc[0,:]
                        
                        r2_score_bootstrap_test = f'[{np.round(best_test["inf_r2_score_bootstrap_test"],2)}, {np.round(best_test["mean_r2_score_bootstrap_test"],2)}, {np.round(best_test["sup_r2_score_bootstrap_test"],2)}]'
                        root_mean_squared_error_bootstrap_test = f'[{np.round(np.sqrt(best_test[f'inf_mean_squared_error_bootstrap_test']),2)}, {np.round(np.sqrt(best_test[f"mean_mean_squared_error_bootstrap_test"]),2)}, {np.round(np.sqrt(best_test[f"sup_mean_squared_error_bootstrap_test"]),2)}]'
                        mean_absolute_error_bootstrap_test = f'[ {np.round(best_test["inf_mean_absolute_error_bootstrap_test"],2)}, {np.round(best_test["mean_mean_absolute_error_bootstrap_test"],2)}, {np.round(best_test["sup_mean_absolute_error_bootstrap_test"],2)}]'
                    
                    best_regressors.loc[len(best_regressors),:] = pd.Series({'task':task,'dimension':dimension,'target':y_label,'model_type':best['model_type'],
                                                                                'r2_score_bootstrap_dev':r2_score_bootstrap,
                                                                                'r2_score_bootstrap_holdout':r2_score_bootstrap_test,
                                                                                'root_mean_squared_error_bootstrap_dev':mean_squared_error_bootstrap,
                                                                                'root_mean_squared_error_bootstrap_holdout':root_mean_squared_error_bootstrap_test,
                                                                                'mean_absolute_error_bootstrap_dev':mean_absolute_error_bootstrap,
                                                                                'mean_absolute_error_bootstrap_holdout':mean_absolute_error_bootstrap_test
                                                                                })
                best_config = pd.read_csv(best_file).iloc[0,:].dropna()
                best_params = [col for col in best_config.index if all([x not in col for x in task.split('__') + ['mean','inf','sup']])]
                best_features = [col for col in best_config.index if any([x in col for x in task.split('__')]) and best_config[col] == 1]

                params_dict = {param:best_config[param] for param in best_params}

                if 'gamma' in params_dict.keys():
                    try:
                        params_dict['gamma'] = float(params_dict['gamma'])
                    except:
                        pass
                    
                #best_model_ = Model(models_dict[best['model_type']](**params_dict),StandardScaler)

                #best_model_.model.feature_names_in_ = best_features 

                #pickle.dump(best_model_,open(Path(best_file.parent,f'best_model_{best['model_type']}.pkl'),'wb'))
                
    filename_to_save = f'best_regressors_{kfold_folder}_hyp_opt.csv' if hyp_opt else f'best_regressors_{kfold_folder}_no_hyp_opt.csv'
    best_regressors.to_csv(Path(results_dir,filename_to_save),index=False)
