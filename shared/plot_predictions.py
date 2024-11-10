import pandas as pd
from pathlib import Path
import numpy as np

project_name = 'GeroApathy'

results_dir = Path(Path.home(),'results',project_name)

scaler = 'StandardScaler'

scoring = 'roc_auc'

n_folds = 5

kfold_folder = f'{n_folds}_folds'

best_classifiers = pd.read_csv(Path(results_dir,f'best_classifiers_{scoring}_{kfold_folder}_{scaler}_hyp_opt_feature_selection.csv'))

tasks = best_classifiers['task'].unique()
dimension = best_classifiers['dimension'].unique()

for task in tasks:
    for dim in dimension:
        model_name = best_classifiers[(best_classifiers['task'] == task) & (best_classifiers['dimension'] == dim)]['model_type'].values[0]
        model_index = best_classifiers[(best_classifiers['task'] == task) & (best_classifiers['dimension'] == dim)]['model_index'].values[0]
        print(f'{task}___{dim}___{model_name}')
        
        model_path = Path(results_dir,task,dim,model_name,f'{model_name}_{model_index}.pkl')