from pathlib import Path
import pickle
from sklearn.model_selection import KFold 
import sys
import numpy as np
from pingouin import rm_anova,anova
import pandas as pd
import pickle,itertools

sys.path.append(str(Path(Path.home(),'scripts_generales')))

from utils import *

tasks = [
        'animales','fas',
         #'letra_f','letra_a','letra_s'
         ]

l2ocv = False
n_seeds_train = 10

random_seeds_train = range(n_seeds_train)
if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

hyp_opt = True
n_seeds_test = 1

feature_selection = True
bootstrap = True

random_seeds_test = [0]

best_models = dict((task,dict()) for task in tasks)
metrics_best_models = dict((task,dict()) for task in tasks)
metric_name = 'mean_absolute_error'

results_dir = Path(Path.home(),'results','GERO_Ivo') if 'Users/gp' in str(Path.home()) else Path('D:','results','GERO_Ivo')

for task in tasks:
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    metrics_best_models[task] = dict((dimension,dict()) for dimension in dimensions)
    best_models[task] = dict((dimension,dict()) for dimension in dimensions)

    for d,dimension in enumerate(dimensions):

        print(task,dimension)
        path = Path(results_dir,task,dimension,'StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test' if len(random_seeds_test) > 0 else '')
        y_labels = [folder.name for folder in path.iterdir() if folder.is_dir()]
        
        y_labels = ['MMSE_Total_Score']
        metrics_best_models[task][dimension] = dict((y_label,dict()) for y_label in y_labels)
        best_models[task][dimension] = dict((y_label,dict()) for y_label in y_labels)
        for y_label in y_labels:

            path_to_data = Path(path,y_label,'hyp_opt','feature_selection','bootstrap')

            for random_seed_test in random_seeds_test:
                file_name = [file for file in Path(path_to_data,f'random_seed_{random_seed_test}').iterdir() if 'metrics_bootstrap_best_model' in file.stem][0]

                metrics_best_models[task][dimension][y_label] = pd.read_csv(file_name)

all_metrics = pd.DataFrame(columns=['bootstrap','task','dimension',f'{metric_name}_{y_label}'])
all_metrics_diff = pd.DataFrame(columns=['task',f'diff_{metric_name}_{y_label}'])

#metrics = np.empty((metrics_best_models[tasks[0]][dimensions[0]][y_labels[0]].shape[0],len(dimensions)))

for task,dimension,y_label in itertools.product(tasks,dimensions,y_labels):
        
    metrics_append = pd.DataFrame({f'{metric_name}_{y_label}':metrics_best_models[task][dimension][y_label][metric_name]})
    metrics_append['task'] = task
    metrics_append['dimension'] = dimension
    metrics_append['bootstrap'] = np.arange(metrics_best_models[task][dimension][y_label].shape[0])
    
    all_metrics = pd.concat([all_metrics,pd.DataFrame(metrics_append)],axis=0)
    
pickle.dump(all_metrics,open(Path(results_dir,'all_metrics.pkl'),'wb'))
