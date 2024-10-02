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

tasks = ['fas__animales','grandmean',
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
outputs_best_models = dict((task,dict()) for task in tasks)

for task in tasks:
    dimensions = [folder.name for folder in Path(Path(__file__).parent,task).iterdir() if folder.is_dir() and 'psycholinguistic' != folder.name]
    best_models[task] = dict((dimension,dict()) for dimension in dimensions)
    outputs_best_models[task] = dict((dimension,dict()) for dimension in dimensions)
    for dimension in dimensions:

        print(task,dimension)
        path = Path(Path(__file__).parent,task,dimension,'StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test' if len(random_seeds_test) > 0 else '')
        y_labels = [folder.name for folder in path.iterdir() if folder.is_dir()]
        
        #y_labels = ['MMSE_Total_Score']
        best_models[task][dimension] = dict((y_label,dict()) for y_label in y_labels)
        outputs_best_models[task][dimension] = dict((y_label,dict()) for y_label in y_labels)
        for y_label in y_labels:

            path_to_data = Path(path,y_label,'hyp_opt','feature_selection','bootstrap')

            for random_seed_test in random_seeds_test:
                file_name = [file for file in Path(path_to_data,f'random_seed_{random_seed_test}').iterdir() if 'best_model' in file.stem and file.suffix == '.pkl'][0]

                best_models[task][dimension][y_label] = pickle.load(open(file_name,'rb'))

y_label = 'MMSE_Total_Score'
metric_name = 'mean_absolute_error'

all_metrics = pd.DataFrame(columns=['task','dimension',f'{metric_name}_{y_label}'])
all_metrics_diff = pd.DataFrame(columns=['task',f'diff_{metric_name}_{y_label}'])

for task in tasks:
    models_to_compare = {'properties':best_models[task]['properties'][y_label],
                         'valid_responses':best_models[task]['valid_responses'][y_label]
                        }

    for random_seed_test in random_seeds_test:
        IDs_dev = pickle.load(open(Path(Path(__file__).parent,task,'properties','StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test' if len(random_seeds_test) > 0 else '',y_label,'hyp_opt','feature_selection','bootstrap',f'random_seed_{random_seed_test}','IDs_dev.pkl'),'rb'))
        
        X_dev = {'properties':pickle.load(open(Path(Path(__file__).parent,task,'properties','StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test' if len(random_seeds_test) > 0 else '',y_label,'hyp_opt','feature_selection','bootstrap',f'random_seed_{random_seed_test}','X_dev.pkl'),'rb')),
                 'valid_responses':pickle.load(open(Path(Path(__file__).parent,task,'valid_responses','StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test' if len(random_seeds_test) > 0 else '',y_label,'hyp_opt','feature_selection','bootstrap',f'random_seed_{random_seed_test}','X_dev.pkl'),'rb'))}

        y_dev = pickle.load(open(Path(Path(__file__).parent,task,'properties','StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test' if len(random_seeds_test) > 0 else '',y_label,'hyp_opt','feature_selection','bootstrap',f'random_seed_{random_seed_test}','y_dev.pkl'),'rb'))

        iterator = KFold(n_splits=10)

        metrics = compare(models_to_compare,X_dev,y_dev,iterator,random_seeds_train,[metric_name],IDs_dev,n_boot=100,problem_type='reg')

        metrics_append = {'task':task,'dimension':'properties',f'{metric_name}_{y_label}':metrics[:,0]}
        all_metrics = pd.concat([all_metrics,pd.DataFrame(metrics_append)],axis=0)

        metrics_append = {'task':task,'dimension':'valid_responses',f'{metric_name}_{y_label}':metrics[:,1]}
        all_metrics = pd.concat([all_metrics,pd.DataFrame(metrics_append)],axis=0)

pickle.dump(all_metrics,open(Path(Path(__file__).parent,'all_metrics.pkl'),'wb'))
