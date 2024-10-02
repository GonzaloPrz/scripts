import sys
from pathlib import Path
import pandas as pd
import itertools
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import pickle

sys.path.append(str(Path(Path.home(),'scripts_generales')))

from utils import *

tasks = ['MOTOR_LIBRE']

kfold_folder = '10_folds'
n_seeds_train = 10
n_seeds_test = 1

best_classifiers = pd.read_csv(Path(Path(__file__).parent,f'best_classifiers_{kfold_folder}_hyp_opt.csv')) 

tasks = best_classifiers['task'].unique()
dimensions = best_classifiers['dimension'].unique()
random_seeds_test = best_classifiers['random_seed_test'].unique()

y_label = 'group'

models_dict = {'lr':LogisticRegression,'knn':KNeighborsClassifier,'svc':SVC,'xgb':XGBClassifier}

for task,dimension,random_seed_test in itertools.product(tasks,dimensions,random_seeds_test):
    path_to_data = Path(Path(__file__).parent,task,dimension,'StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{len(random_seeds_test)}_seeds_test',y_label,'hyp_opt','feature_selection','bootstrap',f'random_seed_{random_seed_test}')
    path_to_save = Path(Path(__file__).parent,task,dimension,'StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{len(random_seeds_test)}_seeds_test',y_label,'hyp_opt','feature_selection','bootstrap',f'random_seed_{random_seed_test}','final_model')
    path_to_save.mkdir(parents=True,exist_ok=True)

    best_model = best_classifiers[(best_classifiers['task'] == task) & (best_classifiers['dimension'] == dimension) & (best_classifiers['random_seed_test'] == random_seed_test)]['model_type'].values[0]
    
    best_classifier = pd.read_csv(Path(path_to_data,f'all_performances_{best_model}.csv'))

    all_features = [col for col in best_classifier.columns if any(x in col for x in dimension.split('_'))]
    features = [col for col in all_features if best_classifier[col].values[0] == 1]
    params = [col for col in best_classifier.columns if col not in all_features and 'inf' not in col and 'sup' not in col and 'mean' not in col]

    params_dict = {param:best_classifier.loc[0,param] for param in params if str(best_classifier.loc[0,param]) != 'nan'}
    if 'gamma' in params_dict.keys():
        params_dict['gamma'] = float(params_dict['gamma'])
        
    model = Model(models_dict[best_model](**params_dict),StandardScaler())

    X_dev = pickle.load(open(Path(path_to_data,'X_train.pkl'),'rb'))
    y_dev = pickle.load(open(Path(path_to_data,'y_train.pkl'),'rb'))

    model.train(X_dev[features],y_dev)

    trained_model = model.model
    scaler = model.scaler

    pickle.dump(trained_model,open(Path(path_to_save,f'final_model_{task}_{dimension}.pkl'),'wb'))
    pickle.dump(scaler,open(Path(path_to_save,f'scaler_{task}_{dimension}.pkl'),'wb'))


