import numpy as np
import pandas as pd
from pathlib import Path

import scipy.stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split,LeavePOut,StratifiedShuffleSplit
from xgboost import XGBClassifier as xgboost
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm
import itertools,pickle,sys, json
from scipy.stats import loguniform, uniform, randint
from random import randint as randint_random 
from joblib import Parallel, delayed
from collections import Counter 

from skopt.space import Real, Integer, Categorical

from random import randint as randint_random 

#sys.path.append(str(Path(Path.home() / 'Doctorado' / 'Codigo' / 'machine_learning')))

sys.path.append(str(Path(Path.home(),'scripts_generales')))

#from machine_learning_module import *
from utils import *

from expected_cost.ec import *
from expected_cost.utils import *

import scipy
#Parámetros

n_iter = 15
scaler_name = 'StandardScaler'
scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
cmatrix = None
feature_importance = True 
shuffle_labels = False
held_out = False
hyp_tuning_list = [True]
feature_selection_list = [True]
metrics_names = ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy']
l2ocv = False
exp_ft = False
n_boot = 100

id_col = 'Codigo'

tasks = ['Animales','P','Animales_P'] 

dimensions = ['properties','timing','properties_timing']

y_labels = ['Grupo']

test_size = .2

config = {'n_iter':n_iter,
          'test_size':test_size,
          'bootstrap':n_boot,
          'cmatrix':str(cmatrix)}

n_seeds_test = 1
n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

random_seeds_train = np.arange(n_seeds_train)

CV_type = LeavePOut(1) if l2ocv else StratifiedKFold(n_splits=n_folds,shuffle=True)

models_dict = {
    'lr':LR,
    'knn':KNN,
    #'lda':LDA,
    'svc':SVC,
    'xgb':xgboost
    }

base_dir = Path(Path(__file__).parent,'data')

path_to_data = base_dir

for y_label,task,dimension in itertools.product(y_labels,tasks,dimensions):
    data = pd.read_excel(Path(path_to_data,f'data_total.xlsx'),sheet_name=dimension)

    if shuffle_labels:
        data[y_label] = pd.Series(np.random.permutation(data[y_label]))

    y = data.pop(y_label).map({'CTR':0,'AD':1})

    ID = data.pop(id_col)

    if '_' in task:
        features = [col for col in data.columns if any([f'{t}_' in col for t in task.split('_')])]
    else:
        features = [col for col in data.columns if f'{task}_' in col]

    #impute mising data
    imputer = KNNImputer(n_neighbors=5)
    data = pd.DataFrame(imputer.fit_transform(data[features]),columns=features)
        
    for feature_selection,model in itertools.product(feature_selection_list,models_dict.keys()):        
        if held_out:
            if l2ocv:
                n_folds = int(data.shape[0]*test_size-2)
        else:
            if l2ocv:
                n_folds = int(data.shape[0]*test_size-2)
            n_seeds_test = 1
            test_size = 0
        
        path_to_save = Path(Path(__file__).parent,task,dimension,scaler_name,'exp_ft',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'hyp_opt','bayes','feature_selection')
        path_to_save = Path(path_to_save,'bootstrap') if n_boot and 'bootstrap' not in str(path_to_save) else path_to_save

        if not exp_ft:
            path_to_save = Path(str(path_to_save).replace('exp_ft','all_features'))

        path_to_save = Path(str(path_to_save).replace('feature_selection','')) if not feature_selection else path_to_save
        
        random_seeds_test = np.arange(n_seeds_test)
        
        n_seeds = n_seeds_train*n_seeds_test
        '''
        if Path(path_to_save / f'all_outputs_bootstrap_{model}.pkl').exists():
           continue
        '''
        path_to_save.mkdir(parents=True,exist_ok=True)

        with open(Path(path_to_save,'config.json'),'w') as f:
            json.dump(config,f)
        
        hyperp = {'lr': {'C':Real(1e-3,1e3,'log-uniform'),
                         'random_state':Integer(42,43)},
                'knn': {'n_neighbors':Integer(1,15)},
                'svc': {'C': Real(1e-3,1e3,'log-uniform'),
                        'gamma': Real(1e-3,1e3,'log-uniform'),
                        'kernel':Categorical(['linear','rbf','poly','sigmoid']),
                        'probability':[True],
                        'random_state':Integer(42,43)},
                'xgb': {'n_estimators':Integer(1,1000),
                        'max_depth':Integer(1,10),
                        'learning_rate':Real(1e-4,10,'log-uniform'),
                        'random_state':Integer(42,43)}
        }
        
        for random_seed_test in random_seeds_test:
            if test_size == 0:
                X_train = data
                y_train = y
                ID_train = ID
                X_test = pd.DataFrame()
                y_test = pd.Series()
                ID_test = pd.Series()
            else:
                sss = StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=random_seed_test)
                for train_index,test_index in sss.split(data,y):
                    X_train = data.iloc[train_index,:].reset_index(drop=True)
                    y_train = y[train_index].reset_index(drop=True)
                    X_test = data.iloc[test_index,:].reset_index(drop=True)
                    y_test = y[test_index].reset_index(drop=True)
                    ID_train = ID[train_index].reset_index(drop=True)
                    ID_test = ID[test_index].reset_index(drop=True)
                        
        all_models,best_models,all_outputs_bootstrap,outputs_bootstrap_best,all_y_pred_bootstrap,y_pred_bootstrap_best,all_metrics_bootstrap,metrics_bootstrap_best,y_true_bootstrap,IDs_val_bootstrap = nestedCVT_bayes(models_dict[model],scaler,X_train,y_train,n_iter,CV_type,random_seeds_train,hyperp[model],metrics_names,ID_train,n_boot=n_boot,cmatrix=None,priors=None,scoring='roc_auc',problem_type='clf')

        with open(Path(path_to_save,f'all_outputs_bootstrap_{model}.pkl'),'wb') as f:
            pickle.dump(all_outputs_bootstrap,f)

        with open(Path(path_to_save,f'outputs_bootstrap_best_{model}.pkl'),'wb') as f:
            pickle.dump(outputs_bootstrap_best,f)

        with open(Path(path_to_save,f'all_y_pred_bootstrap_{model}.pkl'),'wb') as f:
            pickle.dump(all_y_pred_bootstrap,f)

        with open(Path(path_to_save,f'y_pred_bootstrap_best_{model}.pkl'),'wb') as f:
            pickle.dump(y_pred_bootstrap_best,f)
        
        all_models.to_csv(Path(path_to_save,f'all_models_{model}.csv'),index=False)
        best_performance = {}
        for metric in metrics_names:
            inf = np.percentile(metrics_bootstrap_best[metric],2.5)
            sup = np.percentile(metrics_bootstrap_best[metric],97.5)
            mean = np.mean(metrics_bootstrap_best[metric])

            best_performance[f'inf_{metric}'] = inf
            best_performance[f'mean_{metric}'] = mean
            best_performance[f'sup_{metric}'] = sup
        
        best_performance = pd.DataFrame(best_performance,index=[0]).to_csv(Path(path_to_save,f'best_performance_{model}.csv'),index=False)
        best_models.to_csv(Path(path_to_save,f'best_models_{model}.csv'),index=False)
