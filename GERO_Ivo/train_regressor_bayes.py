import numpy as np
import pandas as pd
from pathlib import Path

import scipy.stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge as RR
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.model_selection import train_test_split,LeavePOut,StratifiedShuffleSplit,KFold
from xgboost import XGBRegressor as xgboost
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm
import itertools,pickle,sys, json
from scipy.stats import loguniform, uniform, randint
from random import randint as randint_random 
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

n_iter = 20
scaler_name = 'StandardScaler'
scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
shuffle_labels = False
metrics_names = ['r2_score','mean_absolute_error','mean_squared_error']
l2ocv = False
n_boot = 100

id_col = 'Codigo'

tasks = ['animales','letra_p','animales__letra_p'] 

y_labels = ['Grupo']

test_size = 0

config = {'n_iter':n_iter,
          'test_size':test_size,
          'bootstrap':n_boot}

n_seeds_test = 1
random_seeds_test = np.arange(n_seeds_test)

n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

random_seeds_train = np.arange(n_seeds_train)

CV_type = KFold(n_splits=n_folds)

models_dict = {
    #'xgb':xgboost,
    'ridge':RR,
    'knn':KNN,
    'svm':SVR,
    }

base_dir = Path(Path(__file__).parent,'data')

path_to_data = base_dir

data_features = pd.read_excel(Path(path_to_data,f'psych_granularity.xlsx'))

neuro_data = pd.read_excel(Path(path_to_data,f'Neuropsico_features_GERO.xlsx'))

data = pd.merge(data_features,neuro_data,on='Codigo',how='inner')

for task in tasks:    
    y_labels = [col for col in neuro_data.columns if col != 'Grupo' and col != id_col]
    for y_label in y_labels:
        if shuffle_labels:
            data[y_label] = pd.Series(np.random.permutation(data[y_label]))

        y = data.pop(y_label).astype(float)

        ID = data.pop(id_col)

        if '__' in task:
            features = [col for col in data.columns if any([f'{t}_' in col for t in task.split('__')])]
        else:
            features = [col for col in data.columns if f'{task}_' in col]

        #impute mising data
        imputer = KNNImputer(n_neighbors=5)
        data = pd.DataFrame(imputer.fit_transform(data[features]),columns=features)
                
        for model in models_dict.keys():        
            if l2ocv:
                n_folds = int(data.shape[0]*test_size-2)
            
            path_to_save = Path(Path(__file__).parent,task,'properties',scaler_name,kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'hyp_opt','bayes','feature_selection')
            path_to_save = Path(path_to_save,'bootstrap') if n_boot and 'bootstrap' not in str(path_to_save) else path_to_save
            path_to_save = Path(str(path_to_save).replace(f'{n_seeds_test}_seeds_test','')) if test_size == 0 else path_to_save
            
            
            if Path(path_to_save / f'all_outputs_bootstrap_{model}.pkl').exists():
                continue
            
            path_to_save.mkdir(parents=True,exist_ok=True)

            with open(Path(path_to_save,'config.json'),'w') as f:
                json.dump(config,f)
            
            hyperp = {'ridge': {'alpha': Real(1e-3,1e3,'log-uniform'),
                                'tol':Real(.0001,.1,'log-uniform'),
                                'solver':Categorical(['auto'])},
                        'knn': {'n_neighbors':Integer(1,15)},
                        'svm': {'C': Real(1e-3,1e3,'log-uniform'),
                                'gamma': Real(1e-3,1e3,'log-uniform'),
                                'kernel':Categorical(['rbf','linear','sigmoid','poly'])},
                        'xgb': {'n_estimators':Integer(10,1000),
                                        'max_depth':Integer(1,10),
                                        'learning_rate':Real(1e-4,10,'log-uniform')}
                    }
        
            for random_seed_test in random_seeds_test:
                if test_size > 0:
                    X_train,X_test,y_train,y_test,ID_train,ID_test = train_test_split(data,y,ID,test_size=test_size,random_state=random_seed_test)            
                    X_train.reset_index(drop=True,inplace=True)
                    X_test.reset_index(drop=True,inplace=True)
                    y_train.reset_index(drop=True,inplace=True)
                    y_test.reset_index(drop=True,inplace=True)
                    ID_train.reset_index(drop=True,inplace=True)
                    ID_test.reset_index(drop=True,inplace=True)

                else:
                    X_train = data
                    y_train = y
                    ID_train = ID

                    X_test = pd.DataFrame()
                    y_test = pd.Series()
                    ID_test = pd.Series()

                all_models,best_models,all_outputs_bootstrap,outputs_bootstrap_best,all_y_pred_bootstrap,y_pred_bootstrap_best,all_metrics_bootstrap,metrics_bootstrap_best,y_true_bootstrap,IDs_val_bootstrap = nestedCVT_bayes(models_dict[model],scaler,X_train,y_train,n_iter,CV_type,random_seeds_train,hyperp[model],metrics_names,ID_train,n_boot=n_boot,scoring='r2',problem_type='reg')

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