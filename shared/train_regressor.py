import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Ridge as RR
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.model_selection import train_test_split,LeavePOut,KFold
from xgboost import XGBRegressor as xgboost
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm
import itertools,pickle,sys, json
from scipy.stats import loguniform, uniform, randint
from random import randint as randint_random 
import math 

from random import randint as randint_random 

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

from expected_cost.ec import *
from expected_cost.utils import *

import scipy
#Parámetros

cmatrix = None
feature_importance = True 
shuffle_labels = False
held_out_default = False
hyp_tuning_list = [True]
metrics_names = ['r2_score','mean_squared_error','mean_absolute_error']
l2ocv = False
exp_ft = False
n_boot = 100

n_iter = 1
n_iter_features = 1
feature_sample_ratio = .5 

scaler_name = 'StandardScaler'
if scaler_name == 'StandardScaler':
    scaler = StandardScaler
elif scaler_name == 'MinMaxScaler':
    scaler = MinMaxScaler
else:
    scaler = None

imputer = KNNImputer

project_name = 'GERO_Ivo'

single_dimensions = [
                     'properties',
                     'speech_timing',
                     ]

dimensions = single_dimensions

for ndim in range(2,len(single_dimensions)+1):
    for dimension in itertools.combinations(single_dimensions,ndim):
        dimensions.append('__'.join(dimension))

id_col = 'Codigo'

tasks = ['fas__animales','grandmean',
         'animales','fas',
         #'letra_f','letra_a','letra_s'
         ] 

test_size = 0.3

n_seeds_test = 1
random_seeds_test = np.arange(n_seeds_test)

n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

random_seeds_train = np.arange(n_seeds_train)

CV_type = LeavePOut(1) if l2ocv else KFold(n_splits=n_folds)

models_dict = {
    'ridge':RR,
    'knn':KNN,
    'lasso':Lasso,
    }

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

results_dir = Path(str(data_dir).replace('data','results'))

neuro_data = pd.read_excel(Path(data_dir,f'Neuropsico_features_GERO.xlsx'))

scoring = 'mean_absolute_error'

extremo = 'sup' if 'error' in scoring else 'inf'

ascending = True if 'error' in scoring else False

for hyp_tuning,task,dimension in itertools.product(hyp_tuning_list,tasks,dimensions):
    data_features = pd.read_excel(Path(data_dir,f'{dimension}_fas_animales.xlsx'))
    
    #y_labels = [col for col in neuro_data.columns if col != 'Grupo' and col != id_col]
    y_labels = ['MMSE_Total_Score']
    held_out = True if hyp_tuning else held_out_default
    for y_label in y_labels:
        data = pd.merge(data_features,neuro_data,on='Codigo',how='inner')

        print(task,y_label)
        if shuffle_labels:
            data[y_label] = pd.Series(np.random.permutation(data[y_label]))

        y = data[y_label]

        ID = data[id_col]
        
        features = [col for col in data.columns if any([f'{t}_' in col for t in task.split('__')])]
                
        for model in models_dict.keys():        
            print(model)
            if l2ocv:
                n_folds = int(data.shape[0]*test_size-2)
            
            path_to_save = Path(results_dir,task,dimension,scaler_name,kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'hyp_opt','feature_selection')
            path_to_save = Path(path_to_save,'bootstrap') if n_boot and 'bootstrap' not in str(path_to_save) else path_to_save
            path_to_save = Path(str(path_to_save).replace(f'{n_seeds_test}_seeds_test','')) if test_size == 0 else path_to_save
            
            path_to_save.mkdir(parents=True,exist_ok=True)

            config = {'n_iter':n_iter,
                'test_size':test_size,
                'bootstrap':n_boot,
                'n_feature_sets': n_iter_features,
                'feature_sample_ratio':feature_sample_ratio,
                'cmatrix':str(cmatrix)}
            
            with open(Path(path_to_save,'config.json'),'w') as f:
                json.dump(config,f)
            
            hyperp = {'ridge': pd.DataFrame({'alpha': 1,
                                            'tol':.0001,
                                            'solver':'auto',
                                            'random_state':42},index=[0]),
                        'knn': pd.DataFrame({'n_neighbors':5},index=[0]),
                        'lasso': pd.DataFrame({'alpha': 1,
                                            'tol':.0001,
                                            'random_state':42},index=[0]),
                        'svm': pd.DataFrame({'C':1,
                                            'kernel':'rbf',
                                            'gamma':'scale'},index=[0]),
                    }
            
            if hyp_tuning:
                for n in range(n_iter):
                    new_combination = dict((key,{}) for key in models_dict.keys())
                    new_combination['ridge'] = {'alpha': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-3, 2))]),
                                                'tol': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-5, 0))]),
                                                'solver':'auto',
                                                'random_state':42}
                    new_combination['lasso'] = {'alpha': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-3, 2))]),
                                                'tol': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-5, 0))]),
                                                'random_state':42}

                    new_combination['knn'] = {'n_neighbors': randint(1, int((n_folds - 1) / n_folds * (data.shape[0] * test_size))).rvs()} if test_size > 0 else {'n_neighbors': randint(1,int((n_folds - 1) / n_folds * data.shape[0])).rvs()}
                    new_combination['svm'] = {'C': loguniform(1e-1, 1e3).rvs(),
                                            'kernel': np.random.choice(['linear','poly','rbf','sigmoid']),
                                            'gamma': 'scale'}
                    for key in models_dict.keys():
                        hyperp[key].loc[len(hyperp[key].index),:] = new_combination[key]
            
            hyperp[model].drop_duplicates(inplace=True)
            hyperp[model] = hyperp[model].reset_index(drop=True)
            if model == 'knn':
                hyperp[model] = hyperp[model].astype(int)
            elif model == 'xgb':
                hyperp[model] = hyperp[model].astype({'n_estimators':int,'max_depth':int})
            
            num_comb = 0

            for k in range(np.min((int(feature_sample_ratio*data.shape[0]*(1-test_size))-1,len(features)-1))):
                num_comb += math.comb(len(features),k+1)

            feature_sets = list()
            
            if n_iter_features > num_comb:
                for k in range(np.min((int(feature_sample_ratio*data.shape[0]*(1-test_size))-1,len(features)-1))):
                    for comb in itertools.combinations(features,k+1):
                        feature_sets.append(list(comb))
                n_iter_features = len(feature_sets)
            else:
                feature_sets = [np.unique(np.random.choice(features,int(np.sqrt(data.shape[0]*(1-test_size))),replace=True)) for _ in range(n_iter_features)]
            
            feature_sets.append(features)
            
            for random_seed_test in random_seeds_test:
            
                if test_size > 0:
                    path_to_save_final = Path(path_to_save,f'random_seed_{random_seed_test}')

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
                    path_to_save_final = path_to_save

                path_to_save_final.mkdir(parents=True,exist_ok=True)
                
                if Path(path_to_save_final,f'all_performances_{model}.csv').exists():
                    continue
                
                models,outputs_bootstrap,y_pred_bootstrap,metrics_bootstrap,y_dev_bootstrap,IDs_dev_bootstrap,metrics_oob,best_model_index = BBCCV(models_dict[model],scaler,imputer,X_train,y_train,CV_type,random_seeds_train,hyperp[model],feature_sets,metrics_names,ID_train,Path(path_to_save_final,f'nan_models_{model}.json'),n_boot=n_boot,cmatrix=cmatrix,parallel=False,scoring=scoring,problem_type='reg')        

                metrics_bootstrap_json = {metric:metrics_bootstrap[metric][best_model_index] for metric in metrics_names}

                with open(Path(path_to_save_final,f'outputs_best_model_{model}.pkl'),'wb') as f:
                    pickle.dump(outputs_bootstrap[:,:,best_model_index,:],f)

                with open(Path(path_to_save_final,f'metrics_bootstrap_{model}.pkl'),'wb') as f:
                    pickle.dump(metrics_bootstrap,f)

                pd.DataFrame(metrics_bootstrap_json).to_csv(Path(path_to_save_final,f'metrics_bootstrap_best_model_{model}.csv'))

                if Path(path_to_save_final,'X_dev.pkl').exists() == False:
                    with open(Path(path_to_save_final,f'y_dev_bootstrap.pkl'),'wb') as f:
                        pickle.dump(y_dev_bootstrap,f)
                    with open(Path(path_to_save_final,f'IDs_dev_bootstrap.pkl'),'wb') as f:
                        pickle.dump(IDs_dev_bootstrap,f)
                    with open(Path(path_to_save_final,f'X_dev.pkl'),'wb') as f:
                        pickle.dump(X_train,f)
                    with open(Path(path_to_save_final,f'y_dev.pkl'),'wb') as f:
                        pickle.dump(y_train,f)
                    with open(Path(path_to_save_final,f'IDs_dev.pkl'),'wb') as f:
                        pickle.dump(ID_train,f)
                    with open(Path(path_to_save_final,f'X_test.pkl'),'wb') as f:
                        pickle.dump(X_test,f)
                    with open(Path(path_to_save_final,f'y_test.pkl'),'wb') as f:
                        pickle.dump(y_test,f)
                    with open(Path(path_to_save_final,f'IDs_test.pkl'),'wb') as f:
                        pickle.dump(ID_test,f)
                
                models_performances = pd.DataFrame()
                for model_index in range(models.shape[0]):
                    model_performance = {}
                    for param in models.keys():
                        model_performance[param] = models.iloc[model_index][param]

                    for metric in metrics_names:
                        mean, inf, sup = conf_int_95(metrics_bootstrap[metric][model_index])
                        
                        model_performance[f'inf_{metric}_bootstrap'] = inf
                        model_performance[f'mean_{metric}_bootstrap'] = mean
                        model_performance[f'sup_{metric}_bootstrap'] = sup

                        mean, inf, sup = conf_int_95(metrics_oob[metric][model_index])

                        model_performance[f'inf_{metric}_oob'] = inf  
                        model_performance[f'mean_{metric}_oob'] = mean
                        model_performance[f'sup_{metric}_oob'] = sup

                    models_performances = pd.concat([models_performances,pd.DataFrame(model_performance,index=[0])],ignore_index=True,axis=0)
                
                models_performances.to_csv(Path(path_to_save_final,f'all_performances_{model}.csv'),index=False)
                