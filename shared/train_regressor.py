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

mean_std = True

test_size = .3

project_name = 'GeroApathy'

stats_exclude = ['skewness','kurtosis','min','max'] if mean_std else []

parallel = True

l2ocv = False

data_file = {'GeroApathy':'all_data.csv'}

tasks = {'GeroApathy':['Fugu']}

single_dimensions = {'GeroApathy':['voice-quality','pitch','talking_intervals','mfcc','formants']}

y_labels = {'GeroApathy': ['DASS_21_Depression','DASS_21_Anxiety','DASS_21_Stress','AES_Total_Score','MiniSea_MiniSea_Total_FauxPas','Depression_Total_Score','MiniSea_emf_total','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total']}

dimensions = list()

for ndim in range(1,len(single_dimensions[project_name])+1):
    for dimension in itertools.combinations(single_dimensions[project_name],ndim):
        dimensions.append('__'.join(dimension))

stratify = False

n_iter = 5
n_iter_features = 5

feature_sample_ratio = .5 

scaler_name = 'StandardScaler'

id_col = 'id'

if scaler_name == 'StandardScaler':
    scaler = StandardScaler
elif scaler_name == 'MinMaxScaler':
    scaler = MinMaxScaler
else:
    scaler = None
imputer = KNNImputer

shuffle_labels = False
hyp_tuning_list = [True]

n_seeds_test_ = 1
n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

random_seeds_train = np.arange(n_seeds_train)

id_col = 'id'

tasks = ['Fugu']

n_seeds_test = 1
random_seeds_test = np.arange(n_seeds_test)

n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
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

scoring = 'r2_score'

extremo = 'sup' if 'error' in scoring else 'inf'

ascending = True if 'error' in scoring else False

data = pd.read_csv(Path(data_dir,data_file[project_name])) #CHANGE THIS LINE

for hyp_tuning,task,dimension in itertools.product(hyp_tuning_list,tasks,dimensions):
    held_out = True if hyp_tuning else False
    for y_label in y_labels[project_name]:
        print(task,y_label)
        if shuffle_labels:
            data[y_label] = pd.Series(np.random.permutation(data[y_label]))

        data = data.dropna(subset=[y_label])

        y = data[y_label]

        ID = data[id_col]
        
        features = [col for col in data.columns if any([f'{x}_{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__'))]) and isinstance(data[col][0],(float,int)) and all(x not in col for x in stats_exclude)]
                
        for model in models_dict.keys():        
            print(model)
            if l2ocv:
                n_folds = int(data.shape[0]*test_size-2)
            
            path_to_save = Path(results_dir,task,dimension,scaler_name,kfold_folder,'mean_std',y_label,'hyp_opt','feature_selection')
            path_to_save = Path(str(path_to_save).replace('mean_std','')) if not mean_std else path_to_save
            
            path_to_save.mkdir(parents=True,exist_ok=True)

            config = {'n_iter':n_iter,
                'test_size':test_size,
                'stratify':stratify,
                'n_feature_sets': n_iter_features,
                'feature_sample_ratio':feature_sample_ratio}
            
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
                feature_sets = [np.unique([np.random.choice(features,int(feature_sample_ratio*data.shape[0]*(1-test_size)),replace=True) for _ in range(n_iter_features)])]
            
            feature_sets.append(features)
            
            #Drop duplicate feature sets:
            feature_sets = [list(x) for x in set(tuple(x) for x in feature_sets)]

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
                
                with open(Path(path_to_save_final,'config.json'),'w') as f:
                    json.dump(config,f)

                models,outputs,y_pred,y_dev,IDs_dev = CVT(models_dict[model],scaler,imputer,X_train,y_train,CV_type,random_seeds_train,hyperp[model],feature_sets,ID_train,parallel=parallel,problem_type='reg')        
            
                all_models = pd.DataFrame()
                
                for model_index in range(models.shape[0]):
                    model_ = {}
                    for param in models.keys():
                        if param in [y_label,id_col]:
                            continue
                        model_[param] = models.iloc[model_index][param]

                    all_models = pd.concat([all_models,pd.DataFrame(model_,index=[0])],ignore_index=True,axis=0)
                
                all_models.to_csv(Path(path_to_save_final,f'all_models_{model}.csv'),index=False)

                with open(Path(path_to_save_final,f'X_dev.pkl'),'wb') as f:
                    pickle.dump(X_train,f)
                with open(Path(path_to_save_final,f'y_true_dev.pkl'),'wb') as f:
                    pickle.dump(y_dev,f)
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
                
                with open(Path(path_to_save_final,f'outputs_{model}.pkl'),'wb') as f:
                    pickle.dump(outputs,f)