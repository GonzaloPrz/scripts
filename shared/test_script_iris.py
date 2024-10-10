from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from pathlib import Path
import math 

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,LeavePOut
from xgboost import XGBClassifier as xgboost
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm
import itertools,pickle,sys, json
from scipy.stats import loguniform, uniform, randint
from random import randint as randint_random 
from joblib import Parallel, delayed

from random import randint as randint_random 

#sys.path.append(str(Path(Path.home() / 'Doctorado' / 'Codigo' / 'machine_learning')))

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

from expected_cost.ec import *
from expected_cost.utils import *

X = load_iris()

n_iter = 50
n_iter_features = 5

feature_sample_ratio = .5 
scaler_name = 'StandardScaler'
if scaler_name == 'StandardScaler':
    scaler = StandardScaler
elif scaler_name == 'MinMaxScaler':
    scaler = MinMaxScaler
else:
    scaler = None
imputer = KNNImputer

cmatrix = None
feature_importance = True 
shuffle_labels = False
held_out_default = False
hyp_tuning_list = [True]
metrics_names = ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy']
l2ocv = False

id_col = 'id'

n_boot = 100

test_size = .3

n_seeds_test_ = 1
n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

random_seeds_train = np.arange(n_seeds_train)

models_dict = {
    'lr':LR,
    'svc':SVC,
    #'lda':LDA,
    #'knn':KNN,
    #'xgb':xgboost
    }

project_name = 'iris'
data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)
results_dir = Path(str(data_dir).replace('data','results'))

y_labels = ['target']

for y_label in y_labels:
    data = pd.DataFrame(columns=['iris_x1','iris_x2','iris_x3','iris_x4'],data=X.data)
    data = data[X.target != 0]
    data[y_label] = X.target[X.target != 0]
    ID = np.arange(data.shape[0])

    if shuffle_labels:
        data[y_label] = pd.Series(np.random.permutation(data[y_label]))

    data = data.dropna(axis=1,how='any')

    all_features = [col for col in data.columns if 'iris' in col]
    
    features = all_features
    
    y = data[y_label].map({1:0,2:1})

    #impute mising data
    for hyp_tuning,model in itertools.product(hyp_tuning_list,models_dict.keys()):        
        print(model)
        held_out = True if hyp_tuning else held_out_default

        if held_out:
            if l2ocv:
                n_folds = int((data.shape[0]*(1-test_size)-2)/2)
            n_seeds_test = n_seeds_test_
        else:
            if l2ocv:
                n_folds = int((data.shape[0]-2)/2)
            n_seeds_test = 1
        
        random_seeds_test = np.arange(n_seeds_test)

        CV_type = StratifiedKFold(n_splits=n_folds,shuffle=True)

        path_to_save = Path(results_dir,scaler_name,kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'no_hyp_opt','feature_selection')
        path_to_save = Path(path_to_save,'bootstrap') if n_boot and 'bootstrap' not in str(path_to_save) else path_to_save

        path_to_save = Path(str(path_to_save).replace('no_hyp_opt','hyp_opt')) if hyp_tuning else path_to_save
        path_to_save = Path(str(path_to_save).replace('feature_selection','')) if n_iter_features == 0 else path_to_save
                        
        path_to_save.mkdir(parents=True,exist_ok=True)

        config = {'n_iter':n_iter,
          'test_size':test_size,
          'bootstrap':n_boot,
          'n_feature_sets': n_iter_features,
          'feature_sample_ratio':feature_sample_ratio,
          'cmatrix':str(cmatrix)}
        
        hyperp = {'lr': pd.DataFrame({'C': 1},index=[0]),
                            'lda':pd.DataFrame({'solver':'lsqr'},index=[0]),
                            'knn': pd.DataFrame({'n_neighbors':5},index=[0]),
                            'svc': pd.DataFrame({'C': 1,
                                    'gamma': 'scale',
                                    'kernel':'rbf',
                                    'probability':True},index=[0]),
                            'xgb': pd.DataFrame({'n_estimators':100,
                                    'max_depth':6,
                                    'learning_rate':0.3
                                    },index=[0])
                }
        
        if hyp_tuning:
            for n in range(n_iter):
                new_combination = dict((key,{}) for key in models_dict.keys())
                new_combination['lr'] = {'C': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))])}
                new_combination['svc'] = {'C': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))]),
                                        'kernel': np.random.choice(['linear', 'rbf', 'sigmoid']),
                                        'gamma': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))]),
                                        'probability': True}
                new_combination['knn'] = {'n_neighbors': int(randint(1, int((n_folds - 1) / n_folds * (data.shape[0] * test_size))).rvs())}
                new_combination['xgb'] = {'n_estimators': int(randint(10,1000).rvs()),
                                        'max_depth': randint(1, 10).rvs(),
                                        'learning_rate': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))])
                                        }
                
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
            else:
                X_train = data
                y_train = y
                ID_train = ID

                X_test = pd.DataFrame()
                y_test = pd.Series()
                ID_test = np.empty(0)
                path_to_save_final = path_to_save

            path_to_save_final.mkdir(parents=True,exist_ok=True)
            #assert set(ID_train).intersection(set(ID_test)), "Data leakeage detected between train and test sets!"

            #if Path(path_to_save_final,f'outputs_best_model_{model}.pkl').exists():
            #    continue

            models,outputs_bootstrap,y_pred_bootstrap,metrics_bootstrap,y_dev_bootstrap,IDs_dev_bootstrap,metrics_oob,best_model_index = BBCCV(models_dict[model],scaler,imputer,X_train,y_train,CV_type,random_seeds_train,hyperp[model],feature_sets,metrics_names,ID_train,Path(path_to_save,f'random_seed_{random_seed_test}',f'errors_{model}.json'),n_boot=n_boot,cmatrix=cmatrix,parallel=True,scoring='roc_auc',problem_type='clf')        
        
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
            