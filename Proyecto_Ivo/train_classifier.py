import numpy as np
import pandas as pd
from pathlib import Path

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
from random import randint

#sys.path.append(str(Path(Path.home() / 'Doctorado' / 'Codigo' / 'machine_learning')))

sys.path.append(str(Path(Path.home(),'scripts_generales')))

#from machine_learning_module import *
from utils import *

from expected_cost.ec import *
from expected_cost.utils import *

#Parámetros

n_iter = 50
n_iter_features = 50
scaler_name = 'StandardScaler'
scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
cmatrix = None
feature_importance = True 
shuffle_labels = False
held_out_default = False
hyp_tuning_list = [True]
metrics_names = ['roc_auc','accuracy','recall','f1','norm_cross_entropy']
l2ocv = False
exp_ft = False
n_boot = 100

id_col = 'Codigo'

tasks = ['conn'] 

dimensions = {'cog':['neuropsico','neuropsico_mmse'],
              'brain':['norm_brain_lit'],
              'AAL':['norm_AAL'],
              'conn':['connectivity'],
              'Animales':['properties','timing','properties_timning'],
                'P':['properties','timing','properties_timing'],
                'Animales_P':['properties','timing','properties_timing']
}

y_labels = ['Grupo']

test_size = 0

config = {'n_iter':n_iter,
          'test_size':test_size,
          'bootstrap':n_boot,
          'n_feature_sets': n_iter_features,
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
    'svc':SVC,
    'xgb':xgboost
    }

base_dir = Path(Path(__file__).parent,'data')

path_to_data = base_dir

for y_label,task in itertools.product(y_labels,tasks):
    for dimension in dimensions[task]:
        print(task,dimension)
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
        
        feature_sets = [np.unique(np.random.choice(features,len(features),replace=True)) for _ in range(n_iter_features)]
        feature_sets.append(features)
        
        for hyp_tuning,model in itertools.product(hyp_tuning_list,models_dict.keys()):     
            print(model)   
            held_out = True if hyp_tuning else held_out_default

            if held_out:
                if l2ocv:
                    n_folds = int(data.shape[0]*test_size-2)
            else:
                if l2ocv:
                    n_folds = int(data.shape[0]*test_size-2)
                n_seeds_test = 1
            
            path_to_save = Path(Path(__file__).parent,task,dimension,scaler_name,'exp_ft',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'no_hyp_opt','feature_selection')
            path_to_save = Path(path_to_save,'bootstrap') if n_boot and 'bootstrap' not in str(path_to_save) else path_to_save

            if not exp_ft:
                path_to_save = Path(str(path_to_save).replace('exp_ft','all_features'))

            path_to_save = Path(str(path_to_save).replace('no_hyp_opt','hyp_opt')) if hyp_tuning else path_to_save
            path_to_save = Path(str(path_to_save).replace('feature_selection','')) if n_iter_features == 0 else path_to_save
            
            random_seeds_test = np.arange(n_seeds_test)
            
            n_seeds = n_seeds_train*n_seeds_test
            
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
                    new_combination['lda'] = {'solver':np.random.choice(['lsqr', 'eigen', 'svd'])}
                    new_combination['svc'] = {'C': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))]),
                                            'kernel': np.random.choice(['linear', 'rbf', 'sigmoid']),
                                            'gamma': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))]),
                                            'probability': True}
                    new_combination['knn'] = {'n_neighbors': randint(1, int((n_folds - 1) / n_folds * (data.shape[0] * test_size)))} if test_size > 0 else {'n_neighbors': randint(1,int((n_folds - 1) / n_folds * data.shape[0]))}
                    new_combination['xgb'] = {'n_estimators': randint(10, 1000),
                                            'max_depth': randint(1, 10),
                                            'learning_rate': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-4, 1))])
                                            }
                    
                    for key in models_dict.keys():
                        hyperp[key].loc[len(hyperp[key].index),:] = new_combination[key]
            
                hyperp[model].drop_duplicates(inplace=True)
                hyperp[model] = hyperp[model].reset_index(drop=True)
            if model == 'knn':
                hyperp[model] = hyperp[model].astype(int)
            elif model == 'xgb':
                hyperp[model] = hyperp[model].astype({'n_estimators':int,'max_depth':int})
            
            feature_sets = [np.unique(np.random.choice(features,len(features),replace=True)) for _ in range(n_iter_features)]
            feature_sets.append(features)
            
            for random_seed_test in random_seeds_test:
                if test_size > 0:
                    path_to_save_final = Path(path_to_save,f'random_seed_{random_seed_test}')

                    sss = StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=random_seed_test)
                    for train_index,test_index in sss.split(data,y):
                        X_train = data.iloc[train_index,:].reset_index(drop=True)
                        y_train = y[train_index].reset_index(drop=True)
                        X_test = data.iloc[test_index,:].reset_index(drop=True)
                        y_test = y[test_index].reset_index(drop=True)
                        ID_train = ID[train_index].reset_index(drop=True)
                        ID_test = ID[test_index].reset_index(drop=True)
                else:
                    X_train = data
                    y_train = y
                    ID_train = ID
                    X_test = pd.DataFrame()
                    y_test = pd.Series()
                    ID_test = pd.DataFrame()
                    path_to_save_final = path_to_save
                
                path_to_save_final.mkdir(parents=True,exist_ok=True)
                
                if Path(path_to_save_final / f'all_performances_{model}.csv').exists():
                    continue
                
                with open(Path(path_to_save_final,'config.json'),'w') as f:
                    json.dump(config,f)
                    
                models,outputs_bootstrap,y_pred_bootstrap,metrics_bootstrap,y_dev_bootstrap,IDs_dev_bootstrap,metrics_oob,best_model_index = BBCCV(models_dict[model],scaler,X_train,y_train,CV_type,random_seeds_train,hyperp[model],feature_sets,metrics_names,ID_train,Path(path_to_save_final,f'nan_models_{model}.json'),n_boot=n_boot,cmatrix=cmatrix,parallel=True,scoring='roc_auc')        

                with open(Path(path_to_save_final,f'outputs_bootstrap_{model}.pkl'),'wb') as f:
                    pickle.dump(outputs_bootstrap,f)
                
                if Path(path_to_save_final,'X_dev_bootstrap.pkl').exists() == False:
                    with open(Path(path_to_save_final,f'y_dev_bootstrap.pkl'),'wb') as f:
                        pickle.dump(y_dev_bootstrap,f)
                    with open(Path(path_to_save_final,f'IDs_dev_bootstrap.pkl'),'wb') as f:
                        pickle.dump(IDs_dev_bootstrap,f)
                    with open(Path(path_to_save_final,'X_dev.pkl'),'wb') as f:
                        pickle.dump(X_train,f)
                    with open(Path(path_to_save_final,'y_dev.pkl'),'wb') as f:
                        pickle.dump(y_train,f)
                    with open(Path(path_to_save_final,'ID_dev.pkl'),'wb') as f:
                        pickle.dump(ID_train,f)
                    
                    if test_size > 0:
                        with open(Path(path_to_save_final,'X_test.pkl'),'wb') as f:
                            pickle.dump(X_test,f)
                        with open(Path(path_to_save_final,'y_test.pkl'),'wb') as f:
                            pickle.dump(y_test,f)
                        with open(Path(path_to_save_final,'ID_test.pkl'),'wb') as f:
                            pickle.dump(ID_test,f)

                best_models_performances = pd.DataFrame()

                for model_index in range(models.shape[0]):
                    model_performance = {}
                    for param in models.keys():
                        model_performance[param] = models.iloc[model_index][param]

                    for metric in metrics_names:
                        inf = np.nanpercentile(metrics_bootstrap[metric][model_index],2.5)
                        sup = np.nanpercentile(metrics_bootstrap[metric][model_index],97.5)
                        model_performance[f'inf_{metric}_bootstrap'] = inf
                        model_performance[f'mean_{metric}_bootstrap'] = np.nanmean(metrics_bootstrap[metric][model_index])
                        model_performance[f'sup_{metric}_bootstrap'] = sup

                        inf = np.nanpercentile(metrics_oob[metric][model_index],2.5)
                        sup = np.nanpercentile(metrics_oob[metric][model_index],97.5)
                        model_performance[f'inf_{metric}_oob'] = inf  
                        model_performance[f'mean_{metric}_oob'] = np.nanmean(metrics_oob[metric][model_index])
                        model_performance[f'sup_{metric}_oob'] = sup

                    best_models_performances = pd.concat([best_models_performances,pd.DataFrame(model_performance,index=[0])],ignore_index=True,axis=0)
                
                best_models_performances = best_models_performances.sort_values(by='inf_roc_auc_bootstrap',ascending=False)
                best_models_performances.to_csv(Path(path_to_save_final,f'all_performances_{model}.csv'),index=False)