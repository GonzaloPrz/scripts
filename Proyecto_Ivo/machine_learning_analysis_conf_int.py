import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as xgboost
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm
import itertools,pickle,sys
from scipy.stats import loguniform, uniform, randint
from random import randint as randint_random 

from random import randint as randint_random 

#sys.path.append(str(Path(Path.home() / 'Doctorado' / 'Codigo' / 'machine_learning')))

sys.path.append(str(Path(Path(__file__).parent.parent.parent,'scripts_generales')))

from machine_learning_module import *

from expected_cost.ec import *
from expected_cost.utils import *

boot_train = False
boot_val = 0
n_iter = 50
scaler_name = 'StandardScaler'
scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
cmatrix = None
shuffle_labels = False
held_out_default = False
feature_selection_list = [True]
hyp_tuning_list = [True]
scoring = 'roc_auc'
y_labels = ['Grupo']
l2ocv = False

test_size = .3

n_seeds_train = 10
n_seeds_test = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

n_seeds = n_seeds_train*n_seeds_test

random_seeds_train = np.arange(n_seeds_train)
random_seeds_test = np.arange(n_seeds_test)

dimensions = ['norm_AAL','AAL']

tasks = ['AAL']

metrics_names = ['accuracy','precision','recall','f1','roc_auc','norm_expected_cost','norm_cross_entropy']

CV_type = StratifiedKFold

models_dict = {
    #'lda':LDA,
    'knn':KNN,
    'lr':LR,
    'svc':SVC,
    #'lightgbm': LGBM,
    'xgb':xgboost,
    }

hyperp = {
        'lr': {'C': 1},
        'lda':{'solver':'lsqr'},
        'knn': {'n_neighbors':5},
        'svc': {'C': 1,
                'gamma': 'scale',
                'kernel':'rbf',
                'probability':True},
        'gnb': {},
        'xgb': {'n_estimators':100,
                'objective': 'binary:logistic',
                'eval_metric':'logloss',
                'max_depth':6,
                'learning_rate':0.3,
                }
        }

for y_label,task,dimension in itertools.product(y_labels,tasks,dimensions):
    print(task,dimension)
    path_to_save = Path(Path(__file__).parent,task,dimension,scaler_name,kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'no_hyp_opt','feature_selection','bootstrap')
    data = pd.read_excel(Path(Path(__file__).parent,'data','data_total.xlsx'),sheet_name=dimension)

    if shuffle_labels:
        data[y_label] = pd.Series(np.random.permutation(data[y_label]))

    y = data.pop(y_label).map({'AD':1,'CTR':0})
    
    if l2ocv:
        n_folds = int(np.floor(data.shape[0]*(1-test_size))/2)
        
    ID = data.pop('Codigo')

    if task == 'both':
        features = [col for col in data.columns if 'Animales_' in col or 'P_' in col]
    else:
        features = [col for col in data.columns if f'{task}_' in col]
    
    #impute mising data
    imputer = KNNImputer(n_neighbors=5)
    data = pd.DataFrame(imputer.fit_transform(data[features]),columns=features)

    for hyp_tuning,feature_selection,model in itertools.product(hyp_tuning_list,feature_selection_list,models_dict.keys()):
        print(model)
        all_results = pd.DataFrame(columns=[param for param in hyperp[model].keys()]+['random_seed_train','random_seed_test','bootstrap']+ metrics_names)
        conf_int_results = pd.DataFrame(columns=[param for param in hyperp[model].keys()] + [f'inf_{metric}' for metric in metrics_names] + [f'mean_{metric}' for metric in metrics_names] + [f'sup_{metric}' for metric in metrics_names] + [f'std_{metric}' for metric in metrics_names])
        selected_features = pd.DataFrame(columns=['random_seed_train','random_seed_test','bootstrap'] + features)

        all_scores = pd.DataFrame(columns = ['random_seed_train','random_seed_test','bootstrap','ID','logpost','y_true','y_pred'])
        
        held_out = True if hyp_tuning or feature_selection else held_out_default

        path_to_save = Path(str(path_to_save).replace('no_hyp_opt','hyp_opt')) if hyp_tuning else path_to_save
        path_to_save = Path(str(path_to_save).replace('feature_selection','')) if feature_selection == False else path_to_save
        path_to_save = Path(str(path_to_save).replace('bootstrap','')) if boot_train == False and boot_val == 0 else path_to_save

        held_out = True if hyp_tuning else held_out_default

        filename_to_save = model

        '''
        if Path(path_to_save / f'all_results_val_{filename_to_save}.xlsx').exists():
            continue
        '''
        if hyp_tuning == False:
            n_iter = 1
            
        i = -1
        for _ in tqdm(range(n_iter)):
            tested = True
            it = 0
            while tested and it < 10:
                it += 1
                if hyp_tuning:
                    
                    hyperp = {'lr': {'C':float(np.random.choice([x*10**y for x in range(1,10) for y in range(-3,2)]))},
                                'lda': {'solver':str(np.random.choice(['lsqr','eigen','svd']))},
                    'svc': {'C': float(np.random.choice([x*10**y for x in range(1,10) for y in range(-3,2)])),
                            'kernel':str(np.random.choice(['linear','rbf','sigmoid'])),
                            'gamma':float(np.random.choice([x*10**y for x in range(1,10) for y in range(-3,2)])),
                            'probability':True},
                    'knn': {'n_neighbors':int(randint(1,np.floor(((n_folds-1)/n_folds)*np.floor(data.shape[0]*(test_size)))).rvs())},
                    'xgb': {'n_estimators':int(randint(10,500).rvs()),
                                'max_depth':int(randint(1,10).rvs()),
                                'learning_rate':float(np.random.choice([x*10**y for x in range(1,9) for y in range(-4,1)]))
                                }
                            }
                    
                tested = False

                tested_hyperparams = conf_int_results[[param for param in hyperp[model].keys()]].copy()

                if tested_hyperparams.empty:
                    continue

                for r, row in tested_hyperparams.iterrows():
                    if all([row[param] == hyperp[model][param] for param in hyperp[model].keys()]):
                        tested = True
                        break
            
            if not tested:
                i += 1
            else:
                continue
            
            selected_features_to_append = pd.DataFrame(columns=list(hyperp[model].keys()) + ['random_seed_test'] + features)

            for random_seed_test in random_seeds_test:

                if held_out:
                    ID_train, ID_test,_,_ = train_test_split(ID,y,test_size=test_size,random_state=random_seed_test,stratify=y)
                    ID_train = ID_train.reset_index(drop=True)
                    ID_test = ID_test.reset_index(drop=True)
                    data_train, data_test, y_train, y_test = train_test_split(data,y,test_size=test_size,random_state=random_seed_test,stratify=y)

                    data_train = data_train.reset_index(drop=True)
                    data_test = data_test.reset_index(drop=True)

                    y_train = y_train.reset_index(drop=True)
                    y_test = y_test.reset_index(drop=True)

                    path_to_save = Path(path_to_save,f'test_{int(test_size*100)}') if 'test' not in str(path_to_save) else path_to_save
                else:
                    ID_train = ID
                    ID_test = pd.Series()
                    data_train = data
                    y_train = y
                    data_test = pd.DataFrame()
                    y_test = np.empty(0)

                n_features_to_select = int(np.floor(data_train.shape[1]/2)) if feature_selection else None

                if 'random_test' in str(path_to_save):
                    path_to_save = Path(path_to_save.parent,f'random_test_{random_seed_test}')
                else:    
                    path_to_save = Path(path_to_save,f'random_test_{random_seed_test}')
                 
                path_to_save.mkdir(parents=True,exist_ok=True)

                df_append = pd.DataFrame({param:[hyperp[model][param]]*np.max((1,boot_val)) for param in hyperp[model].keys()},index=np.arange(boot_val))

                scores_append = pd.DataFrame(columns=['random_seed_test','ID','logpost','y_true','y_pred'])
                #results= train_predict_evaluate(models[model],features_by_task[config][[col for col in features_by_task[config] if feature in col]],y,CV=CV,cmatrix=cmatrix)
                results = train_models(model_type=models_dict[model],scaler=scaler,X_dev=data_train[features],y_dev=y_train,random_seeds=random_seeds_train,
                                    hyperp=hyperp[model],metrics_names=metrics_names,costs=cmatrix,priors=None,
                                    CV_type=CV_type,IDs=ID_train,boot_train=boot_train,boot_val=boot_val,n_features_to_select=n_features_to_select)                       

                for random_seed in random_seeds_train:   
                    df_append['random_seed_train'] = [random_seed]*np.max((1,boot_val))
                    df_append['random_seed_test'] = [random_seed_test]*np.max((1,boot_val))          
                    df_append['bootstrap'] = np.arange(boot_val) if boot_val > 0 else np.nan

                    for metric in metrics_names:
                        df_append[metric] = results['metrics_val'][f'random_seed_{random_seed}'][metric]
                    
                    if all_results.empty:
                        all_results = df_append.copy()
                    else:
                        all_results= pd.concat((all_results,df_append),ignore_index=True,axis=0)

                    if scores_append.empty:
                        scores_append = pd.DataFrame.from_dict({'random_seed_train':[random_seed]*results['raw_logpost_val'][f'random_seed_{random_seed}'].shape[0],
                                                                'random_seed_test':[random_seed_test]*results['raw_logpost_val'][f'random_seed_{random_seed}'].shape[0],
                                                                'ID':results['IDs_val'][f'random_seed_{random_seed}'],
                                                                'raw_logpost':results['raw_logpost_val'][f'random_seed_{random_seed}'][:,1],
                                                                'y_true':results['y_true_val'][f'random_seed_{random_seed}'],
                                                                'y_pred':results['y_pred_val'][f'random_seed_{random_seed}']})
                        
                    else:
                        scores_append = pd.concat((scores_append,pd.DataFrame.from_dict({'random_seed_train':[random_seed]*results['raw_logpost_val'][f'random_seed_{random_seed}'].shape[0],
                                                                                                'random_seed_test':[random_seed_test]*results['raw_logpost_val'][f'random_seed_{random_seed}'].shape[0],
                                                                                                'ID':results['IDs_val'][f'random_seed_{random_seed}'],
                                                                                                'raw_logpost':results['raw_logpost_val'][f'random_seed_{random_seed}'][:,1],
                                                                                                'y_true':results['y_true_val'][f'random_seed_{random_seed}'],
                                                                                                'y_pred':results['y_pred_val'][f'random_seed_{random_seed}']})),axis=0,ignore_index=True)
                    
                    for param in hyperp[model].keys():
                        scores_append[param] = hyperp[model][param]
                    
                    if all_scores.empty:
                        all_scores = scores_append.copy()
                    else:
                        all_scores = pd.concat((all_scores,scores_append),ignore_index=True,axis=0)

                selected_features_to_append['random_seed_test'] = [random_seed_test]*np.max((1,boot_val))

                for param in hyperp[model].keys():
                    selected_features_to_append[param] = [hyperp[model][param]]*np.max((1,boot_val))
                for feature in features:
                    selected_features_to_append[feature] = 1 if results['selected_features'][feature] > 7/10*n_seeds_train else 0

                if selected_features.empty:
                    selected_features = selected_features_to_append.copy()
                else:
                    selected_features = pd.concat((selected_features,selected_features_to_append),ignore_index=True,axis=0)
                
                ID_test.to_csv(Path(path_to_save,f'ID_test.csv'),index=False)
                pickle.dump(data_test,open(Path(path_to_save,f'data_test.pkl'),'wb'))
                pickle.dump(y_test,open(Path(path_to_save,f'y_test.pkl'),'wb'))
                pickle.dump(data_train,open(Path(path_to_save,f'data_train.pkl'),'wb'))
                pickle.dump(y_train,open(Path(path_to_save,f'y_train.pkl'),'wb'))
        
            conf_int_append = hyperp[model].copy()

            for metric in metrics_names:
                conf_int_append[f'inf_{metric}'] = np.nanpercentile(all_results[metric][i*n_seeds*np.max((1,boot_val)):(i+1)*n_seeds*np.max((1,boot_val))],5)
                conf_int_append[f'mean_{metric}'] = np.nanmean(all_results[metric][i*n_seeds*np.max((1,boot_val)):(i+1)*n_seeds*np.max((1,boot_val))])
                conf_int_append[f'sup_{metric}'] = np.nanpercentile(all_results[metric][i*n_seeds*np.max((1,boot_val)):(i+1)*n_seeds*np.max((1,boot_val))],95)
                conf_int_append[f'std_{metric}'] = np.nanstd(all_results[metric][i*n_seeds*np.max((1,boot_val)):(i+1)*n_seeds*np.max((1,boot_val))])

            conf_int_results.loc[conf_int_results.shape[0],:] = conf_int_append

        all_results.to_excel(Path(path_to_save.parent,f'all_results_val_{filename_to_save}.xlsx'),index=False)
        conf_int_results.to_excel(Path(path_to_save.parent,f'conf_int_results_val_{filename_to_save}.xlsx'),index=False)
        all_scores.to_csv(Path(path_to_save.parent,f'all_scores_val_{filename_to_save}.csv'),index=False)
        selected_features.to_csv(Path(path_to_save.parent,f'selected_features_val_{filename_to_save}.csv'),index=False)