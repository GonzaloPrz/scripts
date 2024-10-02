import numpy as np
import pandas as pd
from pathlib import Path
import os 

from sklearn.model_selection import StratifiedKFold
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
import random
import warnings
from lightgbm import LGBMClassifier as LGBM

warnings.filterwarnings("ignore")

from random import randint as randint_random 

#sys.path.append(str(Path(Path.home() / 'Doctorado' / 'Codigo' / 'machine_learning')))

sys.path.append(str(Path(Path.home(),'Proyectos','scripts_generales')))

from machine_learning_module import *

from expected_cost.ec import *
from expected_cost.utils import *

#Parámetros
n_seeds = 10
boot_train = False
boot_val = 20
boot_test = 50
n_folds = 5
n_iter = 50
scaler_name = 'StandardScaler'
scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
cmatrix = None
random_seeds = np.arange(n_seeds)
feature_importance = False 
calibrate_list = [True,False]
cheat = False
feature_selection_list = [False]
shuffle_labels = False
held_out_default = True
hyp_tuning_list = [False]
scoring = 'roc_auc'
y_labels = ['Grupo']

dimensions = ['properties','timing','properties_timing',
         'properties_vr','timing_vr','properties_timing_vr']

tasks = ['Animales','both','P']

metrics_names = ['accuracy','precision','recall','f1','roc_auc','norm_expected_cost','norm_cross_entropy']

models_dict = {
    #'lda':LDA,
    'knn':KNN,
    'lr':LR,
    'svc':SVC,
    #'lightgbm': LGBM,
    'xgboost':xgboost,
    }
#Hyperparameter spaces for random search

for y_label,task,dimension in itertools.product(y_labels,tasks,dimensions):
    data = pd.read_excel(Path(Path(__file__).parent,'data','data_total.xlsx'),sheet_name=dimension)

    if shuffle_labels:
        data[y_label] = pd.Series(np.random.permutation(data[y_label]))

    y = data.pop(y_label).map({'AD':1,'CTR':0})

    ID = data.pop('Codigo')

    if task == 'both':
        features = [col for col in data.columns if 'Animales_' in col or 'P_' in col]
    else:
        features = [col for col in data.columns if f'{task}_' in col]
    
    #impute mising data
    imputer = KNNImputer(n_neighbors=5)
    data = pd.DataFrame(imputer.fit_transform(data[features]),columns=features)

    for hyp_tuning,feature_selection,calibrate,model in itertools.product(hyp_tuning_list,feature_selection_list,calibrate_list,models_dict.keys()):
        
        held_out = True if hyp_tuning else held_out_default

        if hyp_tuning:
            hyperp_spaces = { 
            'lr': {'C': loguniform(1e-5,1e5)},
            'lda': {'solver':['svd','lsqr','eigen']},
            'svc': {'C': loguniform(1e-5,1e5),
                    'gamma': loguniform(1e-5,1e5),
                    'kernel':['linear','rbf','poly','sigmoid']},
            'gnb': {},
            'xgboost': {'n_estimators':randint(10,1000),
                        'max_depth':randint(1,10),
                        'learning_rate':loguniform(1e-5,1e5),
                        'gamma': uniform(1e-5,1)
                        },
            'lightgbm': {'n_estimators':randint(10,1000),
                        'objective': ['binary'],
                        'metric':['binary_logloss'],
                        'boosting_type':['gbdt'],
                        'num_leaves':randint(2,100),
                        'learning_rate':loguniform(1e-5,1e5),
                        'max_depth':randint(1,10),
                        'min_child_samples':randint(1,20),
                        'subsample':uniform(0.5,0.5),
                        'colsample_bytree':uniform(0.5,0.5),
                        'reg_alpha':loguniform(1e-5,1e5),
                        'reg_lambda':loguniform(1e-5,1e5),
                        'n_jobs':[-1]
                        },
            }
        else:
            hyperp_spaces = { 
            'lr': {'C': [1]},
            'lda': {'solver':['svd']},
            'knn': {'n_neighbors':[5]},
            'svc': {'C': [1],
                    'gamma': ['scale'],
                    'kernel':['rbf']},
            'gnb': {},
            'xgboost': {'n_estimators':[100],
                        'objective': ['binary:logistic'],
                        'eval_metric':['logloss'],
                        },
            'lightgbm': {'n_estimators':[100],
                        'objective': ['binary'],
                        'metric':['binary_logloss'],
                        'boosting_type':['gbdt'],
                        'num_leaves':[31],
                        'learning_rate':[0.1],
                        'max_depth':[-1],
                        'min_child_samples':[20],
                        'subsample_for_bin':[200000],
                        'colsample_bytree':[1.0],
                        'reg_alpha':[0.0],
                        'reg_lambda':[0.0],
                        'n_jobs':[-1]
                        }
            }

        if held_out:
            ID_train, ID_test,_,_ = train_test_split(ID,y,test_size=0.15,random_state=0,stratify=y)
            ID_train = ID_train.reset_index(drop=True)
            ID_test = ID_test.reset_index(drop=True)
            data_train, data_test, y_train, y_test = train_test_split(data,y,test_size=0.15,random_state=0,stratify=y)

            data_train = data_train.reset_index(drop=True)
            data_test = data_test.reset_index(drop=True)

            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)
        else:
            ID_train = ID
            ID_test = pd.Series()
            data_train = data
            y_train = y
            data_test = pd.DataFrame()
            y_test = np.empty(0)

        path_to_save = Path(Path(__file__).parent,task,dimension,scaler_name,f'{n_folds}_folds',f'{n_seeds}_seeds',y_label)
        path_to_save = Path(path_to_save,'bt') if boot_train or boot_val or boot_test else path_to_save
    
        path_to_save = Path(path_to_save,f'no_hyp_opt') if not hyp_tuning else Path(path_to_save,f'hyp_opt',scoring)
        
        if model in ['gnb','knn'] and feature_selection:
            continue

        if calibrate and hyp_tuning:
            hyperp_spaces['knn'] = {'n_neighbors':randint(1,np.floor(((n_folds-1)/n_folds)**2*data_train.shape[0]))}
        elif hyp_tuning:
            hyperp_spaces['knn'] = {'n_neighbors':randint(1,np.floor(((n_folds-1)/n_folds)*data_train.shape[0]))}
        
        path_to_save = Path(path_to_save,'held_out') if held_out and 'held_out' not in str(path_to_save) else path_to_save
        path_to_save = Path(path_to_save,'shuffle_labels') if shuffle_labels and 'shuffle_labels' not in str(path_to_save) else path_to_save
        path_to_save = Path(path_to_save,'feature_selection') if feature_selection and 'feature_selection' not in str(path_to_save) else path_to_save
        path_to_save = Path(path_to_save,'calibrated') if calibrate and 'calibrated' not in str(path_to_save) else path_to_save
        path_to_save = Path(path_to_save,'cheat') if cheat and 'cheat' not in str(path_to_save) else path_to_save

        path_to_save.mkdir(parents=True,exist_ok=True)

        filename_to_save = model

        print(f'results {filename_to_save}')
        
        if Path(path_to_save / f'results_val_{filename_to_save}.xlsx').exists():
            continue

        all_results = pd.DataFrame(columns=['random_seed']) 

        IDs_val = pd.DataFrame(columns=['random_seed','ID'])

        if held_out:
            all_results_test = pd.DataFrame(columns=['random_seed'])
            IDs_test = pd.DataFrame(columns=['random_seed','ID'])
            y_true_test = np.empty(0)
            y_pred_test = np.empty(0)
            
            score_seeds_test = np.empty(0)
            
            logpost_test = np.empty((0,2))
            cal_logpost_test = np.empty((0,2))


        score_seeds = np.empty(0)
        
        logpost = np.empty((0,2))
        cal_logpost = np.empty((0,2))

        y_true_val = np.empty(0)
        y_pred_val = np.empty(0)
        
        bootstrap_index = np.empty(0)

        feature_importances = dict([(feature,[]) for feature in features])

        for i,seed in enumerate(tqdm(random_seeds)):
                
            CV = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=seed)
            
            #results= train_predict_evaluate(models[model],features_by_task[config][[col for col in features_by_task[config] if feature in col]],y,CV=CV,cmatrix=cmatrix)
            results = train_tune_calibrate_eval(model=models_dict[model],scaler=scaler,X_dev=data_train[features],y_dev=y_train,X_test=data_test,y_test=y_test,random_seed=seed,hyperp_space=hyperp_spaces[model],n_iter=n_iter,metrics_names=metrics_names,scoring=scoring,costs=cmatrix,priors=None,CV_out=CV,CV_in=CV,cal=calibrate,cheat=cheat,feature_selection=feature_selection,IDs=ID_train,ID_test=ID_test,boot_train=boot_train,boot_val=boot_val,boot_test=boot_test)
            results['IDs_val'].loc[:,'random_seed'] = np.array([seed]*results['IDs_val'].shape[0])

            logpost = np.vstack([logpost,np.array(results['raw_logpost_val'])])
            cal_logpost = np.vstack([cal_logpost,np.array(results['cal_logpost_val'])])
            
            score_seeds = np.hstack([score_seeds,np.array([seed]*results['raw_logpost_val'].shape[0])])
            y_true_val = np.hstack([y_true_val,results['y_true_val']])
            y_pred_val = np.hstack([y_pred_val,results['y_pred_val']])
            
            df_append = pd.DataFrame({'random_seed':[seed]*np.max((1,boot_val))})

            for metric in metrics_names:
                df_append[metric] = results['metrics_val'][metric]

            best_model = results['model'].get_params()
            df_append['best_params'] = str(dict([(param,best_model[param]) for param in hyperp_spaces[model].keys()]))

            all_results = pd.concat([all_results,df_append],axis=0,ignore_index=True)
            IDs_val = pd.concat((IDs_val,results['IDs_val']),axis=0)
            
            if held_out:
                df_append_test = pd.DataFrame({'random_seed':[seed]*np.max((1,boot_test))})
                
                score_seeds_test = np.hstack([score_seeds_test,np.array([seed]*results['raw_logpost_test'].shape[0])])
                logpost_test = np.vstack([logpost_test,np.array(results['raw_logpost_test'])])
                cal_logpost_test = np.vstack([cal_logpost_test,np.array(results['cal_logpost_test'])])
                y_true_test = np.hstack([y_true_test,results['y_true_test']])
                y_pred_test = np.hstack([y_pred_test,results['y_pred_test']])
                for metric in metrics_names:
                    df_append_test[metric] = results['metrics_test'][metric]
               
                df_append_test['best_params'] = str(dict([(param,best_model[param]) for param in hyperp_spaces[model].keys()]))
                
                all_results_test = pd.concat([all_results_test,df_append_test],axis=0,ignore_index=True)
                
                IDs_test = pd.concat((IDs_test,pd.DataFrame({'random_seed':[seed]*results['raw_logpost_test'].shape[0],'ID':results['IDs_test']['ID']})),axis=0)
            
            # Get feature importance 
            
            if feature_importance:
                features_model = results['model'].feature_names_in_ if hasattr(results['model'],'feature_names_in_') else results['model'].feature_name_
                for i,feature in enumerate(features_model):
                    if hasattr(results['model'],'feature_importances_'):
                        feature_importances[feature].extend([results['model'].feature_importances_[i]])
                    elif hasattr(results['model'],'coef_'):
                        feature_importances[feature].extend([results['model'].coef_[0][i]])
        
        if feature_importance:
            for feature in features:
                feature_importances[feature] = np.nanmean(feature_importances[feature])
            #Convert to dataframe and save as csv:
            #
            feature_importances = pd.DataFrame.from_dict(feature_importances,orient='index',columns=['mean_coef'])
            feature_importances.to_csv(Path(path_to_save,f'feature_importances_{model}.csv'))
        
        all_results.to_excel(Path(path_to_save,f'results_val_{filename_to_save}.xlsx'),index=False)
        IDs_val.to_excel(Path(path_to_save,'IDs_val.xlsx'),index=False)

        scores_dev = {'random_seed':score_seeds,
                'ID':IDs_val['ID'],
                'raw_logpost':logpost,
                'cal_logpost':cal_logpost,
                'y_true':y_true_val,
                'y_pred':y_pred_val}

        pickle.dump(scores_dev,open(Path(path_to_save,f'scores_val_{filename_to_save}.pkl'),'wb'))
        scores_dev['raw_logodds'] = logpost[:,1] - logpost[:,0]
        scores_dev['cal_logodds'] = cal_logpost[:,1] - cal_logpost[:,0]

        del scores_dev['raw_logpost']
        del scores_dev['cal_logpost'] 
        
        pd.DataFrame.from_dict(scores_dev).to_excel(Path(path_to_save,f'scores_val_{filename_to_save}.xlsx'),index=False)
        
        if held_out:
            all_results_test.to_excel(Path(path_to_save,f'results_test_{filename_to_save}.xlsx'),index=False)

            IDs_test.to_excel(Path(path_to_save,'IDs_test.xlsx'),index=False)

            scores_test = {'random_seed':score_seeds_test,
                        'ID':IDs_test['ID'],
                        'raw_logpost':logpost_test,
                        'cal_logpost':cal_logpost_test,
                        'y_true':y_true_test,
                        'y_pred':y_pred_test}
            
            pickle.dump(scores_test,open(Path(path_to_save,f'scores_test_{filename_to_save}.pkl'),'wb'))
            scores_test['raw_logodds'] = logpost_test[:,1] - logpost_test[:,0]
            scores_test['cal_logodds'] = cal_logpost_test[:,1] - cal_logpost_test[:,0]

            del scores_test['raw_logpost']
            del scores_test['cal_logpost'] 
            
            pd.DataFrame.from_dict(scores_test).to_excel(Path(path_to_save,f'scores_test_{filename_to_save}.xlsx'),index=False)

        if feature_selection:
            selected_features = pd.DataFrame.from_dict(results['selected_features']).sort_values(by='score',ascending=False)
            selected_features.to_excel(Path(path_to_save,f'selected_features_{filename_to_save}.xlsx'),index=False)