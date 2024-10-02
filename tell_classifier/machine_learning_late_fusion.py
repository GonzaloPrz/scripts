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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as xgboost
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm
import itertools,pickle,sys
from scipy.stats import loguniform, uniform, randint
from random import randint as randint_random 
import random

from random import randint as randint_random 

#sys.path.append(str(Path(Path.home() / 'Doctorado' / 'Codigo' / 'machine_learning')))

sys.path.append(str(Path(Path.home(),'Proyectos')))

from machine_learning_module import *

from expected_cost.ec import *
from expected_cost.utils import *

#Parámetros
n_seeds = 20
boot_train = False
boot_val = 0
n_folds = 5
n_iter = 100
scaler_name = 'StandardScaler'
scaler = MinMaxScaler() if scaler_name == 'minmax' else StandardScaler()
cmatrix = None
thresholds = ['bayes']
random_seeds = np.arange(n_seeds)
feature_importance = True 
calibrate_list = [False,True]
cheat = False
random_seeds = np.arange(n_seeds)
feature_importance = True
feature_selection_list = [False]
shuffle_labels_list = [False]
held_out = False
hyp_tuning = False
shuffle_labels = False
tasks = ['MOTOR_LIBRE','NEUTRO_LECTURA']
scoring = 'roc_auc'

models_dict = {
    'lda':LDA(),
    'knn':KNN(),
    'lr':LR(),
    'svc':SVC(),
    'xgboost':xgboost()
    }
#Hyperparameter spaces for random search

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
                }
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
                #'use_label_encoder':[False],
                'eval_metric':['logloss'],
                #'max_depth':randint(1,10),
                #'learning_rate':loguniform(1e-5,1e5),
                #'gamma': uniform(1e-5,1)
                }
    }


base_dir = Path(Path(__file__).parent,'data')

y_labels = ['group']
for i,seed in enumerate(tqdm(random_seeds)):
    
    CV = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=seed)
       
    for y_label,task in itertools.product(y_labels,tasks):                        
        data = pd.read_csv(Path(base_dir,f'{task}.csv'))

        if shuffle_labels:
            data[y_label] = pd.Series(np.random.permutation(data[y_label]))

        y = data.pop(y_label).map({'C':0,'P':1})

        ID = data.pop('id')

        features = data.columns
        all_features = list(features)
        
        #impute mising data
        imputer = KNNImputer(n_neighbors=5)
        data = pd.DataFrame(imputer.fit_transform(data[features]),columns=features)

        if held_out:
            ID_train, IDs_test,_,_ = train_test_split(ID,y,test_size=0.2,random_state=0,stratify=y)
            ID_train = ID_train.reset_index(drop=True)
            IDs_test = IDs_test.reset_index(drop=True)
            data, data_test, y, y_test = train_test_split(data,y,test_size=0.2,random_state=0,stratify=y)

            data = data.reset_index(drop=True)
            data_test = data_test.reset_index(drop=True)

            y = y.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)
        else:
            ID_train = ID
            ID_test = []
            data_test = pd.DataFrame()
            y_test = np.empty(0)

        for feature_selection,calibrate,model in itertools.product(feature_selection_list,calibrate_list,models_dict.keys()):
            path_to_save = Path(Path(__file__).parent,task,scaler_name,f'{n_folds}_folds',f'{n_seeds}_seeds',y_label)
            path_to_save = Path(path_to_save,'bt') if boot_train or boot_val else path_to_save
        
            path_to_save = Path(path_to_save,f'no_hyp_opt') if not hyp_tuning else Path(path_to_save,f'hyp_opt',scoring)
            
            if model in ['gnb','knn'] and feature_selection:
                continue

            if calibrate and hyp_tuning:
                hyperp_spaces['knn'] = {'n_neighbors':randint(1,np.floor(((n_folds-1)/n_folds)**2*data.shape[0]))}
            elif hyp_tuning:
                hyperp_spaces['knn'] = {'n_neighbors':randint(1,np.floor(((n_folds-1)/n_folds)*data.shape[0]))}
            
            path_to_save = Path(path_to_save,'held_out') if held_out and 'held_out' not in str(path_to_save) else path_to_save
            path_to_save = Path(path_to_save,'shuffle_labels') if shuffle_labels and 'shuffle_labels' not in str(path_to_save) else path_to_save
            path_to_save = Path(path_to_save,'feature_selection') if feature_selection and 'feature_selection' not in str(path_to_save) else path_to_save
            path_to_save = Path(path_to_save,'calibrated') if calibrate and 'calibrated' not in str(path_to_save) else path_to_save
            path_to_save = Path(path_to_save,'cheat') if cheat and 'cheat' not in str(path_to_save) else path_to_save

            path_to_save.mkdir(parents=True,exist_ok=True)

            filename_to_save = model

            print(path_to_save / f'results_{filename_to_save}.csv')

            """ if Path(path_to_save / f'results_{filename_to_save}.csv').exists():
                continue """

            all_results = pd.DataFrame(columns=['random_seed']) 
            IDs_train = pd.DataFrame(columns=['random_seed','fold','bootstrap_index','ID'])
            IDs_val = pd.DataFrame(columns=['random_seed','fold','bootstrap_index','ID'])

            #for feature in features:
            '''
            exp = Experiment(api_key='pBa801obZaGlXeXkBBNHZg28u',
            project_name=f'{config.replace(f"{config}_T1","").replace(f"_{dimension}","")}-{model}-{estadisticos[dimension]}',
            workspace='ml-cnc',log_code=False,auto_param_logging=False,
            log_env_details=False)
            '''     

            score_seeds = np.empty(0)
            logpost = np.empty((0,2))
            cal_logpost = np.empty((0,2))
            
            y_true_val = np.empty(0)
            y_pred_bayes_val = np.empty(0)
            bootstrap_index = np.empty(0)

            feature_importances = dict([(feature,[]) for feature in features])
                                
            #results= train_predict_evaluate(models[model],features_by_task[config][[col for col in features_by_task[config] if feature in col]],y,CV=CV,cmatrix=cmatrix)
            results = train_tune_calibrate_eval(model=models_dict[model],scaler=scaler,X_dev=data[features],y_dev=y,X_test=data_test,y_test=y_test,random_seed=seed,hyperp_space=hyperp_spaces[model],n_iter=n_iter,scoring=scoring,costs=cmatrix,priors=None,CV_out=CV,CV_in=CV,cal=calibrate,cheat=cheat,feature_selection=feature_selection,IDs=ID,boot_train=boot_train,boot_val=boot_val)
            results['IDs_train']['random_seed'] = [seed]*results['IDs_train'].shape[0]
            results['IDs_val']['random_seed'] = [seed]*results['IDs_val'].shape[0]

            if held_out:
                suffix = '_test'
                IDs_val = IDs_test
            else:
                suffix = '_dev'

            logpost = np.vstack([logpost,np.array(results[f'raw_logpost{suffix}'])])
            cal_logpost = np.vstack([cal_logpost,np.array(results[f'cal_logpost{suffix}'])])
            
            score_seeds = np.hstack([score_seeds,np.array([seed]*results[f'raw_logpost{suffix}'].shape[0])])
            y_true_val = np.hstack([y_true_val,results[f'y_true{suffix}']])
            y_pred_bayes_val = np.hstack([y_pred_bayes_val,results[f'y_pred{suffix}']])
            
            df_append = pd.DataFrame({'random_seed':[seed]})

            df_append['accuracy'] = results['accuracy']
            df_append['roc_auc'] = results['roc_auc']
            df_append['precision'] = results['precision']
            df_append['sensitivity'] = results['recall']
            df_append['f1'] = results['f1']
            df_append['norm_expected_cost'] = results['norm_expected_cost']
            df_append['norm_cross_entropy'] = results['norm_cross_entropy']

            best_model = results['model'].get_params()
            df_append['best_params'] = str(dict([(param,best_model[param]) for param in hyperp_spaces[model].keys()]))

            all_results = pd.concat([all_results,df_append],axis=0,ignore_index=True)
            IDs_train = pd.concat((IDs_train,results['IDs_train']),axis=0)
            IDs_val = pd.concat((IDs_val,results['IDs_val']),axis=0)

            '''
            keys = list(results['all_metrics'].keys())
            
            for key in keys:
                for j in range(np.max([n_bootstrap,1])):
                    exp.log_metric(key,float(results[f'all_metrics'][key][j]),step=i*np.max([n_bootstrap,1])+j)
            exp.end()
            '''
            # Get feature importance 
            
            if feature_importance:
                for i,feature in enumerate(results['model'].feature_names_in_):
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
        
        all_results.to_csv(Path(path_to_save,f'results_{filename_to_save}.csv'),index=False)
        IDs_train.to_csv(Path(path_to_save,f'IDs_train_{filename_to_save}.csv'),index=False)
        IDs_val.to_csv(Path(path_to_save,f'IDs_val_{filename_to_save}.csv'),index=False)

        scores = {'random_seed':score_seeds,'raw_logpost':logpost,'cal_logpost':cal_logpost,'y_true':y_true_val,'bootstrap_index':bootstrap_index,'ID':IDs_val['ID']}

        pickle.dump(scores,open(Path(path_to_save,f'scores_{filename_to_save}.pkl'),'wb'))
        scores['raw_logodds'] = logpost[:,1] - logpost[:,0]
        scores['cal_logodds'] = cal_logpost[:,1] - cal_logpost[:,0]

        del scores['raw_logpost']
        del scores['cal_logpost'] 
        del scores['bootstrap_index']
        #pd.DataFrame.from_dict(scores).to_csv(Path(path_to_save,f'scores_{filename_to_save}.csv'),index=False)
        
        if feature_selection:
            selected_features = pd.DataFrame.from_dict(results['selected_features']).sort_values(by='score',ascending=False)
            selected_features.to_csv(Path(path_to_save,f'selected_features_{filename_to_save}.csv'),index=False)