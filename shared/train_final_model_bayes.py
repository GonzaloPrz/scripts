import sys, itertools, json, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.naive_bayes import GaussianNB

from expected_cost.calibration import calibration_train_on_test
from psrcal.calibration import AffineCalLogLoss, AffineCalBrier, HistogramBinningCal

import pickle

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

from expected_cost.ec import CostMatrix

config = json.load(Path(Path(__file__).parent,'config.json').open())
project_name = config["project_name"]
scaler_name = config['scaler_name']
n_folds = config['n_folds_inner']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = config['feature_selection']
filter_outliers = config['filter_outliers']
calibrate = bool(config["calibrate"])
n_iter = int(config["n_iter"])
init_points = int(config["init_points"])
scaler_name = config['scaler_name']

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
problem_type = main_config['problem_type'][project_name]
cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
thresholds = main_config['thresholds'][project_name]
overwrite = bool(config["overwrite"])
if not isinstance(scoring_metrics,list):
    scoring_metrics = [scoring_metrics]
    
models = main_config["models"][project_name]

models_dict = {
        'clf': {
            'lr': LogisticRegression,
            'svc': SVC,
            'knnc': KNeighborsClassifier,
            'xgb': XGBClassifier,
            'nb':GaussianNB
        },
        'reg': {
            'lasso': Lasso,
            'ridge': Ridge,
            'elastic': ElasticNet,
            'svr': SVR,
            'xgb': XGBRegressor
        }
    }

hyperp = {'lr':{'C':(1e-4,100)},
          'svc':{'C':(1e-4,100),
                 'gamma':(1e-4,1e4)},
            'knnc':{'n_neighbors':(1,40)},
            'xgb':{'max_depth':(1,10),
                   'n_estimators':(1,2000),
                   'learning_rate':(1e-4,1)},
            'lasso':{'alpha':(1e-4,1e4)},
            'ridge':{'alpha':(1e-4,1e4)},
            'elastic':{'alpha':(1e-4,1e4),
                       'l1_ratio':(0,1)},
            'knnr':{'n_neighbors':(1,40)},
            'svr':{'C':(1e-4,100),
                    'gamma':(1e-4,1e4)}
            }

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for scoring,threshold in itertools.product(scoring_metrics,thresholds):
    if str(threshold) == 'None':
        threshold = None
    filename = f'metrics_{kfold_folder}_{scoring}_{stat_folder}_feature_selection_dev.csv'.replace('__','_') if feature_selection else f'metrics_{kfold_folder}_{scoring}_{stat_folder}_dev.csv'.replace('__','_')
    best_models = pd.read_csv(Path(results_dir,filename))

    tasks = best_models['task'].unique()
    dimensions = best_models['dimension'].unique()
    y_labels = best_models['y_label'].unique()
    model_types = best_models['model_type'].unique()

    for task, dimension, y_label,model_type in itertools.product(tasks,dimensions,y_labels,model_types):
        print(task,dimension,y_label,model_type)
        path_to_results = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'bayes',scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '')
        random_seeds = [folder.name for folder in path_to_results.iterdir() if 'random_seed' in folder.name]
        random_seeds.append('')
        
        for random_seed in random_seeds:
            all_models = pd.read_csv(Path(path_to_results,random_seed,f'all_models_{model_type}.csv'))
            
            features = [col for col in all_models.columns if f'{task}__' in col]
            
            X_train = pickle.load(open(Path(path_to_results,random_seed,'X_train.pkl'),'rb'))
            y_train = pickle.load(open(Path(path_to_results,random_seed,'y_train.pkl'),'rb'))

            CV = (StratifiedKFold(n_splits=int(n_folds), shuffle=True)
                        if config['stratify'] and problem_type == 'clf'
                        else KFold(n_splits=n_folds, shuffle=True))  
            
            hyperp['knnc']['n_neighbors'] = (1,int(X_train.shape[0]*(1-1/n_folds)-1))
            hyperp['knnr']['n_neighbors'] = (1,int(X_train.shape[0]*(1-1/n_folds)-1))

            model_class = models_dict[problem_type][model_type]
            scaler = StandardScaler if scaler_name == 'StandardScaler' else MinMaxScaler
            imputer = KNNImputer
            if cmatrix is None:
                cmatrix = CostMatrix.zero_one_costs(K=len(np.unique(y_train)))
            best_features = utils.rfe(utils.Model(model_class(probability=True) if model_class == SVC else model_class(),scaler,imputer,None,None),X_train,y_train,CV,scoring,problem_type,cmatrix=cmatrix,priors=None,threshold=threshold) if feature_selection else X_train.columns
            
            if Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'model_{model_type}.pkl').exists() and not overwrite:
                print('Model already exists')
                continue
            
            best_params, best_score = utils.tuning(model_class,scaler,imputer,X_train,y_train,hyperp[model_type],CV,init_points=int(config['init_points']),n_iter=n_iter,scoring=scoring,problem_type=problem_type,cmatrix=cmatrix,priors=None,threshold=threshold,calmethod=None,calparams=None)
            
            if model_type == 'clf' and model_class == SVC:
                if 'probability' in best_params:
                    del best_params['probability']
                    best_params['probability'] = True
                model = utils.Model(model_class(**best_params),scaler,imputer)
            else:
                model = utils.Model(model_class(**best_params),scaler,imputer)
            
            model.train(X_train[best_features],y_train)

            Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring).mkdir(exist_ok=True,parents=True)
            pickle.dump(model.model,open(Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'model_{model_type}.pkl'),'wb'))
            pickle.dump(model.scaler,open(Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'scaler_{model_type}.pkl'),'wb'))
            pickle.dump(model.imputer,open(Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'imputer_{model_type}.pkl'),'wb'))
            
            