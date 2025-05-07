import sys, itertools, json, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.naive_bayes import GaussianNB

from expected_cost.calibration import calibration_train_on_test
from psrcal.calibration import AffineCalLogLoss, AffineCalBrier, HistogramBinningCal

import pickle

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

config = json.load(Path(Path(__file__).parent,'config.json').open())
project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = config['feature_selection']
filter_outliers = config['filter_outliers']
calibrate = bool(config["calibrate"])

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

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for scoring in scoring_metrics:
    filename = f'metrics_{scoring}_feature_selection_dev.csv' if feature_selection else f'metrics_{scoring}_dev.csv'
    best_models = pd.read_csv(Path(results_dir,filename))

    tasks = best_models['task'].unique()
    dimensions = best_models['dimension'].unique()
    y_labels = best_models['y_label'].unique()
    model_types = best_models['model_type'].unique()

    for task, dimension, y_label,model_type in itertools.product(tasks,dimensions,y_labels,model_types):
        
        path_to_results = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'bayes',scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '')
        random_seeds = [folder.name for folder in path_to_results.iterdir() if 'random_seed' in folder.name]
        if len(random_seeds) == 0:
            random_seeds = ['']

        for random_seed in random_seeds:
            all_models = pd.read_csv(Path(path_to_results,random_seed,f'all_models_{model_type}.csv'))
            
            features = [col for col in all_models.columns if f'{task}__' in col]
            params = [col for col in all_models.columns if col not in ['threshold','random_seed','fold'] + scoring_metrics + features]

            #Round models to the first decimal different from zero
            for param in params:
                if all_models[param].dtype == 'float64':
                    order_of_magnitude = np.floor(np.log10(all_models[param]))
                    all_models[param] = np.array([round(all_models[param][i], int(-order_of_magnitude[i])) if order_of_magnitude[i] <= 0 else int(all_models[param][i]/(10**order_of_magnitude[i]))*(10**order_of_magnitude[i]) for i in range(len(all_models[param])) ])
                else:
                    all_models[param] = all_models[param].astype('category')
            #Select the most frequent combination of hyperparameters
            combinations = all_models.groupby(params).size().reset_index(name='counts')
            combinations = combinations[combinations['counts'] > 0]
            combinations = combinations.sort_values(by='counts', ascending=False)
            combinations = combinations.reset_index(drop=True)
            combinations['percent'] = (combinations['counts'] / all_models.shape[0])*100

            best_params = combinations.loc[combinations['counts'].idxmax()]

            best_features = [ft for ft in features if np.sum(all_models[ft])/all_models.shape[0] > .75]

            X_train = pickle.load(open(Path(path_to_results,random_seed,'X_train.pkl'),'rb'))
            y_train = pickle.load(open(Path(path_to_results,random_seed,'y_train.pkl'),'rb'))

            model = utils.Model(models_dict[problem_type][model_type](**best_params[params].to_dict()),StandardScaler if scaler_name == 'StandardScaler' else MinMaxScaler,KNNImputer)

            model.train(X_train,y_train)

            Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring).mkdir(exist_ok=True,parents=True)
            combinations.to_csv(Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'best_params_{model_type}.csv'),index=False)
            pickle.dump(model.model,open(Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'model_{model_type}.pkl'),'wb'))
            pickle.dump(model.scaler,open(Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'scaler_{model_type}.pkl'),'wb'))
            pickle.dump(model.imputer,open(Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'imputer_{model_type}.pkl'),'wb'))