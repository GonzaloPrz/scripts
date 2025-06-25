import pandas as pd
import numpy as np
from pathlib import Path
import itertools, pickle, sys, warnings, json, os
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor as KNNR
from xgboost import XGBRegressor as xgboostr

from sklearn.neighbors import KNeighborsRegressor

from sklearn.utils import resample 

from expected_cost.ec import *

from expected_cost.calibration import calibration_train_on_heldout
from psrcal.calibration import AffineCalLogLoss, AffineCalBrier, HistogramBinningCal

from scipy.stats import bootstrap

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

late_fusion = False

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
stat_folder = config['stat_folder']
kfold_folder = config['kfold_folder']
hyp_opt = config["n_iter"] > 0 
shuffle_labels = config['shuffle_labels']
feature_selection = config['feature_selection']
filter_outliers = config['filter_outliers']
n_boot_test = int(config['n_boot_test'])
n_boot_train = int(config['n_boot_train'])
calibrate = bool(config["calibrate"])

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
model_types = main_config['models'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config["data_file"][project_name]

if not isinstance(scoring_metrics,list):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]
metrics_names_ = main_config['metrics_names'][problem_type]
cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
parallel = bool(config["parallel"])

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]

##---------------------------------PARAMETERS---------------------------------##
data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)
save_dir = Path(str(data_dir).replace('data','results'))    

results_test = pd.DataFrame()

for scoring in scoring_metrics:
    if config["test_size"] == 0:
        continue
    filename = f'best_best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffled_calibrated_bayes.csv'.replace('__','_')

    if not hyp_opt:
        filename = filename.replace("_hyp_opt","")
    if not feature_selection:
        filename = filename.replace("_feature_selection","")
    if not shuffle_labels:
        filename = filename.replace("_shuffled","")
    if not calibrate:
        filename = filename.replace("_calibrated","")

    best_models = pd.read_csv(Path(results_dir,filename))

    for r, row in best_models.iterrows():
        task = row['task']
        dimension = row['dimension']
        model_type = row['model_type']
        random_seed_test = row['random_seed_test']
        if str(random_seed_test) == 'nan':
            continue
            
        y_label = row['y_label']

        print(task,dimension,model_type,y_label)
        try:
            trained_model = pickle.load(open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,f'model_{model_type}.pkl'),'rb'))
            trained_scaler = pickle.load(open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,f'scaler_{model_type}.pkl'),'rb'))
            trained_imputer = pickle.load(open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed_test,f'imputer_{model_type}.pkl'),'rb'))
        except:
            continue
    
        path_to_results = Path(save_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'bayes',scoring,'hyp_opt', 'feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '',"shuffle" if shuffle_labels else "")

        X_train = pickle.load(open(Path(path_to_results,random_seed_test,'X_train.pkl'),'rb'))
        y_train = pickle.load(open(Path(path_to_results,random_seed_test,'y_train.pkl'),'rb'))
        IDs_train = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_train.pkl'),'rb'))

        X_test = pickle.load(open(Path(path_to_results,random_seed_test,'X_test.pkl'),'rb'))
        y_test = pickle.load(open(Path(path_to_results,random_seed_test,'y_test.pkl'),'rb'))
        IDs_test = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_test.pkl'),'rb'))
        params = trained_model.get_params()
        features = trained_model.feature_names_in_
        
        if 'probability' in params.keys():
            params['probability'] = True

        metrics_names = list(set(metrics_names_) - set(['roc_auc','f1','recall','precision'])) if cmatrix is not None or len(np.unique(y_train)) > 2 else metrics_names_
        
        model = utils.Model(type(trained_model)(**params),type(trained_scaler),type(trained_imputer))
        model.train(X_train[features],y_train)

        outputs = model.eval(X_test[features],problem_type)
        
        subfolders = [
                task, dimension, config['scaler_name'],
                config['kfold_folder'], y_label, config['stat_folder'],'bayes',scoring,
                'hyp_opt' if config['n_iter'] > 0 else '','feature_selection' if config['feature_selection'] else '',
                'filter_outliers' if config['filter_outliers'] and problem_type == 'reg' else '',
                'shuffle' if config['shuffle_labels'] else '',random_seed_test
            ]

        path_to_save = results_dir.joinpath(*[str(s) for s in subfolders if s])
        
        pickle.dump(outputs,open(Path(path_to_save,f'outputs_test_{model_type}.pkl'),'wb'))
        # Prepare data for bootstrap: a tuple of index arrays to resample
        data_indices = (np.arange(y_test.shape[-1]),)

        # Define the statistic function with data baked in
        stat_func = lambda indices: utils._calculate_metrics(
            indices, outputs, y_test.values, 
            metrics_names, problem_type, cmatrix
        )

        # 1. Calculate the point estimate (the actual difference on the full dataset)
        point_estimates = stat_func(data_indices[0])

        # 2. Calculate the bootstrap confidence interval
        try:
            # Try the more accurate BCa method first
            res = bootstrap(
                data_indices,
                stat_func,
                n_resamples=n_boot_test, # Use configured n_boot
                method=config["bootstrap_method"],
                vectorized=False,
                random_state=42
            )
            bootstrap_method = config["bootstrap_method"]

        except ValueError as e:
            # If BCa fails (e.g., due to degenerate samples), fall back to percentile
            print(f"WARNING: {config['bootstrap_method']} method failed for {task}/{dimension}/{y_label}. Falling back to 'percentile'. Error: {e}")
            res = bootstrap(
                data_indices,
                stat_func,
                n_resamples=n_boot_test,
                method='percentile',
                vectorized=False,
                random_state=42
            )
            bootstrap_method = 'percentile'

        best_models.loc[r,'bootstrap_method_holdout'] = bootstrap_method
        for i, metric in enumerate(metrics_names):
            est = point_estimates[i]
            ci_low, ci_high = res.confidence_interval.low[i], res.confidence_interval.high[i]
            best_models.loc[r,f'{metric}_holdout'] = f"{est:.3f}, ({ci_low:.3f}, {ci_high:.3f})"
        
    filename_to_save = f'best_best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffle_calibrated_bayes_test.csv'.replace('__','_')

    if not hyp_opt:
        filename_to_save = filename_to_save.replace('_hyp_opt','')
    
    if not feature_selection:
        filename_to_save = filename_to_save.replace('_feature_selection','')
    
    if not shuffle_labels:
        filename_to_save = filename_to_save.replace('_shuffle','')
    
    if not calibrate:
        filename_to_save = filename_to_save.replace('_calibrated','')
    best_models.to_csv(Path(results_dir,filename_to_save))
        