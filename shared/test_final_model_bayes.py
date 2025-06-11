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
def _calculate_metrics(indices, outputs, y, metrics_names, prob_type, cost_matrix):
    """
    Statistic function for bootstrap. Calculates differences for ALL metrics at once.
    """
    # Resample y, ensuring we don't operate on an empty or invalid slice
    resampled_y = y[indices].ravel()

    # If a resample is degenerate (e.g., missing a class), metric calculation is impossible.
    # Return NaNs to signal this. The 'bca' method will fail, triggering our fallback.
    if np.unique(resampled_y).shape[0] != np.unique(y).shape[0]:
        return np.full(len(metrics), np.nan)

    # Resample model outputs
    resampled_out = outputs[indices, :].reshape(-1, outputs.shape[-1])

    # Get metrics for both classifiers
    if prob_type == 'clf':
        metrics, _ = utils.get_metrics_clf(resampled_out, resampled_y, metrics_names, cmatrix=cost_matrix)
    else: # 'reg'
        metrics = utils.get_metrics_reg(resampled_out, resampled_y, metrics_names)

    # Return an array of differences
    return np.array([metrics[m] for m in metrics_names])

def test_models_bootstrap(model_class,params,features,scaler,imputer,calmethod,calparams,X_dev,y_dev,X_test,y_test,metrics_names,problem_type,threshold=None,cmatrix=None,priors=None,calibrate=False):
    if not isinstance(X_dev,pd.DataFrame):
        X_dev = pd.DataFrame(X_dev.squeeze(),columns=features)

    if not isinstance(X_test,pd.DataFrame):
        X_test = pd.DataFrame(X_test.squeeze(),columns=features)

    if 'gamma' in params.keys():
        try: 
            params['gamma'] = float(params['gamma'])
        except:
            pass
    if 'probability' in params.keys():
        params['probability'] = True

    outputs = utils.test_model(model_class,params,scaler,imputer, X_dev[features], y_dev, X_test[features], problem_type=problem_type)
        
    if calibrate:
        model = utils.Model(model_class(**params),scaler,imputer,calmethod,calparams)
        model.train(X_dev[features], y_dev)
        outputs_dev = model.eval(X_dev[features],problem_type)
    
        outputs,_ = model.calibrate(outputs,None,outputs_dev,y_dev)
         
    n_samples = X_test.shape[0]
    data_indices = (np.arange(n_samples),)

    # Define the statistic function with data baked in
    stat_func = lambda indices: _calculate_metrics(
        indices, outputs, y_test, 
        metrics_names, problem_type, cmatrix
    )

    # 1. Calculate the point estimate (the actual difference on the full dataset)
    point_estimates = stat_func(data_indices[0])
    result = {}

    # 2. Calculate the bootstrap confidence interval
    try:
        # Try the more accurate BCa method first
        res = bootstrap(
            data_indices,
            stat_func,
            n_resamples=n_boot_test, # Use configured n_boot
            method='bca',
            vectorized=False,
            random_state=42
        )
        result["bootstrap_method_holdout"] = 'bca'

    except ValueError as e:
        # If BCa fails (e.g., due to degenerate samples), fall back to percentile
        print(f"WARNING: BCa method failed for {task}/{dimension}/{y_label}. Falling back to 'percentile'. Error: {e}")
        res = bootstrap(
            data_indices,
            stat_func,
            n_resamples=n_boot_test,
            method='percentile',
            vectorized=False,
            random_state=42
        )
        result["bootstrap_method_holdout"] = 'percentile'                    
                
    for i, metric in enumerate(metrics_names):
        est = point_estimates[i]
        ci_low, ci_high = res.confidence_interval.low[i], res.confidence_interval.high[i]
        result[f"{metric}_holdout"] = f"{est:.3f}, ({ci_low:.3f}, {ci_high:.3f})"

    return result

late_fusion = False

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
stat_folder = config['stat_folder']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
feature_selection = config['feature_selection']
filter_outliers = config['filter_outliers']
n_boot_test = int(config['n_boot_test'])
n_boot_train = int(config['n_boot_train'])

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
    filename = f'metrics_{kfold_folder}_{scoring}_{stat_folder}_feature_selection_dev.csv'.replace('__','_') if feature_selection else f'metrics_{kfold_folder}_{scoring}_{stat_folder}_dev.csv'.replace('__','_')
    best_models = pd.read_csv(Path(results_dir,filename))

    for r, row in best_models.iterrows():
        task = row['task']
        dimension = row['dimension']
        model_type = row['model_type']
        random_seed_test = row['random_seed_test']
        if str(random_seed_test) == 'nan':
            random_seed_test = ''
            
        y_label = row['y_label']

        print(task,dimension,model_type,y_label)
        try:
            trained_model = pickle.load(open(Path(results_dir,f'final_models_bayes',task,dimension,y_label,scoring,kfold_folder,f'model_{model_type}.pkl'),'rb'))
            trained_scaler = pickle.load(open(Path(results_dir,f'final_models_bayes',task,dimension,y_label,scoring,kfold_folder,f'scaler_{model_type}.pkl'),'rb'))
            trained_imputer = pickle.load(open(Path(results_dir,f'final_models_bayes',task,dimension,y_label,scoring,kfold_folder,f'imputer_{model_type}.pkl'),'rb'))
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

        metrics_names = list(set(metrics_names_) - set(['roc_auc','f1','recall','precision'])) if cmatrix is not None or len(np.unique(y_train)) > 2 else metrics_names_
        
        result_append = test_models_bootstrap(type(trained_model),params,features,type(trained_scaler),type(trained_imputer),None,None,X_train,y_train,X_test,y_test,metrics_names,problem_type,threshold=None,cmatrix=cmatrix)
        
        result_append.update({'task':task,'dimension':dimension,'y_label':y_label,'model_type':model_type,'random_seed_test':random_seed_test, "bootstrap_method_dev": row["bootstrap_method_dev"]})

        for metric in metrics_names:
            try:
                result_append.update({f'{metric}_dev':row[metric]})
            except:
                continue
        if results_test.empty:
            results_test = pd.DataFrame(result_append,index=[0])
        else:
            results_test = pd.concat([results_test,pd.DataFrame(result_append,index=[0])],ignore_index=True)
    results_test.to_csv(Path(results_dir,f'best_models_{kfold_folder}_{scoring}_{stat_folder}_feature_selection_test.csv'.replace('__','_') if feature_selection else
                             f'best_models_{kfold_folder}_{scoring}_{stat_folder}_test.csv'.replace('__','_')))