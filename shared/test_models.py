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

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

late_fusion = False

def test_models_bootstrap(model_class,row,scaler,imputer,calmethod,calparams,X_dev,y_dev,X_test,y_test,all_features,y_labels,metrics_names,IDs_test,boot_train,boot_test,problem_type,threshold,cmatrix=None,priors=None,bayesian=False,calibrate=False):

    results_r = row.dropna().to_dict()

    outputs_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test),len(np.unique(y_dev)) if problem_type=='clf' else 1))
    if problem_type == 'reg':
        outputs_bootstrap = outputs_bootstrap.squeeze(axis=-1)

    y_true_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test)))
    y_pred_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test)))
    IDs_test_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test)), dtype=object)

    if cmatrix is not None or len(np.unique(y_dev)) > 2:
        metrics_names = list(set(metrics_names) - set(['roc_auc','accuracy','f1','recall','precision']))

    metrics_test_bootstrap = {metric: np.empty((np.max((1,boot_train)),np.max((1,boot_test)))) for metric in metrics_names}

    params = dict((key,value) for (key,value) in results_r.items() if not isinstance(value,dict) and all(x not in key for x in ['inf','sup','mean'] + all_features + y_labels + ['id','Unnamed: 0','threshold','idx']))

    features = [col for col in all_features if col in results_r.keys() and results_r[col] == 1]
    features_dict = {col:results_r[col] for col in all_features if col in results_r.keys()}

    if not isinstance(X_dev,pd.DataFrame):
        X_dev = pd.DataFrame(X_dev.squeeze(),columns=all_features)

    if not isinstance(X_test,pd.DataFrame):
        X_test = pd.DataFrame(X_test.squeeze(),columns=all_features)

    if 'gamma' in params.keys():
        try: 
            params['gamma'] = float(params['gamma'])
        except:
            pass
    if 'random_state' in params.keys():
        params['random_state'] = int(params['random_state'])
    outputs = np.empty((np.max((1,boot_train)),len(y_test),len(np.unique(y_dev)) if problem_type=='clf' else 1))

    if problem_type == 'reg':
        outputs = outputs.squeeze(axis=-1)

    for b_train in range(np.max((1,boot_train))):
        boot_index_train = resample(X_dev.index, n_samples=X_dev.shape[0], replace=True, random_state=b_train) if boot_train > 0 else X_dev.index
        outputs[b_train,:] = utils.test_model(model_class,params,scaler,imputer, X_dev.loc[boot_index_train,features], y_dev[boot_index_train], X_test[features], problem_type=problem_type)
        
        if calibrate:
            model = utils.Model(model_class(**params),scaler,imputer,calmethod,calparams)
            model.train(X_dev.loc[boot_index_train,features], y_dev)
            outputs_dev = model.eval(X_dev.loc[boot_index_train,features],problem_type)
        
            outputs[b_train,:],_ = model.calibrate(outputs[b_train,:],None,outputs_dev,y_dev)
        
        for b_test in range(np.max((1,boot_test))):
            if bayesian:
                weights = np.random.dirichlet(np.ones(y_test.shape[0]))
            else:
                weights = None

            boot_index = resample(range(outputs.shape[1]), n_samples=outputs.shape[1], replace=True, random_state=b_train * np.max((1,boot_train)) + b_test) if boot_test > 0 else X_test.index
            while len(np.unique(y_test[boot_index])) < len(np.unique(y_dev)):
                boot_index = resample(range(outputs.shape[1]), n_samples=outputs.shape[1], replace=True, random_state=b_train * np.max((1,boot_train)) + b_test*boot_test + b_test)
            outputs_bootstrap[b_train,b_test] = outputs[b_train,boot_index]

            if problem_type == 'clf':
                metrics_test, y_pred = utils.get_metrics_clf(outputs[b_train,boot_index], y_test[boot_index], metrics_names, cmatrix, priors,threshold,weights)
                y_pred_bootstrap[b_train,b_test,:] = y_pred
            else:
                metrics_test = utils.get_metrics_reg(outputs[b_train,boot_index], y_test[boot_index], metrics_names)
                y_pred_bootstrap[b_train,b_test] = outputs[b_train,boot_index]
                
            y_true_bootstrap[b_train,b_test] = y_test[boot_index]
            IDs_test_bootstrap[b_train,b_test] = IDs_test.squeeze()[boot_index]

            for metric in metrics_names:
                metrics_test_bootstrap[metric][b_train,b_test] = metrics_test[metric]

    result_append = params.copy()
    result_append.update(features_dict)

    for metric in metrics_names:
        mean, inf, sup = utils.conf_int_95(metrics_test_bootstrap[metric].flatten())

        result_append[f'inf_{metric}_test'] = np.round(inf,5)
        result_append[f'mean_{metric}_test'] = np.round(mean,5)
        result_append[f'sup_{metric}_test'] = np.round(sup,5)

        result_append[f'inf_{metric}_dev'] = np.round(results_r[f'{metric}_inf'],5)
        result_append[f'mean_{metric}_dev'] = np.round(results_r[f'{metric}_mean'],5)
        result_append[f'sup_{metric}_dev'] = np.round(results_r[f'{metric}_sup'],5)
        
    return result_append,outputs,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_test_bootstrap
##---------------------------------PARAMETERS---------------------------------##
config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
calibrate = config["calibrate"]
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
n_models = int(config["n_models"])
early_fusion = bool(config["early_fusion"])
bayesian = bool(config["bayesian"])
n_boot_test = int(config["n_boot_test"])
n_boot_train = int(config["n_boot_train"])
calibrate = bool(config["calibrate"])
overwrite = bool(config["overwrite"])

if calibrate:    
    calmethod = AffineCalLogLoss
    calparams = {'bias':True, 'priors':None}
else:
    calmethod = None
    calparams = None

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
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
problem_type = main_config['problem_type'][project_name]
if problem_type == 'clf':
    cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
else:
    cmatrix = None
    
parallel = bool(config["parallel"])

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

##---------------------------------PARAMETERS---------------------------------##
data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)
save_dir = Path(str(data_dir).replace('data','results'))    

if scaler_name == 'StandardScaler':
    scaler = StandardScaler
elif scaler_name == 'MinMaxScaler':
    scaler = MinMaxScaler
else:
    scaler = None
imputer = KNNImputer

models_dict = {'clf':{'lr': LogisticRegression,
                    'svc': SVC, 
                    'xgb': XGBClassifier,
                    'knnc': KNNC,
                    'nb': GaussianNB},
                
                'reg':{'lasso':Lasso,
                        'ridge':Ridge,
                        'elastic':ElasticNet,
                        #'knn':KNNR,
                        'svr':SVR,
                        'xgb':xgboostr
                    }
}

for task,scoring in itertools.product(tasks,scoring_metrics):
    scoring = scoring.replace('_score','')
    extremo = 'sup' if any(x in scoring for x in ['norm','error']) else 'inf'
    ascending = True if extremo == 'sup' else False

    dimensions = [folder.name for folder in Path(save_dir,task).iterdir() if folder.is_dir()]

    for dimension in dimensions:
        print(task,dimension)
        for y_label in y_labels:
            print(y_label)
            path_to_results = Path(save_dir,task,dimension,scaler_name,kfold_folder, y_label,stat_folder,'hyp_opt' if hyp_opt else '', 'feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '',"shuffle" if shuffle_labels else "")

            if not path_to_results.exists():
                continue
            
            random_seeds_test = [folder.name for folder in path_to_results.iterdir() if folder.is_dir() if 'random_seed' in folder.name]
              
            for random_seed_test in random_seeds_test:
                if int(config["n_models"] == 0):
                    files = [file for file in Path(path_to_results,random_seed_test,'bayesian' if bayesian else '').iterdir() if all(x in file.stem for x in ['all_models_','dev_bca','calibrated'])] if calibrate else [file for file in Path(path_to_results,random_seed_test,'bayesian' if bayesian else '').iterdir() if all(x in file.stem for x in ['all_models_','dev_bca']) and 'calibrated' not in file.stem]
                else:
                    files = [file for file in Path(path_to_results,random_seed_test,'bayesian' if bayesian else '').iterdir() if all(x in file.stem for x in ['best_models_','dev_bca','calibrated'])] if calibrate else [file for file in Path(path_to_results,random_seed_test,'bayesian' if bayesian else '').iterdir() if all(x in file.stem for x in ['best_models_','dev_bca']) and 'calibrated' not in file.stem]
                
                if len(files) == 0:
                    continue

                X_dev = pickle.load(open(Path(path_to_results,random_seed_test,'X_dev.pkl'),'rb'))[0,0]
                y_dev = pickle.load(open(Path(path_to_results,random_seed_test,'y_dev.pkl'),'rb'))[0,0]
                IDs_dev = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_dev.pkl'),'rb'))[0,0]
                X_test = pickle.load(open(Path(path_to_results,random_seed_test,'X_test.pkl'),'rb'))   
                y_test = pickle.load(open(Path(path_to_results,random_seed_test,'y_test.pkl'),'rb'))
                IDs_test = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_test.pkl'),'rb'))
                
                for file in files:
                    model_name = file.stem.split('_')[2]

                    if file.suffix != '.csv':
                        continue

                    filename_to_save = f'all_models_{model_name}_calibrated'
                    if config["n_models"] != 0:
                        filename_to_save = filename_to_save.replace('all_models',f'best_models_{scoring}')
                    if not calibrate:
                        filename_to_save = filename_to_save.replace('_calibrated','')

                    print(model_name)
                    
                    results_dev = pd.read_excel(file) if file.suffix == '.xlsx' else pd.read_csv(file)
                    
                    if f'{extremo}_{scoring}' in results_dev.columns:
                        scoring_col = f'{extremo}_{scoring}'
                    elif f'{extremo}_{scoring}_dev' in results_dev.columns:
                        scoring_col = f'{extremo}_{scoring}_dev'
                    else:
                        scoring_col = f'{scoring}_{extremo}'

                    results_dev = results_dev.sort_values(by=scoring_col.replace('_score',''),ascending=ascending)
                    
                    all_features = [col for col in results_dev.columns if any([dim in col for dim in dimension.split('__')])]
                    if 'threshold' not in results_dev.columns:
                        results_dev['threshold'] = thresholds[0]

                    if len(all_features) == 0:
                        continue
                    
                    metrics_names = main_config["metrics_names"][problem_type] if len(np.unique(y_dev)) == 2 else list(set(main_config["metrics_names"][problem_type]) - set(['roc_auc','f1','recall']))

                    if Path(file.parent,f'{filename_to_save}_test.csv').exists() and overwrite == False:
                        print(f"Testing already done")
                        continue

                    results = Parallel(n_jobs=-1 if parallel else 1)(delayed(test_models_bootstrap)(models_dict[problem_type][model_name],results_dev.loc[r,:],scaler,imputer,calmethod,calparams,X_dev,y_dev,
                                                                                X_test,y_test,all_features,y_labels,metrics_names,IDs_test,n_boot_train,
                                                                                n_boot_test,problem_type,threshold=results_dev.loc[r,'threshold'],cmatrix=cmatrix,calibrate=calibrate) 
                                                                                for r in results_dev.index)
                    
                    results_test = pd.concat([pd.DataFrame(result[0],index=[0]) for result in results])
                    results_test['idx'] = results_dev['Unnamed: 0'].values

                    outputs_test = np.stack([result[1] for result in results],axis=0)
                    
                    results_test.to_csv(Path(file.parent,f'{filename_to_save}_test.csv'))
                    outputs_filename = f'cal_outputs_test_{model_name}.pkl' if calibrate else f'outputs_test_{model_name}.pkl'
                    with open(Path(file.parent,outputs_filename),'wb') as f:
                        pickle.dump(outputs_test,f)