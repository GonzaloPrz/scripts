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

def test_models_bootstrap(model_class,params,features,scaler,imputer,calmethod,calparams,X_dev,y_dev,X_test,y_test,metrics_names,IDs_test,boot_train,boot_test,problem_type,threshold,cmatrix=None,priors=None,bayesian=False,calibrate=False):
    result_append = {}
    outputs_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test),len(np.unique(y_dev)) if problem_type=='clf' else 1))
    if problem_type == 'reg':
        outputs_bootstrap = outputs_bootstrap.squeeze(axis=-1)

    y_true_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test)))
    y_pred_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test)))
    IDs_test_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test)), dtype=object)

    if cmatrix is not None or len(np.unique(y_dev)) > 2:
        metrics_names = list(set(metrics_names) - set(['roc_auc','accuracy','f1','recall','precision']))

    metrics_test_bootstrap = {metric: np.empty((np.max((1,boot_train)),np.max((1,boot_test)))) for metric in metrics_names}

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
                try:
                    metrics_test_bootstrap[metric][b_train,b_test] = metrics_test[metric]
                except:
                    continue
    for metric in metrics_names:
        mean, inf, sup = utils.conf_int_95(metrics_test_bootstrap[metric].flatten())

        result_append.update({f'{metric}_test':f"{np.round(mean,3)} [{np.round(inf,3)},{np.round(sup,3)}]"})
        
    return result_append,outputs,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_test_bootstrap

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
    for task,model_type in itertools.product(tasks,model_types):
        filename = f'metrics_{kfold_folder}_{scoring}_{stat_folder}_feature_selection_dev.csv'.replace('__','_') if feature_selection else f'metrics_{kfold_folder}_{scoring}_{stat_folder}_dev.csv'.replace('__','_')
        best_models = pd.read_csv(Path(results_dir,filename))

        dimensions = [folder.name for folder in Path(save_dir,task).iterdir() if folder.is_dir()]
        
        for dimension in dimensions:
            print(task,dimension,model_type)
            for y_label in y_labels:
                try:
                    trained_model = pickle.load(open(Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'model_{model_type}.pkl'),'rb'))
                    trained_scaler = pickle.load(open(Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'scaler_{model_type}.pkl'),'rb'))
                    trained_imputer = pickle.load(open(Path(results_dir,f'final_models_{scoring}_bayes',task,dimension,y_label,scoring,f'imputer_{model_type}.pkl'),'rb'))
                except:
                    continue
                
                best_model_dev = best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label) & (best_models['model_type'] == model_type)]

                print(y_label)
                path_to_results = Path(save_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'bayes',scoring,'hyp_opt', 'feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '',"shuffle" if shuffle_labels else "")

                if not path_to_results.exists():
                    continue
                
                random_seeds_test = [folder.name for folder in path_to_results.iterdir() if folder.is_dir() if 'random_seed' in folder.name]
                
                for random_seed_test in random_seeds_test:
                    X_train = pickle.load(open(Path(path_to_results,random_seed_test,'X_train.pkl'),'rb'))
                    y_train = pickle.load(open(Path(path_to_results,random_seed_test,'y_train.pkl'),'rb'))
                    IDs_train = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_train.pkl'),'rb'))

                    X_test = pickle.load(open(Path(path_to_results,random_seed_test,'X_test.pkl'),'rb'))
                    y_test = pickle.load(open(Path(path_to_results,random_seed_test,'y_test.pkl'),'rb'))
                    IDs_test = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_test.pkl'),'rb'))
                    params = trained_model.get_params()
                    features = trained_model.feature_names_in_

                    metrics_names = list(set(metrics_names_) - set(['roc_auc','accuracy','f1','recall','precision'])) if cmatrix is not None or len(np.unique(y_train)) > 2 else metrics_names_

                    result_append,outputs,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_test_bootstrap = test_models_bootstrap(type(trained_model),params,features,type(trained_scaler),type(trained_imputer),None,None,X_train,y_train,X_test,y_test,metrics_names,IDs_test,n_boot_train,n_boot_test,problem_type,threshold=None,cmatrix=cmatrix)
                    result_append.update({'task':task,'dimension':dimension,'y_label':y_label,'model_type':model_type,'random_seed':random_seed_test})
                    for metric in metrics_names:
                        try:
                            result_append.update({f'{metric}_dev':f"{best_model_dev[best_model_dev['metric'] == metric]['mean'].values[0]}, {best_model_dev[best_model_dev['metric'] == metric]['95_ci'].values[0]}"})
                        except:
                            continue
                    if results_test.empty:
                        results_test = pd.DataFrame(result_append,index=[0])
                    else:
                        results_test = pd.concat([results_test,pd.DataFrame(result_append,index=[0])],ignore_index=True)
    results_test.to_csv(Path(results_dir,f'best_models_{scoring}_test.csv'))
#outputs_filename = f'outputs_test_{model_name}.pkl'
#with open(Path(file.parent,outputs_filename),'wb') as f:
#    pickle.dump(outputst,f)


