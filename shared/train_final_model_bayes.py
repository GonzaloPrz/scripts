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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, LeavePOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.multitest import multipletests
from pingouin import partial_corr

from expected_cost.calibration import calibration_train_on_test
from psrcal.calibration import AffineCalLogLoss, AffineCalBrier, HistogramBinningCal

import pickle

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

from expected_cost.ec import CostMatrix

config = json.load(Path(Path(__file__).parent,'config.json').open())
project_name = config["project_name"]
scaler_name = config['scaler_name']
n_folds = int(config['n_folds_inner'])
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = config['n_iter'] > 0
feature_selection = bool(config['feature_selection'])
filter_outliers = bool(config['filter_outliers'])
calibrate = bool(config["calibrate"])
n_iter = int(config["n_iter"])
init_points = int(config["init_points"])
scaler_name = config['scaler_name']
id_col = config['id_col']

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
data_file = main_config['data_file'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
problem_type = main_config['problem_type'][project_name]
cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None
thresholds = main_config['thresholds'][project_name]
covars = main_config['covars'][project_name] if problem_type == 'reg' else []

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
            #'elastic': ElasticNet,
            #'svr': SVR,
            'knnr': KNeighborsRegressor,
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
data_dir = str(results_dir).replace('results','data')
pearsons_results = pd.DataFrame(columns=['task','dimension','y_label','model_type','r','p_value','n','95_ci','covars'])

correction = 'fdr_bh'

covariates = pd.read_csv(Path(data_dir,data_file))[[id_col]+covars]

for scoring,threshold in itertools.product(scoring_metrics,thresholds):
    if str(threshold) == 'None':
        threshold = None
    filename = f'best_best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffled_calibrated_bayes.csv'.replace('__','_')

    if not hyp_opt:
        filename = filename.replace('_hyp_opt','')
    if not feature_selection:
        filename = filename.replace('_feature_selection','')
    if not shuffle_labels:
        filename = filename.replace('_shuffled','')
    if not calibrate:
        filename = filename.replace('_calibrated','')

    best_models = pd.read_csv(Path(results_dir,filename))

    for r, row in best_models.iterrows():
        task = row.task
        dimension = row.dimension
        y_label = row.y_label
        model_type = row.model_type

        print(task,dimension,y_label,model_type)
        path_to_results = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'bayes',scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
        random_seeds = [folder.name for folder in path_to_results.iterdir() if 'random_seed' in folder.name]
        random_seeds.append('')
        
        for random_seed in random_seeds:
            if Path(results_dir,f'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',f'model_{model_type}.pkl').exists() and not overwrite:
                print('Model already exists')
                continue
            
            if not Path(path_to_results,random_seed,f'all_models_{model_type}.csv').exists():
                continue

            all_models = pd.read_csv(Path(path_to_results,random_seed,f'all_models_{model_type}.csv'))
            
            features = [col for col in all_models.columns if f'{task}__' in col]
            
            X_train = pickle.load(open(Path(path_to_results,random_seed,'X_train.pkl'),'rb'))
            y_train = pickle.load(open(Path(path_to_results,random_seed,'y_train.pkl'),'rb'))
            y_dev = pickle.load(open(Path(path_to_results,random_seed,'y_dev.pkl'),'rb'))
            outputs_dev = pickle.load(open(Path(path_to_results,random_seed,f'outputs_{model_type}.pkl'),'rb'))

            if n_folds == -1:
                CV = LeaveOneOut()
                n_max = X_train.shape[0] - 1
            elif n_folds == 0:
                n_folds = np.floor(X_train.shape[0]/np.unique(y_train).shape[0])
                n_max = X_train.shape[0] - np.unique(y_train).shape[0]
                CV = (StratifiedKFold(n_splits=int(n_folds), shuffle=True)
                            if config['stratify'] and problem_type == 'clf'
                            else KFold(n_splits=n_folds, shuffle=True))  
            else:
                CV = (StratifiedKFold(n_splits=int(n_folds), shuffle=True)
                            if config['stratify'] and problem_type == 'clf'
                            else KFold(n_splits=n_folds, shuffle=True))  
                n_max = int(X_train.shape[0]*(1-1/n_folds))
                
            hyperp['knnc']['n_neighbors'] = (1,n_max)
            hyperp['knnr']['n_neighbors'] = (1,n_max)

            model_class = models_dict[problem_type][model_type]
            scaler = StandardScaler if scaler_name == 'StandardScaler' else MinMaxScaler
            imputer = KNNImputer
            if cmatrix is None:
                cmatrix = CostMatrix.zero_one_costs(K=len(np.unique(y_train)))
            
            if int(config["n_iter"]):
                best_params, best_score = utils.tuning(model_class,scaler,imputer,X_train,y_train,hyperp[model_type],CV,init_points=int(config['init_points']),n_iter=n_iter,scoring=scoring,problem_type=problem_type,cmatrix=cmatrix,priors=None,threshold=threshold,calmethod=None,calparams=None)
            else: 
                best_params = model_class().get_params()

            if problem_type == 'clf' and model_class == SVC:
                best_params['probability'] = True
            
            if 'random_state' in best_params.keys():
                best_params['random_state'] = 42

            model = utils.Model(model_class(**best_params),scaler,imputer)
            
            best_features = utils.rfe(utils.Model(model_class(**best_params),scaler,imputer,None,None),X_train,y_train,CV,scoring,problem_type,cmatrix=cmatrix,priors=None,threshold=threshold)[0] if feature_selection else X_train.columns
            
            model.train(X_train[best_features],y_train)

            feature_importance_file = f'feature_importance_{model_type}_shuffled_calibrated.csv'.replace('__','_')

            if not shuffle_labels:
                feature_importance_file = feature_importance_file.replace('_shuffled','')
            if not calibrate:
                feature_importance_file = feature_importance_file.replace('_calibrated','')

            Path(results_dir,f'feature_importance_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '').mkdir(parents=True,exist_ok=True)
            Path(results_dir,f'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '').mkdir(parents=True,exist_ok=True)
            
            if hasattr(model.model,'feature_importance'):
                feature_importance = model.model.feature_importance
                feature_importance = pd.DataFrame({'feature':best_features,'importance':feature_importance}).sort_values('importance',ascending=False)
                feature_importance.to_csv(Path(results_dir,f'feature_importance_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',feature_importance_file),index=False)
            elif hasattr(model.model,'coef_'):
                feature_importance = np.abs(model.model.coef_[0])
                coef = pd.DataFrame({'feature':best_features,'importance':feature_importance / np.sum(feature_importance)}).sort_values('importance',ascending=False)
                coef.to_csv(Path(results_dir,f'feature_importance_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',feature_importance_file),index=False)
            elif hasattr(model.model,'get_booster'):

                feature_importance = pd.DataFrame({'feature':best_features,'importance':model.model.feature_importances_}).sort_values('importance',ascending=False)
                feature_importance.to_csv(Path(results_dir,f'feature_importance_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',feature_importance_file),index=False)
            else:
                print(task,dimension,f'No feature importance available for {model_type}')
            pickle.dump(model.model,open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',f'model_{model_type}.pkl'),'wb'))
            pickle.dump(model.scaler,open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',f'scaler_{model_type}.pkl'),'wb'))
            pickle.dump(model.imputer,open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',f'imputer_{model_type}.pkl'),'wb'))
            
            if problem_type == 'reg':
                sns.set_theme(style="whitegrid")  # Fondo blanco con grid sutil
                plt.rcParams.update({
                    "font.family": "DejaVu Sans",
                    "axes.titlesize": 16,
                    "axes.labelsize": 14,
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12
                })
                
                Path(results_dir,f'plots',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '').mkdir(parents=True,exist_ok=True)
                IDs = pickle.load(open(Path(path_to_results,'IDs_dev.pkl'),'rb'))

                try:
                    predictions = pd.DataFrame({'ID':IDs.flatten(),'y_pred':outputs_dev.flatten(),'y_true':y_dev.flatten()})
                except:
                    continue
                predictions = predictions.drop_duplicates('ID')

                predictions = pd.merge(predictions,covariates,on=id_col,how='inner')

                with open(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',f'predictions_dev.pkl'),'wb') as f:
                    pickle.dump(predictions,f)
                predictions.to_csv(Path(results_dir,'final_models_bayes',task,dimension,y_label,stat_folder,scoring,config["bootstrap_method"],'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',f'predictions_dev.csv'),index=False)
                try:
                    n, r, ci, p = partial_corr(data=predictions,x='y_pred',y='y_true',covar=covars,method='pearson')
                except:
                    r, p = pearsonr(predictions['y_true'], predictions['y_pred'])
                    n = predictions.shape[0]
                    ci = np.nan

                plt.figure(figsize=(8, 6))
                sns.regplot(
                    x='y_true', y='y_pred', data=predictions,
                    scatter_kws={'alpha': 0.6, 's': 50, 'color': '#1f77b4'},  # color base
                    line_kws={'color': 'darkred', 'linewidth': 2}
                )

                plt.xlabel('True Value')
                plt.ylabel('Predicted Value')
                plt.title(f'{dimension} | {y_label.replace("_"," ")}', fontsize=16, pad=15)

                # Añadir estadística en esquina superior izquierda
                plt.text(0.05, 0.95,
                        f'$r$ = {r:.2f}\n$p$ = {p:.2e}',
                        fontsize=12,
                        transform=plt.gca().transAxes,
                        verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

                # Guardar resultado y cerrar
                pearsons_results.loc[len(pearsons_results)] = [task, dimension, y_label, model_type, r, p, n, ci,str(covars)]

                save_path = Path(results_dir, f'plots', task, dimension, y_label,
                                stat_folder, scoring,config["bootstrap_method"],
                                'hyp_opt' if hyp_opt else '','bayes',scoring,
                                'feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',
                                f'{task}_{y_label}_{dimension}_{model_type}.png')
                save_path.parent.mkdir(parents=True, exist_ok=True)

                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close()

    if problem_type == 'reg':
        pearsons_results_file = f'pearons_results_{scoring}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffled.csv'.replace('__','_')
        if not hyp_opt:
            pearsons_results_file = pearsons_results_file.replace('_hyp_opt','')
        if not feature_selection:
            pearsons_results_file = pearsons_results_file.replace('_feature_selection','')
        if not shuffle_labels:
            pearsons_results_file = pearsons_results_file.replace('_shuffled','')

        if Path(results_dir,pearsons_results_file).exists() and not overwrite:
            continue

        p_vals = pearsons_results['p_value'].values
        reject, p_vals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method=correction)
        pearsons_results['p_value_corrected'] = p_vals_corrected
        pearsons_results['correction_method'] = correction

        pearsons_results.to_csv(Path(results_dir,pearsons_results_file),index=False)