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
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.naive_bayes import GaussianNB

from expected_cost.calibration import calibration_train_on_test
from psrcal.calibration import AffineCalLogLoss, AffineCalBrier, HistogramBinningCal

import pickle

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

correction = 'fdr_bh'

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config['project_name']
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
calibrate = config['calibrate']
avoid_stats = config['avoid_stats']
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
n_models = int(config['n_models'])
n_boot = int(config['n_boot'])
early_fusion = bool(config['early_fusion'])
problem_type = config['problem_type']
overwrite = bool(config['overwrite'])
parallel = bool(config['parallel'])
id_col = config['id_col']

if calibrate:
    calmethod = AffineCalLogLoss
    calparams = {'bias': True, 'priors': None}
else:
    calmethod = None
    calparams = None

home = Path(os.environ.get('HOME', Path.home()))
if 'Users/gp' in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path('D:/CNC_Audio/gonza/results', project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
if not isinstance(scoring_metrics,list):
    scoring_metrics = [scoring_metrics]
problem_type = main_config['problem_type'][project_name]
overwrite = bool(config['overwrite'])

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
            'xgb': XGBRegressor,
            'knnr': KNeighborsRegressor
        }
    }

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)
pearsons_results = pd.DataFrame(columns=['task','dimension','y_label','model_type','r','p_value'])

for scoring in scoring_metrics:    
    extremo = 'sup' if any(x in scoring for x in ['norm','error']) else 'inf'
    ascending = True if extremo == 'sup' else False

    best_models_file = f'best_models_{scoring.replace("_score","")}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection_shuffled_calibrated.csv'.replace('__','_')
    if not hyp_opt:
        best_models_file = best_models_file.replace('_hyp_opt','')
    if not feature_selection:
        best_models_file = best_models_file.replace('_feature_selection','')
    if not shuffle_labels:
        best_models_file = best_models_file.replace('_shuffled','')
    if not calibrate:
        best_models_file = best_models_file.replace('_calibrated','')
        
    best_models = pd.read_csv(Path(results_dir,best_models_file))
    
    tasks = best_models['task'].unique()
    y_labels = best_models['y_label'].unique()
    dimensions = best_models['dimension'].unique()

    for task,dimension,y_label in itertools.product(tasks,dimensions,y_labels):
        print(task,dimension,y_label)
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
        
        if not Path(path).exists():
            continue
        
        random_seeds_test = [folder.name for folder in Path(path).iterdir() if folder.is_dir() and 'random_seed' in folder.name]

        random_seeds_test.append('')

        for random_seed_test in random_seeds_test:
            path_to_data = Path(path,random_seed_test)
            
            if Path(results_dir,f'final_models',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','final_model.pkl').exists() and not overwrite:
                print(f'Final model already exists for {task} {dimension} {y_label} {random_seed_test}. Skipping.')
                continue

            model_type = best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['random_seed_test'] == random_seed_test) & (best_models['y_label'] == y_label)]['model_type'].values[0] if random_seed_test != '' else best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label)]['model_type'].values[0]
            print(model_type)
            model_index = best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['random_seed_test'] == random_seed_test) & (best_models['y_label'] == y_label)]['model_index'].values[0] if random_seed_test != '' else best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label)]['model_index'].values[0]
            scoring = scoring.replace("_score","")
            try:
                if calibrate:
                    best_model = pd.read_csv(Path(path_to_data,f'all_models_{model_type}_test_calibrated.csv')).sort_values(f'{extremo}_{scoring}_dev',ascending=ascending).reset_index(drop=True).head(1)
                else:
                    best_model = pd.read_csv(Path(path_to_data,f'all_models_{model_type}_test.csv')).sort_values(f'{extremo}_{scoring}_dev',ascending=ascending).reset_index(drop=True).head(1)

            except:
                if calibrate:
                    best_model = pd.read_csv(Path(path_to_data,f'all_models_{model_type}_dev_bca_calibrated.csv')).sort_values(f'{scoring}_{extremo}',ascending=ascending).reset_index(drop=True).head(1)
                else:
                    best_model = pd.read_csv(Path(path_to_data,f'all_models_{model_type}_dev_bca.csv')).sort_values(f'{scoring}_{extremo}',ascending=ascending).reset_index(drop=True).head(1)

            all_features = [col for col in best_model.columns if any(f'{x}__{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__'))) or col =='group']
            features = [col for col in all_features if best_model[col].values[0] == 1]
            params = [col for col in best_model.columns if all(x not in col for x in  all_features + ['inf','sup','mean','ic'] + [y_label,id_col,'Unnamed: 0','threshold','index'])]

            params_dict = {param:best_model.loc[0,param] for param in params if str(best_model.loc[0,param]) != 'nan'}

            if 'gamma' in params_dict.keys():
                try: 
                    params_dict['gamma'] = float(params_dict['gamma'])
                except:
                    pass

            if 'random_state' in params_dict.keys():
                params_dict['random_state'] = int(params_dict['random_state'])
            
            try:
                model = utils.Model(models_dict[problem_type][model_type](**params_dict),StandardScaler,KNNImputer,calmethod,calparams)
            except:
                params = list(set(params) - set([x for x in params if any(x in params for x in ['Unnamed: 0'])]))
                params_dict = {param:best_model.loc[0,param] for param in params if str(best_model.loc[0,param]) != 'nan'}
                model = utils.Model(models_dict[problem_type][model_type](**params_dict),StandardScaler,KNNImputer,calmethod,calparams)
            
            X_dev = pickle.load(open(Path(path_to_data,'X_dev.pkl'),'rb'))
            y_dev = pickle.load(open(Path(path_to_data,'y_dev.pkl'),'rb'))
            if not Path(path_to_data,f'outputs_{model_type}.pkl').exists():   
                continue

            outputs_dev = pickle.load(open(Path(path_to_data,f'outputs_{model_type}.pkl'),'rb'))[:,model_index]

            if calibrate:
                scores = np.concatenate([outputs_dev[j,r] for j in range(outputs_dev.shape[0]) for r in range(outputs_dev.shape[1])])
                targets = np.concatenate([y_dev[j,r] for j in range(y_dev.shape[0]) for r in range(y_dev.shape[1])])

                _,calmodel = calibration_train_on_test(scores,targets,calmethod,calparams,return_model=True)
            if not isinstance(X_dev,pd.DataFrame):
                X_dev = pd.DataFrame(X_dev[0,0],columns=all_features)
            model.train(X_dev[features],y_dev[0,0])

            trained_model = model.model
            scaler = model.scaler
            imputer = model.imputer

            feature_importance_file = f'feature_importance_{model_type}_shuffled_calibrated.csv'.replace('__','_')

            if not shuffle_labels:
                feature_importance_file = feature_importance_file.replace('_shuffled','')
            if not calibrate:
                feature_importance_file = feature_importance_file.replace('_calibrated','')

            Path(results_dir,f'feature_importance_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '').mkdir(parents=True,exist_ok=True)
            Path(results_dir,f'final_model_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '').mkdir(parents=True,exist_ok=True)
            
            with open(Path(results_dir,f'final_model_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','final_model.pkl'),'wb') as f:
                pickle.dump(trained_model,f)
            with open(Path(results_dir,f'final_model_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',f'scaler.pkl'),'wb') as f:
                pickle.dump(scaler,f)
            with open(Path(results_dir,f'final_model_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',f'imputer.pkl'),'wb') as f:
                pickle.dump(imputer,f)
            if calibrate:
                with open(Path(results_dir,f'final_model_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',f'calmodel.pkl'),'wb') as f:
                    pickle.dump(calmodel,f)
            
            if model_type == 'svc':
                model.model.kernel = 'linear'
        
            model.train(X_dev[features],y_dev[0,0])

            if hasattr(model.model,'feature_importance'):
                feature_importance = model.model.feature_importance
                feature_importance = pd.DataFrame({'feature':features,'importance':feature_importance}).sort_values('importance',ascending=False)
                feature_importance.to_csv(Path(results_dir,f'feature_importance_{scoring}',feature_importance_file),index=False)
            elif hasattr(model.model,'coef_'):
                feature_importance = np.abs(model.model.coef_[0])
                coef = pd.DataFrame({'feature':features,'importance':feature_importance / np.sum(feature_importance)}).sort_values('importance',ascending=False)
                coef.to_csv(Path(results_dir,f'feature_importance_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',feature_importance_file),index=False)
            elif hasattr(model.model,'get_booster'):

                feature_importance = pd.DataFrame({'feature':features,'importance':model.model.feature_importances_}).sort_values('importance',ascending=False)
                feature_importance.to_csv(Path(results_dir,f'feature_importance_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',feature_importance_file),index=False)
            else:
                print(task,dimension,f'No feature importance available for {model_type}')
            
            if problem_type == 'reg':
                sns.set_theme(style="whitegrid")  # Fondo blanco con grid sutil
                plt.rcParams.update({
                    "font.family": "DejaVu Sans",
                    "axes.titlesize": 16,
                    "axes.labelsize": 14,
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12
                })
                
                Path(results_dir,f'plots_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '').mkdir(parents=True,exist_ok=True)
                IDs = pickle.load(open(Path(path_to_data,'IDs_dev.pkl'),'rb'))

                predictions = pd.DataFrame({'ID':IDs.flatten(),'y_pred':outputs_dev.flatten(),'y_true':y_dev.flatten()})
                predictions = predictions.drop_duplicates('ID')

                with open(Path(results_dir,f'final_model_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',f'predictions_dev.pkl'),'wb') as f:
                    pickle.dump(predictions,f)
                predictions.to_csv(Path(results_dir,f'final_model_{scoring}',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',f'predictions_dev.csv'),index=False)
                
                r, p = pearsonr(predictions['y_true'], predictions['y_pred'])

                plt.figure(figsize=(8, 6))
                sns.regplot(
                    x='y_true', y='y_pred', data=predictions,
                    scatter_kws={'alpha': 0.6, 's': 50, 'color': '#1f77b4'},  # color base
                    line_kws={'color': 'darkred', 'linewidth': 2}
                )

                # Línea de identidad (y = x)
                #min_val = min(predictions['y_true'].min(), predictions['y_pred'].min())
                #max_val = max(predictions['y_true'].max(), predictions['y_pred'].max())
                #plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray', linewidth=1)

                #plt.xlim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
                #plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))

                # Etiquetas y título
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
                pearsons_results.loc[len(pearsons_results)] = [task, dimension, y_label, model_type, r, p]

                save_path = Path(results_dir, f'plots_{scoring}', task, dimension, y_label,
                                stat_folder, scoring,
                                'hyp_opt' if hyp_opt else '',
                                'feature_selection' if feature_selection else '',
                                f'{task}_{y_label}_{dimension}_{model_type}.png')
                save_path.parent.mkdir(parents=True, exist_ok=True)

                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close()

    if problem_type == 'reg':
        pearsons_results_file = f'pearons_results_{scoring}_{stat_folder}_hyp_opt_feature_selection_shuffled.csv'.replace('__','_')
        if not hyp_opt:
            pearsons_results_file = pearsons_results_file.replace('_hyp_opt','')
        if not feature_selection:
            pearsons_results_file = pearsons_results_file.replace('_feature_selection','')
        if not shuffle_labels:
            pearsons_results_file = pearsons_results_file.replace('_shuffled','')

        p_vals = pearsons_results['p_value'].values
        reject, p_vals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method=correction)
        pearsons_results['p_value_corrected'] = p_vals_corrected
        pearsons_results['correction_method'] = correction

        pearsons_results.to_csv(Path(results_dir,pearsons_results_file),index=False)