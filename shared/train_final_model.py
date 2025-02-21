import sys, itertools, json, os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

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
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
n_models = int(config["n_models"])
n_boot = int(config["n_boot"])
early_fusion = bool(config["early_fusion"])

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
scoring_metrics = [main_config['scoring_metrics'][project_name]]
problem_type = main_config['problem_type'][project_name]

models_dict = {'lr':LogisticRegression,'knn':KNeighborsClassifier,'svc':SVC,'xgb':XGBClassifier,'knnc':KNeighborsClassifier}

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for scoring in scoring_metrics:
    extremo = 'sup' if any(x in scoring for x in ['norm','error']) else 'inf'
    ascending = True if extremo == 'sup' else False

    best_models = pd.read_csv(Path(str(Path(results_dir,f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection_shuffled.csv')).replace('__','')))
    
    tasks = best_models['task'].unique()
    y_labels = best_models['y_label'].unique()
    id_col = 'id'

    for task,y_label in itertools.product(tasks,y_labels):
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            print(task,dimension)
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
            
            if not Path(path).exists():
                continue

            random_seeds_test = [folder.name for folder in Path(path).iterdir() if folder.is_dir() and 'random_seed' in folder.name]
            
            if len(random_seeds_test) == 0:
                random_seeds_test = ['']

            for random_seed_test in random_seeds_test:
                path_to_data = Path(path,random_seed_test)
                
                Path(path_to_data,f'final_model_{scoring}').mkdir(parents=True,exist_ok=True)

                best_model = best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['random_seed_test'] == random_seed_test)]['model_type'].values[0] if random_seed_test != '' else best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension)]['model_type'].values[0]
            
            try:
                best_classifier = pd.read_csv(Path(path_to_data,f'all_models_{scoring}_{best_model}_test.csv')).sort_values(f'{scoring}_{extremo}',ascending=ascending).reset_index(drop=True).head(1)
            except:
                best_classifier = pd.read_csv(Path(path_to_data,f'all_models_{best_model}_dev_bca.csv')).sort_values(f'{scoring}_{extremo}',ascending=ascending).reset_index(drop=True).head(1)

            all_features = [col for col in best_classifier.columns if any(f'{x}__{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__')))]
            features = [col for col in all_features if best_classifier[col].values[0] == 1]
            params = [col for col in best_classifier.columns if all(x not in col for x in  all_features + ['inf','sup','mean','ic'] + [y_label,id_col,'Unnamed: 0','threshold','index'])]

            params_dict = {param:best_classifier.loc[0,param] for param in params if str(best_classifier.loc[0,param]) != 'nan'}

            if 'gamma' in params_dict.keys():
                try: 
                    params_dict['gamma'] = float(params_dict['gamma'])
                except:
                    pass

            if 'random_state' in params_dict.keys():
                params_dict['random_state'] = int(params_dict['random_state'])
            
            try:
                model = utils.Model(models_dict[best_model](**params_dict),StandardScaler,KNNImputer)
            except:
                params = list(set(params) - set([x for x in params if any(x in params for x in ['Unnamed: 0'])]))
                params_dict = {param:best_classifier.loc[0,param] for param in params if str(best_classifier.loc[0,param]) != 'nan'}
                model = utils.Model(models_dict[best_model](**params_dict),StandardScaler,KNNImputer)
            
            X_dev = pickle.load(open(Path(path_to_data,'X_dev.pkl'),'rb')).squeeze(axis=0)[0]
            y_dev = pickle.load(open(Path(path_to_data,'y_dev.pkl'),'rb')).squeeze(axis=0)[0]
            if not isinstance(X_dev,pd.DataFrame):
                X_dev = pd.DataFrame(X_dev,columns=all_features)
            model.train(X_dev[features],y_dev)

            trained_model = model.model
            scaler = model.scaler
            imputer = model.imputer

            Path(path_to_data,f'final_model_{scoring}').mkdir(parents=True,exist_ok=True)
            with open(Path(path_to_data,f'final_model_{scoring}',f'final_model.pkl'),'wb') as f:
                pickle.dump(trained_model,f)
            with open(Path(path_to_data,f'final_model_{scoring}',f'scaler.pkl'),'wb') as f:
                pickle.dump(scaler,f)
            with open(Path(path_to_data,f'final_model_{scoring}',f'imputer.pkl'),'wb') as f:
                pickle.dump(imputer,f)
            
            if best_model == 'svc':
                model.model.kernel = 'linear'
        
            model.train(X_dev[features],y_dev)

            if hasattr(model.model,'feature_importance'):
                feature_importance = model.model.feature_importance
                feature_importance = pd.DataFrame({'feature':features,'importance':feature_importance}).sort_values('importance',ascending=False)
                feature_importance.to_csv(Path(path_to_data,f'final_model_{scoring}',f'feature_importance.csv'),index=False)
            elif hasattr(model.model,'coef_'):
                feature_importance = np.abs(model.model.coef_[0])
                coef = pd.DataFrame({'feature':features,'importance':feature_importance / np.sum(feature_importance)}).sort_values('importance',ascending=False)
                coef.to_csv(Path(path_to_data,f'final_model_{scoring}',f'feature_importance.csv'),index=False)
            elif hasattr(model.model,'get_booster'):

                feature_importance = pd.DataFrame({'feature':features,'importance':model.model.feature_importances_}).sort_values('importance',ascending=False)
                feature_importance.to_csv(Path(path_to_data,f'final_model_{scoring}',f'feature_importance.csv'),index=False)
            else:
                print(task,dimension,f'No feature importance available for {best_model}')