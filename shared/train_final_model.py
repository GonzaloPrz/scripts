import sys
from pathlib import Path
import pandas as pd
import itertools
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

import pickle

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

project_name = 'arequipa'
scaler_name = 'StandardScaler'

scoring = 'roc_auc'

extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if 'norm' in scoring else False

feature_selection_list = [True]
shuffle_labels_list = [False]

kfold_folder = '5_folds'
n_seeds_train = 10
n_seeds_test = 1

models_dict = {'lr':LogisticRegression,'knn':KNeighborsClassifier,'svc':SVC,'xgb':XGBClassifier,'knnc':KNeighborsClassifier}

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for shuffle_labels in shuffle_labels_list:
    best_models = pd.read_csv(Path(results_dir,f'best_models_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection_shuffled.csv'))  if shuffle_labels else pd.read_csv(Path(results_dir,f'best_models_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection.csv')) 

    tasks = best_models['task'].unique()
    y_labels = best_models['y_label'].unique()
    id_col = 'id'

    for task,y_label,feature_selection in itertools.product(tasks,y_labels,feature_selection_list):
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            print(task,dimension)
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection','shuffle')
            if not feature_selection:
                path = str(path).replace('feature_selection','')
            if not shuffle_labels:
                path = str(path).replace('shuffle','')
            
            if not Path(path).exists():
                continue

            random_seeds_test = [folder.name for folder in Path(path).iterdir() if folder.is_dir() and all(x not in folder.name for x in ['shuffle','final_model'])]
            
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
                model = Model(models_dict[best_model](**params_dict),StandardScaler,KNNImputer)
            except:
                params = list(set(params) - set([x for x in params if any(x in params for x in ['Unnamed: 0'])]))
                params_dict = {param:best_classifier.loc[0,param] for param in params if str(best_classifier.loc[0,param]) != 'nan'}
                model = Model(models_dict[best_model](**params_dict),StandardScaler,KNNImputer)
            
            try:
                X_dev = pickle.load(open(Path(path_to_data,'X_dev.pkl'),'rb'))
                #y_dev = pickle.load(open(Path(path_to_data,'y_dev.pkl'),'rb'))
                y_dev = pd.read_csv(Path('D:','CNC_Audio','gonza','data',project_name,'data_matched_group.csv'))[y_label]
                if not isinstance(X_dev,pd.DataFrame):
                    X_dev = pd.DataFrame(X_dev.squeeze(axis=0),columns=all_features)
                model.train(X_dev[features],y_dev.squeeze(axis=0))

                trained_model = model.model
                scaler = model.scaler
                imputer = model.imputer

                Path(path_to_data,f'final_model_{scoring}').mkdir(parents=True,exist_ok=True)
                
                pickle.dump(trained_model,open(Path(path_to_data,f'final_model_{scoring}',f'final_model_{task}_{dimension}.pkl'),'wb'))
                pickle.dump(scaler,open(Path(path_to_data,f'final_model_{scoring}',f'scaler_{task}_{dimension}.pkl'),'wb'))
                pickle.dump(imputer,open(Path(path_to_data,f'final_model_{scoring}',f'imputer_{task}_{dimension}.pkl'),'wb'))

                if best_model == 'svc':
                    model.model.kernel = 'linear'

                if task == 'Animales__P' and dimension == 'properties':
                    print('.')                
                model.train(X_dev[features],y_dev.squeeze(axis=0))

                if hasattr(model.model,'feature_importance'):
                    feature_importance = model.model.feature_importance
                    feature_importance = pd.DataFrame({'feature':features,'importance':feature_importance}).sort_values('importance',ascending=False)
                    feature_importance.to_csv(Path(path_to_data,f'final_model_{scoring}',f'feature_importance_{task}_{dimension}.csv'),index=False)
                elif hasattr(model.model,'coef_'):
                    feature_importance = np.abs(model.model.coef_[0])
                    coef = pd.DataFrame({'feature':features,'importance':feature_importance / np.sum(feature_importance)}).sort_values('importance',ascending=False)
                    coef.to_csv(Path(path_to_data,f'final_model_{scoring}',f'feature_importance_{task}_{dimension}.csv'),index=False)
                elif hasattr(model.model,'get_booster'):

                    feature_importance = pd.DataFrame({'feature':features,'importance':model.model.feature_importances_}).sort_values('importance',ascending=False)
                    feature_importance.to_csv(Path(path_to_data,f'final_model_{scoring}',f'feature_importance_{task}_{dimension}.csv'),index=False)
                else:
                    print(task,dimension,f'No feature importance available for {best_model}')
            except:
                print(f'Error with {task} - {dimension}')
                pass