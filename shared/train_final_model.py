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

project_name = 'tell_classifier'
scaler_name = 'StandardScaler'
scoring = 'norm_expected_cost'
extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if 'norm' in scoring else False

feature_selection_list = [True]
kfold_folder = '5_folds'
n_seeds_train = 10
n_seeds_test = 1

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path(Path.home(),'gonza','results',project_name)

best_classifiers = pd.read_csv(Path(results_dir,f'best_classifiers_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection.csv')) 

tasks = best_classifiers['task'].unique()
y_label = 'target'
id_col = 'id'

models_dict = {'lr':LogisticRegression,'knn':KNeighborsClassifier,'svc':SVC,'xgb':XGBClassifier}

for task,feature_selection in itertools.product(tasks,feature_selection_list):
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(task,dimension)
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection')
        if not feature_selection:
            path = str(path).replace('feature_selection','')

        random_seeds_test = [folder.name for folder in Path(path).iterdir() if folder.is_dir() and folder.name != f'final_model_{scoring}']
        
        if len(random_seeds_test) == 0:
            random_seeds_test = ['']

        for random_seed_test in random_seeds_test:
            path_to_data = Path(path,random_seed_test)
            
            Path(path_to_data,f'final_model_{scoring}').mkdir(parents=True,exist_ok=True)

            best_model = best_classifiers[(best_classifiers['task'] == task) & (best_classifiers['dimension'] == dimension) & (best_classifiers['random_seed_test'] == random_seed_test)]['model_type'].values[0] if random_seed_test != '' else best_classifiers[(best_classifiers['task'] == task) & (best_classifiers['dimension'] == dimension)]['model_type'].values[0]
        
        best_classifier = pd.read_csv(Path(path_to_data,f'all_models_{best_model}_dev.csv')).sort_values(f'{extremo}_{scoring}',ascending=ascending).reset_index(drop=True).head(1)

        all_features = [col for col in best_classifier.columns if any(x in col for x in task.split('_'))]
        features = [col for col in all_features if best_classifier[col].values[0] == 1]
        params = [col for col in best_classifier.columns if all(x not in col for x in  all_features + ['inf','sup','mean'] + [y_label,id_col,'Unnamed: 0','TIV','Edad','Lateralidad','Sexo','Educacion','Resonador'])]

        if task == 'P':
            params = list(set(params) - set([x for x in params if 'Animales_' in x]))
        elif task == 'Animales':
            params = list(set(params) - set([x for x in params if 'P_' in x]))
        elif dimension == 'neuropsico':
            params = list(set(params) - set([x for x in params if 'mmse' in x]))
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
        
        X_dev = pickle.load(open(Path(path_to_data,'X_dev.pkl'),'rb'))
        y_dev = pickle.load(open(Path(path_to_data,'y_dev.pkl'),'rb'))
        
        model.train(X_dev[features],y_dev)

        trained_model = model.model
        scaler = model.scaler
        imputer = model.imputer

        pickle.dump(trained_model,open(Path(path_to_data,'final_model',f'final_model_{task}_{dimension}.pkl'),'wb'))
        pickle.dump(scaler,open(Path(path_to_data,'final_model',f'scaler_{task}_{dimension}.pkl'),'wb'))
        pickle.dump(imputer,open(Path(path_to_data,'final_model',f'imputer_{task}_{dimension}.pkl'),'wb'))

        if best_model == 'svc':
            model.model.kernel = 'linear'
        
        model.train(X_dev[features],y_dev)

        if hasattr(model.model,'feature_importance'):
            feature_importance = model.model.feature_importance
            feature_importance = pd.DataFrame({'feature':features,'importance':feature_importance}).sort_values('importance',ascending=False)
            feature_importance.to_csv(Path(path_to_data,'final_model',f'feature_importance_{task}_{dimension}.csv'),index=False)
        elif hasattr(model.model,'coef_'):
            feature_importance = np.abs(model.model.coef_[0])
            coef = pd.DataFrame({'feature':features,'importance':feature_importance / np.sum(feature_importance)}).sort_values('importance',ascending=False)
            coef.to_csv(Path(path_to_data,'final_model',f'feature_importance_{task}_{dimension}.csv'),index=False)
        elif hasattr(model.model,'get_booster'):
            feature_importance = list(model.model.get_booster().get_score(importance_type='weight').values())
            try:
                feature_importance = pd.DataFrame({'feature':features,'importance':feature_importance/np.sum(feature_importance)}).sort_values('importance',ascending=False)
                feature_importance.to_csv(Path(path_to_data,'final_model',f'feature_importance_{task}_{dimension}.csv'),index=False)
            except:
                pass       
        else:
            print(task,dimension,f'No feature importance available for {best_model}')