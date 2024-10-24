import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

from sklearn.linear_model import Ridge as RR
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import Lasso

from xgboost import XGBClassifier 

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

from expected_cost.ec import *
from psrcal import *

project_name = 'GeroApathy'
tasks = ['Fugu']
id_col = 'id'

scaler_name = 'StandardScaler'

imputer = KNNImputer

l2ocv = False

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

hyp_opt_list = [True]
feature_selection_list = [True]
bootstrap_list = [True]

boot_test = 10
boot_train = 10

n_seeds_test = 1

scaler = StandardScaler if scaler_name == 'StandardScaler' else MinMaxScaler

models_dict = {'ridge': RR,
               'lasso':Lasso,
               'knn': KNN,
               'svm': SVR,
               }

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

results_dir = Path(str(data_dir).replace('data','results'))

metrics_names = ['r2_score','mean_squared_error','mean_absolute_error']

scoring = 'r2_score'
extremo = 'sup' if 'error' in scoring else 'inf'
ascending = True if 'error' in scoring else False

for task in tasks:
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,'mean_std')

        y_labels = [folder.name for folder in path.iterdir() if folder.is_dir()]
        for y_label,hyp_opt,feature_selection in itertools.product(y_labels,hyp_opt_list,feature_selection_list):
            print(task,dimension,y_label)
            
            data = pd.read_csv(Path(data_dir,'all_data.csv'))

            data = data.dropna(subset=[y_label])

            path_to_results = Path(path,y_label,'hyp_opt','feature_selection')
            
            path_to_results = Path(str(path_to_results).replace('no_hyp_opt', 'hyp_opt')) if hyp_opt else path_to_results
            path_to_results = Path(str(path_to_results).replace('feature_selection', '')) if not feature_selection else path_to_results
            random_seeds_test = [folder.name for folder in path_to_results.iterdir() if folder.is_dir()]

            for random_seed_test in random_seeds_test:
                X_dev = pickle.load(open(Path(path_to_results,random_seed_test,'X_dev.pkl'),'rb'))
                X_test = pickle.load(open(Path(path_to_results,random_seed_test,'X_test.pkl'),'rb'))
                y_dev = pickle.load(open(Path(path_to_results,random_seed_test,'y_dev.pkl'),'rb'))
                y_test = pickle.load(open(Path(path_to_results,random_seed_test,'y_test.pkl'),'rb'))
                IDs_test = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_test.pkl'),'rb'))
                IDs_dev = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_dev.pkl'),'rb'))

                files = [file for file in Path(path_to_results,random_seed_test).iterdir() if 'all_models_' in file.stem and 'dev' in file.stem]
                
                for file in files:
                    model_name = file.stem.split('_')[-2]

                    print(model_name)
                    
                    #if Path(file.parent,f'all_models_{model_name}_test.csv').exists():
                    #    continue
                    
                    results = pd.read_excel(file) if file.suffix == '.xlsx' else pd.read_csv(file)

                    results = results.sort_values(by=f'{extremo}_{scoring}',ascending=ascending).reset_index(drop=True)
                    results_test = pd.DataFrame()

                    for r, row in tqdm.tqdm(results.iterrows()):
                        all_features = [col for col in results.columns if any(f'{x}_{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__')))] 

                        results_r = row.dropna().to_dict()
                                        
                        params = dict((key,value) for (key,value) in results_r.items() if all (x not in key for x in ['inf','mean','sup',id_col,'threshold','Unnamed: 0'] + all_features + y_labels))

                        if 'gamma' in params.keys():
                            try: 
                                params['gamma'] = float(params['gamma'])
                            except:
                                pass
                        if 'random_state' in params.keys():
                            params['random_state'] = int(params['random_state'])   

                        features = [col for col in all_features if results_r[col] == 1]
                        features_dict = {col:results_r[col] for col in all_features}

                        mod = Model(models_dict[model_name](**params),scaler,imputer)
                        
                        if Path(path_to_results,random_seed_test,'X_dev.pkl').exists() == False:
                            pickle.dump(X_dev,open(Path(path_to_results,random_seed_test,'X_dev.pkl'),'wb'))
                            pickle.dump(X_test,open(Path(path_to_results,random_seed_test,'X_test.pkl'),'wb'))
                            pickle.dump(y_dev,open(Path(path_to_results,random_seed_test,'y_dev.pkl'),'wb'))
                            pickle.dump(y_test,open(Path(path_to_results,random_seed_test,'y_test.pkl'),'wb'))
                        
                        orig_features = X_dev.columns
                        for feature in orig_features:
                            if task in feature:
                                continue
                            X_dev[f'{task}_{feature}'] = X_dev[feature] 
                            X_test[f'{task}_{feature}'] = X_test[feature]

                            #X_dev.drop(columns=[feature],inplace=True)
                            #X_test.drop(columns=[feature],inplace=True)

                        metrics_test_bootstrap,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_test_bootstrap = test_model(mod,X_dev[features],y_dev,X_test[features],y_test,metrics_names,IDs_test,boot_train,boot_test,problem_type='reg')

                        result_append = params.copy()
                        result_append.update(features_dict)
                        
                        for metric in metrics_names:
                            inf = np.percentile(metrics_test_bootstrap[metric],2.5).round(2)
                            mean = np.mean(metrics_test_bootstrap[metric]).round(2)
                            sup = np.percentile(metrics_test_bootstrap[metric],97.5).round(2)

                            result_append[f'inf_{metric}_test'] = inf
                            result_append[f'mean_{metric}_test'] = mean
                            result_append[f'sup_{metric}_test'] = sup
                            
                            result_append[f'inf_{metric}_dev'] = np.round(results_r[f'inf_{metric}'],2)
                            result_append[f'mean_{metric}_dev'] = np.round(results_r[f'mean_{metric}'],2)
                            result_append[f'sup_{metric}_dev'] = np.round(results_r[f'sup_{metric}'],2)

                        if results_test.empty:
                            results_test = pd.DataFrame(columns=result_append.keys())
                        
                        results_test.loc[len(results_test.index),:] = result_append

                    pd.DataFrame(results_test).to_csv(Path(file.parent,f'all_models_{model_name}_test.csv'),index=False)