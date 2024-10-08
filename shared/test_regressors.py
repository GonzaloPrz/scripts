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

n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

hyp_opt_list = [True]
feature_selection_list = [True]
bootstrap_list = [True]

boot_test = 100
boot_train = 0

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

random_seeds_test = [0]

n_models = 10

neuro_data = pd.read_excel(Path(data_dir,f'nps_data_filtered_no_missing.xlsx'))

dimensions = ['pitch','talking_intervals','pitch__talking_intervals']

sort_by = 'mean_absolute_error'
extremo = 'sup' if 'error' in sort_by else 'inf'
ascending = True if 'error' in sort_by else False

data_features = pd.read_csv(Path(data_dir,'features_data.csv'))

for task in tasks:
    for dimension in dimensions:
        print(task,dimension)
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test' if len(random_seeds_test) > 0 else '')

        #y_labels = [folder.name for folder in path.iterdir() if folder.is_dir()]
        y_labels = ['AES_Total_Score','Well_Being_Total_Score_V']
        for y_label,hyp_opt,feature_selection,bootstrap in itertools.product(y_labels,hyp_opt_list,feature_selection_list,bootstrap_list):
            if y_label == 'Well_Being_Total_Score_V':
                continue
            data = pd.merge(data_features,neuro_data,on=id_col,how='inner')
            
            data = data.dropna(subset=[y_label])

            path_to_results = Path(path,y_label,'hyp_opt','feature_selection','bootstrap')
            
            path_to_results = Path(str(path_to_results).replace('no_hyp_opt', 'hyp_opt')) if hyp_opt else path_to_results
            path_to_results = Path(str(path_to_results).replace('feature_selection', '')) if not feature_selection else path_to_results
            path_to_results = Path(str(path_to_results).replace('bootstrap', '')) if not bootstrap else path_to_results

            for random_seed_test in random_seeds_test:
                X_dev = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','X_dev.pkl'),'rb'))
                X_test = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','X_test.pkl'),'rb'))
                y_dev = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','y_dev.pkl'),'rb'))
                y_test = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','y_test.pkl'),'rb'))
                IDs_test = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','IDs_test.pkl'),'rb'))
                IDs_dev = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','IDs_dev.pkl'),'rb'))

                files = [file for file in Path(path_to_results,f'random_seed_{random_seed_test}').iterdir() if 'all_performances' in file.stem and 'test' not in file.stem]
                
                for file in files:
                    model_name = file.stem.split('_')[-1]

                    print(model_name)
                    '''
                    if Path(file.parent,f'best_{n_models}_{model_name}_test.csv').exists():
                        continue
                    '''
                    results = pd.read_excel(file) if file.suffix == '.xlsx' else pd.read_csv(file)
                    results = results.sort_values(by=f'{extremo}_{sort_by}_bootstrap',ascending=ascending).reset_index(drop=True)
                    #selected_features = pd.read_csv(Path(file.parent,file.stem.replace('conf_int','selected_features') + '.csv'))
                    results_test = pd.DataFrame()
                    
                    if 'Fugu_logit__POS_std' in results.columns:
                        print(f'Fufu_logit__POS_std in {task} {dimension} {y_label}')
                        continue

                    for r, row in tqdm.tqdm(results.loc[:n_models,].iterrows(),total=n_models):
                        all_features = [col for col in results.columns if any(x in col for x in ['pitch','talking_intervals','___'])] 

                        results_r = row.dropna().to_dict()
                                        
                        params = dict((key,value) for (key,value) in results_r.items() if all (x not in key for x in ['inf','mean','sup',id_col] + all_features + y_labels))

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
                        
                        if Path(path_to_results,f'random_seed_{random_seed_test}','X_dev.pkl').exists() == False:
                            pickle.dump(X_dev,open(Path(path_to_results,f'random_seed_{random_seed_test}','X_dev.pkl'),'wb'))
                            pickle.dump(X_test,open(Path(path_to_results,f'random_seed_{random_seed_test}','X_test.pkl'),'wb'))
                            pickle.dump(y_dev,open(Path(path_to_results,f'random_seed_{random_seed_test}','y_dev.pkl'),'wb'))
                            pickle.dump(y_test,open(Path(path_to_results,f'random_seed_{random_seed_test}','y_test.pkl'),'wb'))
                        
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

                            result_append[f'inf_{metric}_bootstrap_test'] = inf
                            result_append[f'mean_{metric}_bootstrap_test'] = mean
                            result_append[f'sup_{metric}_bootstrap_test'] = sup
                            
                            result_append[f'inf_{metric}_bootstrap_dev'] = np.round(results_r[f'inf_{metric}_bootstrap'],2)
                            result_append[f'mean_{metric}_bootstrap_dev'] = np.round(results_r[f'mean_{metric}_bootstrap'],2)
                            result_append[f'sup_{metric}_bootstrap_dev'] = np.round(results_r[f'sup_{metric}_bootstrap'],2)

                        if results_test.empty:
                            results_test = pd.DataFrame(columns=result_append.keys())
                        
                        results_test.loc[len(results_test.index),:] = result_append

                    pd.DataFrame(results_test).to_csv(Path(file.parent,f'best_{n_models}_{model_name}_test.csv'),index=False)