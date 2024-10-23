import seaborn as sns

import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings

warnings.filterwarnings('ignore')

l2ocv = False

test_size = 0

project_name = 'Proyecto_Ivo'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:/','CNC_Audio','gonza','results',project_name)

scaler_name = 'StandardScaler'

y_labels = ['target']

id_col = 'id'

hyp_tuning_list = [True]
feature_selection_list = [True]

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

scoring = 'norm_cross_entropy'
extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if extremo == 'sup' else False

best_classifiers = pd.read_csv(Path(results_dir,f'best_classifiers_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection.csv'))

tasks = best_classifiers['task'].unique()
dimensions = best_classifiers.dimension.unique()

for task,y_label in itertools.product(tasks,y_labels):
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(task,dimension)
        for y_label,hyp_opt,feature_selection in itertools.product(y_labels,hyp_tuning_list,feature_selection_list):
            path_to_results = results_dir / task / dimension / scaler_name / kfold_folder / y_label / 'no_hyp_opt' / 'feature_selection'
            
            path_to_results = Path(str(path_to_results).replace('no_hyp_opt', 'hyp_opt')) if hyp_opt else path_to_results
            path_to_results = Path(str(path_to_results).replace('feature_selection', '')) if not feature_selection else path_to_results

            random_seeds_test = [folder.name for folder in path_to_results.iterdir() if folder.is_dir()]
            if len(random_seeds_test) == 0:
                continue

            for random_seed_test in random_seeds_test:
                files = [file for file in Path(path_to_results,random_seed_test).iterdir() if 'all_models_' in file.stem and 'dev' in file.stem]

                X_dev = pickle.load(open(Path(path_to_results,random_seed_test,'X_dev.pkl'),'rb'))
            
                y_dev = pickle.load(open(Path(path_to_results,random_seed_test,'y_dev.pkl'),'rb'))
                
                try:
                    X_test = pickle.load(open(Path(path_to_results,random_seed_test,'X_test.pkl'),'rb'))
                
                    y_test = pickle.load(open(Path(path_to_results,random_seed_test,'y_test.pkl'),'rb'))
                
                    IDs_test = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_test.pkl'),'rb'))
                
                except:
                    pass

                all_features = X_dev.columns

                for file in files:
                    model_name = file.stem.split('_')[-1]

                    print(model_name)
                    
                    #if Path(file.parent,f'best_{n_models}_{model_name}_test.csv').exists():
                    #    continue
                    
                    results = pd.read_excel(file) if file.suffix == '.xlsx' else pd.read_csv(file)
                    results = results.sort_values(by=[f'{extremo}_{scoring}_bootstrap'],ascending=ascending).head(1)

                    results_test = pd.DataFrame()
                    
                    for r, row in tqdm.tqdm(results.iterrows()):
                        results_r = row.dropna().to_dict()
                                        
                        params = dict((key,value) for (key,value) in results_r.items() if all (x not in key for key in ['inf','mean','sup','threshold'] + all_features))

                        features = [col for col in all_features if results_r[col] == 1]
                        features_dict = {col:results_r[col] for col in all_features}

                        if 'gamma' in params.keys():
                            try: 
                                params['gamma'] = float(params['gamma'])
                            except:
                                pass

                        mod = Model(models_dict[model_name](**params),scaler,imputer)
                        metrics_test_bootstrap,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_test_bootstrap = test_model(mod,X_dev[features],y_dev,X_test[features],y_test,metrics_names,IDs_test,boot_train,boot_test)

                        result_append = params.copy()
                        result_append.update(features_dict)
                        
                        for metric in metrics_names:
                            mean, inf, sup = conf_int_95(metrics_test_bootstrap[metric])
                            
                            result_append[f'inf_{metric}_bootstrap_test'] = np.round(inf,2)
                            result_append[f'mean_{metric}_bootstrap_test'] = np.round(mean,2)
                            result_append[f'sup_{metric}_bootstrap_test'] = np.round(sup,2)
                            
                            try: 
                                result_append[f'inf_{metric}_bootstrap_dev'] = np.round(results_r[f'inf_{metric}_bootstrap'],2)
                                result_append[f'mean_{metric}_bootstrap_dev'] = np.round(results_r[f'mean_{metric}_bootstrap'],2)
                                result_append[f'sup_{metric}_bootstrap_dev'] = np.round(results_r[f'sup_{metric}_bootstrap'],2)
                            except:
                                pass
                        if results_test.empty:
                            results_test = pd.DataFrame(columns=result_append.keys())
                        
                        results_test.loc[len(results_test.index),:] = result_append

                    pd.DataFrame(results_test).to_csv(Path(file.parent,f'best_{n_models}_{model_name}_test.csv'),index=False)
                    