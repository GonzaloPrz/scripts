import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier 

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

from expected_cost.ec import *
from psrcal import *

project_name = 'MCI_classifier'

tasks = ['fas','animales','fas__animales','grandmean']
y_labels = ['target']

scaler_name = 'StandardScaler'

l2ocv = False

n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

hyp_opt_list = [True]
feature_selection_list = [True,False]
bootstrap_list = [True]

boot_test = 100
boot_train = 0

n_seeds_test = 1

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:/','CNC_Audio','gonza','data',project_name)
save_dir = Path(str(data_dir).replace('data','results'))    

if scaler_name == 'StandardScaler':
    scaler = StandardScaler
elif scaler_name == 'MinMaxScaler':
    scaler = MinMaxScaler
else:
    scaler = None
imputer = KNNImputer

models_dict = {'lr': LogisticRegression,
               'svc': SVC, 
               'xgb': XGBClassifier,
               'knn': KNeighborsClassifier,
               }

metrics_names = ['roc_auc','accuracy','f1','recall','norm_expected_cost','norm_cross_entropy']

random_seeds_test = np.arange(n_seeds_test)

n_models = 10

scoring = 'roc_auc'
extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if 'norm' in scoring else False

for task in tasks:
    dimensions = [folder.name for folder in Path(save_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(task,dimension)
        for y_label,hyp_opt,feature_selection,bootstrap in itertools.product(y_labels,hyp_opt_list,feature_selection_list,bootstrap_list):
            path_to_results = save_dir / task / dimension / scaler_name / kfold_folder / f'{n_seeds_train}_seeds_train' / f'{n_seeds_test}_seeds_test' / y_label / 'no_hyp_opt' / 'feature_selection' / 'bootstrap'
            
            path_to_results = Path(str(path_to_results).replace('no_hyp_opt', 'hyp_opt')) if hyp_opt else path_to_results
            path_to_results = Path(str(path_to_results).replace('feature_selection', '')) if not feature_selection else path_to_results
            path_to_results = Path(str(path_to_results).replace('bootstrap', '')) if not bootstrap else path_to_results

            for random_seed_test in random_seeds_test:
                files = [file for file in Path(path_to_results,f'random_seed_{random_seed_test}').iterdir() if 'all_performances' in file.stem and 'test' not in file.stem]

                X_dev = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','X_dev.pkl'),'rb'))

                try:
                    y_dev = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','y_dev.pkl'),'rb'))
                except:
                    y_dev = pickle.load(open(Path(str(path_to_results).replace('feature_selection',''),f'random_seed_{random_seed_test}','y_dev.pkl'),'rb'))
                
                X_test = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','X_test.pkl'),'rb'))
                try:
                    y_test = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','y_test.pkl'),'rb'))
                except:
                    y_test = pickle.load(open(Path(str(path_to_results).replace('feature_selection',''),f'random_seed_{random_seed_test}','y_test.pkl'),'rb'))

                try:
                    IDs_test = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','IDs_test.pkl'),'rb'))
                except:
                    IDs_test = pickle.load(open(Path(str(path_to_results).replace('feature_selection',''),f'random_seed_{random_seed_test}','IDs_test.pkl'),'rb'))
                    
                all_features = X_dev.columns

                for file in files:
                    model_name = file.stem.split('_')[-1]

                    print(model_name)
                    
                    #if Path(file.parent,f'best_{n_models}_{model_name}_test.csv').exists():
                    #    continue
                    
                    results = pd.read_excel(file) if file.suffix == '.xlsx' else pd.read_csv(file)
                    results = results.sort_values(by=[f'{extremo}_{scoring}_bootstrap'],ascending=ascending).reset_index(drop=True)
                    results_test = pd.DataFrame()
                    
                    for r, row in tqdm.tqdm(results.iloc[:n_models,].iterrows()):
                        results_r = row.dropna().to_dict()
                                        
                        params = dict((key,value) for (key,value) in results_r.items() if 'inf' not in key and 'sup' not in key and 'mean' not in key and 'std' not in key and all(x not in key for x in all_features))

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
                    
                    with open(Path(file.parent,'y_test_bootstrap.pkl'),'wb') as f:
                        pickle.dump(y_test,f)
                    with open(Path(file.parent,f'y_pred_bootstrap_{model_name}.pkl'),'wb') as f:
                        pickle.dump(y_pred_bootstrap,f)
                    
                    with open(Path(file.parent,f'IDs_test_bootstrap.pkl'),'wb') as f:
                        pickle.dump(IDs_test_bootstrap,f)

                    scores_df = {'ID':IDs_test_bootstrap.flatten(),'y_true':y_true_bootstrap.flatten(),'output':outputs_bootstrap[:,:,1].flatten(),'y_pred':y_pred_bootstrap.flatten()}
                    scores_df = pd.DataFrame(scores_df).sort_values(by='ID',ascending=True)
                    scores_df.drop_duplicates(subset='ID',keep='first',inplace=True)
                    scores_df.to_csv(Path(file.parent,f'scores_test_{model_name}.csv'))