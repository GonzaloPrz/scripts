import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier 

sys.path.append(str(Path(Path.home(),'scripts_generales')))

from utils import *

from expected_cost.ec import *
from psrcal import *

l2ocv = False

n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

y_labels = ['Grupo']

hyp_opt_list = [True]
feature_selection_list = [True]
bootstrap_list = [True]

boot_test = 100
boot_train = 0

n_seeds_test = 1

base_dir = Path(Path(__file__).parent)

tasks = ['Animales','P']

scaler_name = 'StandardScaler'

scaler = StandardScaler if scaler_name == 'StandardScaler' else MinMaxScaler

models_dict = {'lr': LogisticRegression,
               'svc': SVC, 
               'xgb': XGBClassifier,
               'knn': KNeighborsClassifier,
               }

metrics_names = ['roc_auc','accuracy','f1','recall','norm_cross_entropy']

random_seeds_test = np.arange(n_seeds_test)

n_models = 100

for task in tasks:
    dimensions = [folder.name for folder in Path(Path(__file__).parent,task).iterdir() if folder.is_dir()]
    #dimensions = ['talking-intervals','voice-quality','voice-quality_talking-intervals']
    for dimension in dimensions:
        print(task,dimension)
        for y_label,hyp_opt,feature_selection,bootstrap in itertools.product(y_labels,hyp_opt_list,feature_selection_list,bootstrap_list):
            path_to_results = base_dir / task / dimension / scaler_name / 'all_features' /kfold_folder / f'{n_seeds_train}_seeds_train' / f'{n_seeds_test}_seeds_test' / y_label / 'no_hyp_opt' / 'feature_selection' / 'bootstrap'
            
            path_to_results = Path(str(path_to_results).replace('no_hyp_opt', 'hyp_opt')) if hyp_opt else path_to_results
            path_to_results = Path(str(path_to_results).replace('feature_selection', '')) if not feature_selection else path_to_results
            path_to_results = Path(str(path_to_results).replace('bootstrap', '')) if not bootstrap else path_to_results

            for random_seed_test in random_seeds_test:
                files = [file for file in Path(path_to_results,f'random_seed_{random_seed_test}').iterdir() if 'all_performances' in file.stem and 'test' not in file.stem]

                X_dev = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','X_train.pkl'),'rb'))
                y_dev = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','y_train.pkl'),'rb'))

                X_test = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','X_test.pkl'),'rb'))
                y_test = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','y_test.pkl'),'rb'))

                IDs_test = pickle.load(open(Path(path_to_results,f'random_seed_{random_seed_test}','IDs_test.pkl'),'rb'))

                all_features = X_dev.columns

                for file in files:
                    model_name = file.stem.split('_')[-1]

                    print(model_name)
                    
                    if Path(file.parent,f'best_{n_models}_{model_name}_test.csv').exists():
                        continue
                    
                    results = pd.read_excel(file) if file.suffix == '.xlsx' else pd.read_csv(file)

                    #selected_features = pd.read_csv(Path(file.parent,file.stem.replace('conf_int','selected_features') + '.csv'))
                    results_test = pd.DataFrame()
                    
                    for r, row in tqdm.tqdm(results.iloc[:n_models,].iterrows(),total=n_models):
                        results_r = row.dropna().to_dict()
                                        
                        params = dict((key,value) for (key,value) in results_r.items() if 'inf' not in key and 'sup' not in key and 'mean' not in key and 'std' not in key and all(x not in key for x in dimension.split('_')))

                        features = [col for col in all_features if results_r[col] == 1]
                        features_dict = {col:results_r[col] for col in all_features}

                        if 'gamma' in params.keys():
                            params['gamma'] = float(params['gamma'])
                        
                        mod = Model(models_dict[model_name](**params),scaler())
                        metrics_test_bootstrap,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_test_bootstrap = test_model(mod,X_dev[features],y_dev,X_test[features],y_test,metrics_names,IDs_test,boot_train,boot_test)

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
                    