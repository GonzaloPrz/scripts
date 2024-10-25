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
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier 

from sklearn.linear_model import Ridge as RR
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor as KNN

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

def test_models_bootstrap(model_class,row,scaler,imputer,X_dev,y_dev,X_test,y_test,all_features,y_labels,metrics_names,IDs_test,boot_train,boot_test,problem_type,threshold):
    results_r = row.dropna().to_dict()
                                        
    params = dict((key,value) for (key,value) in results_r.items() if not isinstance(value,dict) and all(x not in key for x in ['inf','sup','mean'] + all_features + y_labels + ['id','Unnamed: 0','threshold']))

    features = [col for col in all_features if col in results_r.keys() and results_r[col] == 1]
    features_dict = {col:results_r[col] for col in all_features if col in results_r.keys()}

    if 'gamma' in params.keys():
        try: 
            params['gamma'] = float(params['gamma'])
        except:
            pass
    if 'random_state' in params.keys():
        params['random_state'] = int(params['random_state'])
    
    metrics_test_bootstrap,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_test_bootstrap = test_model(model_class,params,scaler,imputer,X_dev[features],y_dev,X_test[features],y_test,metrics_names,IDs_test,boot_train,boot_test,cmatrix=None,priors=None,problem_type=problem_type,threshold=threshold)

    result_append = params.copy()
    result_append.update(features_dict)

    for metric in metrics_names:
        mean, inf, sup = conf_int_95(metrics_test_bootstrap[metric])

        result_append[f'inf_{metric}_test'] = np.round(inf,2)
        result_append[f'mean_{metric}_test'] = np.round(mean,2)
        result_append[f'sup_{metric}_test'] = np.round(sup,2)

        result_append[f'inf_{metric}_dev'] = np.round(results_r[f'inf_{metric}'],2)
        result_append[f'mean_{metric}_dev'] = np.round(results_r[f'mean_{metric}'],2)
        result_append[f'sup_{metric}_dev'] = np.round(results_r[f'sup_{metric}'],2)
    
    return result_append,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_test_bootstrap

from utils import *

from expected_cost.ec import *
from psrcal import *

##---------------------------------PARAMETERS---------------------------------##

project_name = 'GeroApathy'

scoring = 'r2_score'

l2ocv = False

hyp_opt_list = [True]
feature_selection_list = [True]
shuffle_labels = False

y_labels = {'tell_classifier':['target'],
            'MCI_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GeroApathy': ['DASS_21_Depression','DASS_21_Anxiety','DASS_21_Stress','AES_Total_Score','MiniSea_MiniSea_Total_FauxPas','Depression_Total_Score','MiniSea_emf_total','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total']}

metrics_names = {'tell_classifier': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'MCI_classifier': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'Proyecto_Ivo': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'GeroApathy': ['r2_score','mean_absolute_error','mean_squared_error']}

thresholds = {'tell_classifier':[0.5],
                'MCI_classifier':[0.5],
                'Proyecto_Ivo':[0.5],
                'GeroApathy':[None]}

scaler_name = 'StandardScaler'

boot_test = 10
boot_train = 0

n_seeds_test = 1

##---------------------------------PARAMETERS---------------------------------##

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['Animales','P','Animales__P','cog','brain','AAL','conn'],
         'GeroApathy':['Fugu']}

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GeroApathy':'reg'}

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)
save_dir = Path(str(data_dir).replace('data','results'))    

if scaler_name == 'StandardScaler':
    scaler = StandardScaler
elif scaler_name == 'MinMaxScaler':
    scaler = MinMaxScaler
else:
    scaler = None
imputer = KNNImputer

models_dict = {'tell)_classifier':{'lr': LogisticRegression,
                                    'svc': SVC, 
                                    'xgb': XGBClassifier,
                                    'knn': KNeighborsClassifier},
                'MCI_classifier':{'lr': LogisticRegression,
                                    'svc': SVC, 
                                    'xgb': XGBClassifier,
                                    'knn': KNeighborsClassifier},
                'MCI_classifier':{'lr': LogisticRegression,
                                    'svc': SVC, 
                                    'xgb': XGBClassifier,
                                    'knn': KNeighborsClassifier},
                'GeroApathy':{'ridge':RR,
                            'knn':KNN,
                            'lasso':Lasso}}

extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if 'norm' in scoring else False

for task in tasks[project_name]:
    dimensions = [folder.name for folder in Path(save_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(task,dimension)
        for y_label,hyp_opt,feature_selection in itertools.product(y_labels[project_name],hyp_opt_list,feature_selection_list):
            path_to_results = Path(save_dir,task,dimension,scaler_name,kfold_folder,'mean_std' if project_name == 'GeroApathy' else '', y_label, 'no_hyp_opt', 'feature_selection')
            
            path_to_results = Path(str(path_to_results).replace('no_hyp_opt', 'hyp_opt')) if hyp_opt else path_to_results
            path_to_results = Path(str(path_to_results).replace('feature_selection', '')) if not feature_selection else path_to_results

            if not path_to_results.exists():
                continue
            
            random_seeds_test = [folder.name for folder in path_to_results.iterdir() if folder.is_dir()]

            if len(random_seeds_test) == 0:
                random_seeds_test = ['']
                
            for random_seed_test in random_seeds_test:
                files = [file for file in Path(path_to_results,random_seed_test).iterdir() if 'all_models_' in file.stem and 'dev' in file.stem]
                
                if len(files) == 0:
                    continue

                X_dev = pickle.load(open(Path(path_to_results,random_seed_test,'X_dev.pkl'),'rb'))

                y_dev = pickle.load(open(Path(path_to_results,random_seed_test,'y_dev.pkl'),'rb'))
                
                IDs_dev = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_dev.pkl'),'rb'))

                X_test = pickle.load(open(Path(path_to_results,random_seed_test,'X_test.pkl'),'rb'))
                
                y_test = pickle.load(open(Path(path_to_results,random_seed_test,'y_test.pkl'),'rb'))
            
                IDs_test = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_test.pkl'),'rb'))

                all_features = [col for col in X_dev.columns if any(f'{x}_{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__')))]
                
                for file in files:
                    model_name = file.stem.split('_')[-2]

                    print(model_name)
                    
                    if Path(file.parent,f'all_models_{model_name}_test.csv').exists():
                        continue
                    
                    results_dev = pd.read_excel(file) if file.suffix == '.xlsx' else pd.read_csv(file)
                    results_dev = results_dev.sort_values(by=f'{extremo}_{scoring}',ascending=ascending)
                    
                    if 'threshold' not in results_dev.columns:
                        results_dev['threshold'] = thresholds[project_name][0]

                    results = Parallel(n_jobs=-1)(delayed(test_models_bootstrap)(models_dict[project_name][model_name],results_dev.loc[r,:],scaler,imputer,X_dev,y_dev,
                                                                                 X_test,y_test,all_features,y_labels[project_name],metrics_names[project_name],IDs_test,boot_train,
                                                                                 boot_test,problem_type[project_name],threshold=results_dev.loc[r,'threshold']) 
                                                                                 for r in results_dev.index)
                    
                    results_test = pd.concat([pd.DataFrame(result[0],index=[0]) for result in results])
                    results_test['index'] = results_dev.index

                    outputs_bootstrap = np.stack([result[1] for result in results],axis=0)
                    y_true_bootstrap = np.stack([result[2] for result in results],axis=0)
                    y_pred_bootstrap = np.stack([result[3] for result in results],axis=0)
                    IDs_test_bootstrap = np.stack([result[4] for result in results],axis=0)

                    results_test.to_csv(Path(file.parent,f'best_models_{scoring}_{model_name}_test.csv'))
                    
                    with open(Path(file.parent,'y_test_bootstrap.pkl'),'wb') as f:
                        pickle.dump(y_true_bootstrap,f)
                    with open(Path(file.parent,f'y_pred_bootstrap_{model_name}.pkl'),'wb') as f:
                        pickle.dump(y_pred_bootstrap,f)
                    
                    with open(Path(file.parent,f'IDs_test_bootstrap.pkl'),'wb') as f:
                        pickle.dump(IDs_test_bootstrap,f)