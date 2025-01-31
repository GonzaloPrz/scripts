import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings
from joblib import Parallel, delayed
import logging, sys

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier as KNNC
from xgboost import XGBClassifier

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor as KNNR
from xgboost import XGBRegressor as xgboostr

from sklearn.neighbors import KNeighborsRegressor

from sklearn.utils import resample 

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

def test_models_bootstrap(model_class,row,scaler,imputer,X_dev,y_dev,X_test,y_test,all_features,y_labels,metrics_names,IDs_test,boot_train,boot_test,problem_type,threshold,cmatrix=None,priors=None,):
    results_r = row.dropna().to_dict()

    outputs_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test),len(np.unique(y_dev)) if problem_type=='clf' else 1))
    y_true_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test)))
    y_pred_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test)))
    IDs_test_bootstrap = np.empty((np.max((1,boot_train)),np.max((1,boot_test)),len(y_test)), dtype=object)
    metrics_test_bootstrap = {metric: np.empty((np.max((1,boot_train)),np.max((1,boot_test)))) for metric in metrics_names}

    if not isinstance(X_dev,pd.DataFrame):
        X_dev = pd.DataFrame(X_dev.squeeze(),columns=all_features)

    if not isinstance(X_test,pd.DataFrame):
        X_test = pd.DataFrame(X_test.squeeze(),columns=all_features)

    params = dict((key,value) for (key,value) in results_r.items() if not isinstance(value,dict) and all(x not in key for x in ['inf','sup','mean'] + all_features + y_labels + ['id','Unnamed: 0','threshold','idx']))

    features = [col for col in all_features if col in results_r.keys() and results_r[col] == 1]
    features_dict = {col:results_r[col] for col in all_features if col in results_r.keys()}

    if 'gamma' in params.keys():
        try: 
            params['gamma'] = float(params['gamma'])
        except:
            pass
    if 'random_state' in params.keys():
        params['random_state'] = int(params['random_state'])
    
    for b_train in range(np.max((1,boot_train))):
        boot_index_train = resample(X_dev.index, n_samples=X_dev.shape[0], replace=True, random_state=b_train) if boot_train > 0 else X_dev.index

        for b_test in range(np.max((1,boot_test))):
            boot_index = resample(X_test.index, n_samples=X_test.shape[0], replace=True, random_state=b_train * np.max((1,boot_train)) + b_test) if boot_test > 0 else X_test.index

            outputs = test_model(model_class,params,scaler,imputer, X_dev.loc[boot_index_train,:], y_dev[boot_index_train], X_test.loc[boot_index,:], y_test[boot_index], metrics_names, IDs_test.squeeze()[boot_index], cmatrix, priors, problem_type=problem_type,threshold=threshold)

            outputs_bootstrap[b_train,b_test,:] = outputs

            if problem_type == 'clf':
                metrics_test, y_pred = get_metrics_clf(outputs, y_test[boot_index], metrics_names, cmatrix, priors,threshold)
                y_pred_bootstrap[b_train,b_test,:] = y_pred
            else:
                metrics_test = get_metrics_reg(outputs, y_test[boot_index], metrics_names)
                y_pred_bootstrap[b_train,b_test,:] = outputs
                
            y_true_bootstrap[b_train,b_test,:] = y_test[boot_index]
            IDs_test_bootstrap[b_train,b_test,:] = IDs_test.squeeze()[boot_index]

            for metric in metrics_names:
                metrics_test_bootstrap[metric][b_train,b_test] = metrics_test[metric]

    result_append = params.copy()
    result_append.update(features_dict)

    for metric in metrics_names:
        mean, inf, sup = conf_int_95(metrics_test_bootstrap[metric].flatten())

        result_append[f'inf_{metric}_test'] = np.round(inf,5)
        result_append[f'mean_{metric}_test'] = np.round(mean,5)
        result_append[f'sup_{metric}_test'] = np.round(sup,5)

        result_append[f'inf_{metric}_dev'] = np.round(results_r[f'{metric}_inf'],5)
        result_append[f'mean_{metric}_dev'] = np.round(results_r[f'{metric}_mean'],5)
        result_append[f'sup_{metric}_dev'] = np.round(results_r[f'{metric}_sup'],5)
    
    return result_append,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_test_bootstrap

from utils import *

from expected_cost.ec import *
from psrcal import *

##---------------------------------PARAMETERS---------------------------------##
project_name = 'ad_mci_hc'
stat_folder = 'mean_std'

scaler_name = 'StandardScaler'
boot_test = 200
hyp_opt = True
filter_outliers = False
shuffle_labels = False
feature_selection = False
n_folds = 5
parallel = True

# Check if required arguments are provided
if len(sys.argv) > 1:
    #print("Usage: python test_models.py <project_name> [hyp_opt] [filter_outliers] [shuffle_labels] [feature_selection] [k]")
    project_name = sys.argv[1]
if len(sys.argv) > 2:
    hyp_opt = bool(int(sys.argv[2]))
if len(sys.argv) > 3:
    filter_outliers = bool(int(sys.argv[3]))
if len(sys.argv) > 4:
    shuffle_labels = bool(int(sys.argv[4]))
if len(sys.argv) > 5:
    feature_selection = bool(int(sys.argv[5]))
if len(sys.argv) > 6:
    n_folds = int(sys.argv[6])

y_labels = {'tell_classifier':['target'],
            'MCI_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GeroApathy': ['DASS_21_Depression_label','AES_Total_Score_label','Depression_Total_Score_label','MiniSea_MiniSea_Total_EkmanFaces_label','MiniSea_minisea_total_label'],
            'GeroApathy_reg': ['DASS_21_Depression','AES_Total_Score','Depression_Total_Score','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total'],
            'GERO_Ivo': ['GM_norm','WM_norm','norm_vol_bilateral_HIP','norm_vol_mask_AD', 
                         'GM','WM','vol_bilateral_HIP','vol_mask_AD',
                         'MMSE_Total_Score','ACEIII_Total_Score','IFS_Total_Score','MoCA_Total_Boni_3'
                        ],
            'ad_mci_hc': ['group']           
            }

metrics_names = {'tell_classifier': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'MCI_classifier': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'ad_mci_hc':['accuracy','norm_expected_cost','norm_cross_entropy'],
                 'Proyecto_Ivo': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'GeroApathy': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'GeroApthy_reg':['r2_score','mean_absolute_error','mean_squared_error'],
                 'GERO_Ivo':['r2_score','mean_absolute_error','mean_squared_error']
}

thresholds = {'tell_classifier':[np.log(0.5)],
                'MCI_classifier':[np.log(0.5)],
                'Proyecto_Ivo':[np.log(0.5)],
                'GeroApathy':[np.log(0.5)],
                'GeroApathy_reg':[None],
                'GERO_Ivo':[None],
                'ad_mci_hc':[None]
                }

boot_train = 0

n_seeds_test = 1

##---------------------------------PARAMETERS---------------------------------##

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['Animales','P','Animales__P','cog','brain','AAL','conn'],
         'GeroApathy':['agradable'],
         'GeroApathy_reg':['agradable'],
         'GERO_Ivo':['fas','animales','fas__animales','grandmean'],
         'ad_mci_hc':['fugu'],
         'AKU':['picture_description',
                'pleasant_memory',
                 'routine',
                 'video_retelling'
                ],
            'AKU_outliers_as_nan':['picture_description',
                'pleasant_memory',
                 'routine',
                 'video_retelling']}

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GeroApathy':'clf',
                'GeroApathy_reg':'reg',
                'GERO_Ivo':'reg',
                'ad_mci_hc':'clf'}

scoring_metrics = {'MCI_classifier':['norm_cross_entropy'],
           'tell_classifier':['norm_cross_entropy'],
           'Proyecto_Ivo':['roc_auc'],
           'GeroApathy':['norm_cross_entropy','roc_auc'],
           'GeroApathy_reg':['r2_score','mean_absolute_error'], 
           'GERO_Ivo':['r2_score','mean_absolute_error'],
           'ad_mci_hc':['norm_cross_entropy']}

if n_folds == 0:
    kfold_folder = 'l2ocv'
elif n_folds == -1:
    kfold_folder = 'loocv'
else:
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

models_dict = {'clf':{'lr': LogisticRegression,
                    'svc': SVC, 
                    'xgb': XGBClassifier,
                    'knnc': KNNC},
                
                'reg':{'lasso':Lasso,
                        'ridge':Ridge,
                        'elastic':ElasticNet,
                        #'knn':KNNR,
                        'svr':SVR,
                        'xgb':xgboostr
                    }
}

for task,scoring in itertools.product(tasks[project_name],scoring_metrics[project_name]):
    extremo = 'sup' if any(x in scoring for x in ['norm','error']) else 'inf'
    ascending = True if extremo == 'sup' else False

    dimensions = [folder.name for folder in Path(save_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(task,dimension)
        for y_label in y_labels[project_name]:
            print(y_label)
            path_to_results = Path(save_dir,task,dimension,scaler_name,kfold_folder, y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt', 'feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type[project_name] == 'reg' else '','shuffle' if shuffle_labels else '')

            if not path_to_results.exists():
                continue
            
            random_seeds_test = [folder.name for folder in path_to_results.iterdir() if folder.is_dir()]

            if len(random_seeds_test) == 0:
                random_seeds_test = ['']
                
            for random_seed_test in random_seeds_test:

                files = [file for file in Path(path_to_results,random_seed_test).iterdir() if 'all_models_' in file.stem and 'dev_bca' in file.stem]
                filename_to_save = 'all_models'
                if len(files) == 0:
                    files = [file for file in Path(path_to_results,random_seed_test).iterdir() if 'best_models_' in file.stem and 'dev' in file.stem and scoring in file.stem]
                    filename_to_save = f'best_models_{scoring[project_name]}'
                if len(files) == 0:
                    continue

                X_dev = pickle.load(open(Path(path_to_results,random_seed_test,'X_dev.pkl'),'rb'))
                y_dev = pickle.load(open(Path(path_to_results,random_seed_test,'y_true_dev.pkl'),'rb'))
                IDs_dev = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_dev.pkl'),'rb'))
                dev = pd.DataFrame({'y_dev':y_dev.flatten(), 'ID':IDs_dev.flatten()})
                dev = dev.drop_duplicates(subset=['ID'])
                X_test = pickle.load(open(Path(path_to_results,random_seed_test,'X_test.pkl'),'rb'))   
                y_test = pickle.load(open(Path(path_to_results,random_seed_test,'y_test.pkl'),'rb'))
                IDs_test = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_test.pkl'),'rb'))
                
                for file in files:
                    model_name = file.stem.split('_')[2]

                    print(model_name)
                    
                    #if Path(file.parent,f'{filename_to_save}_{model_name}_test.csv').exists():
                    #    continue
                    
                    results_dev = pd.read_excel(file) if file.suffix == '.xlsx' else pd.read_csv(file)
                    
                    if f'{extremo}_{scoring}' in results_dev.columns:
                        scoring_col = f'{extremo}_{scoring}'
                    elif f'{extremo}_{scoring}_dev' in results_dev.columns:
                        scoring_col = f'{extremo}_{scoring}_dev'
                    else:
                        scoring_col = f'{scoring}_{extremo}'

                    results_dev = results_dev.sort_values(by=scoring_col,ascending=ascending)
                    
                    all_features = [col for col in results_dev.columns if any([dim in col for dim in dimension.split('__')])]
                    if 'threshold' not in results_dev.columns:
                        results_dev['threshold'] = thresholds[project_name][0]

                    if len(all_features) == 0:
                        continue

                    results = Parallel(n_jobs=-1 if parallel else 1)(delayed(test_models_bootstrap)(models_dict[problem_type[project_name]][model_name],results_dev.loc[r,:],scaler,imputer,X_dev,dev.y_dev,
                                                                                X_test,y_test,all_features,y_labels[project_name],metrics_names[project_name],IDs_test,boot_train,
                                                                                boot_test,problem_type[project_name],threshold=results_dev.loc[r,'threshold']) 
                                                                                for r in results_dev.index)
                    
                    results_test = pd.concat([pd.DataFrame(result[0],index=[0]) for result in results])
                    results_test['idx'] = results_dev['Unnamed: 0'].values

                    outputs_bootstrap = np.stack([result[1] for result in results],axis=0)
                    y_true_bootstrap = np.stack([result[2] for result in results],axis=0)
                    y_pred_bootstrap = np.stack([result[3] for result in results],axis=0)
                    IDs_test_bootstrap = np.stack([result[4] for result in results],axis=0)

                    results_test.to_csv(Path(file.parent,f'{filename_to_save}_{model_name}_test.csv'))

                    if not Path(file.parent,f'all_models_{model_name}_test.csv').exists():
                        with open(Path(file.parent,'y_test_bootstrap.pkl'),'wb') as f:
                            pickle.dump(y_true_bootstrap,f)
                        with open(Path(file.parent,f'IDs_test_bootstrap.pkl'),'wb') as f:   
                            pickle.dump(IDs_test_bootstrap,f)
            
                    with open(Path(file.parent,f'y_pred_bootstrap_{model_name}_{scoring}.pkl'),'wb') as f:
                        pickle.dump(y_pred_bootstrap,f)
