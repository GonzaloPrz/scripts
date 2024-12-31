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

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

def test_models_bootstrap(model_class,row,scaler,imputer,X_dev,y_dev,X_test,y_test,all_features,y_labels,metrics_names,IDs_test,boot_train,boot_test,problem_type,threshold):
    results_r = row.dropna().to_dict()
                                        
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
    
    metrics_test_bootstrap,outputs_bootstrap,y_true_bootstrap,y_pred_bootstrap,IDs_test_bootstrap = test_model(model_class,params,scaler,imputer,X_dev[features],y_dev,X_test[features],y_test,metrics_names,IDs_test,boot_train,boot_test,cmatrix=None,priors=None,problem_type=problem_type,threshold=threshold)

    result_append = params.copy()
    result_append.update(features_dict)

    for metric in metrics_names:
        mean, inf, sup = conf_int_95(metrics_test_bootstrap[metric])

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
project_name = 'GERO_Ivo'

scaler_name = 'StandardScaler'
boot_test = 200
hyp_opt = True
filter_outliers = True
shuffle_labels = False
feature_selection = True
n_folds = 5

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
    xn_folds = int(sys.argv[6])

l2ocv = False

y_labels = {'tell_classifier':['target'],
            'MCI_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GeroApathy': ['DASS_21_Depression_label','AES_Total_Score_label','Depression_Total_Score_label','MiniSea_MiniSea_Total_EkmanFaces_label','MiniSea_minisea_total_label'],
            'GeroApathy_reg': ['DASS_21_Depression','AES_Total_Score','Depression_Total_Score','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total'],
            'GERO_Ivo': [#'GM_norm','WM_norm','norm_vol_bilateral_HIP','norm_vol_mask_AD', 
                         'MMSE_Total_Score','ACEIII_Total_Score','IFS_Total_Score','MoCA_Total_Boni_3'
                        ]
            }

metrics_names = {'clf': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                'reg':['r2_score','mean_absolute_error','mean_squared_error']}

thresholds = {'tell_classifier':[0.5],
                'MCI_classifier':[0.5],
                'Proyecto_Ivo':[0.5],
                'GeroApathy':[0.5],
                'GeroApathy_reg':[None],
                'GERO_Ivo':[None]}

boot_train = 0

n_seeds_test = 1

##---------------------------------PARAMETERS---------------------------------##

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['Animales','P','Animales__P','cog','brain','AAL','conn'],
         'GeroApathy':['agradable'],
         'GeroApathy_reg':['agradable'],
         'GERO_Ivo':['fas','animales','fas__animales','grandmean']}

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GeroApathy':'clf',
                'GeroApathy_reg':'reg',
                'GERO_Ivo':'reg'}

scoring_metrics = {'MCI_classifier':['norm_cross_entropy'],
           'tell_classifier':['norm_cross_entropy'],
           'Proyecto_Ivo':['roc_auc'],
           'GeroApathy':['norm_cross_entropy','roc_auc'],
           'GeroApathy_reg':['r2_score','mean_absolute_error'], 
           'GERO_Ivo':['r2_score','mean_absolute_error']}

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)
save_dir = Path(str(data_dir).replace('data','results'))    

log_file = Path(save_dir,Path(__file__).stem + '.log')

logging.basicConfig(
    level=logging.DEBUG,  # Log all messages (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Log to a file
        logging.StreamHandler(sys.stdout)  # Keep output in the terminal as well
    ]
)

# Redirect stdout and stderr to the logger
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():  # Avoid logging blank lines
            self.level(message)

    def flush(self):  # Required for file-like behavior
        pass

sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)

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
                    'knn': KNNC},
                
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
            path_to_results = Path(save_dir,task,dimension,scaler_name,kfold_folder, y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt', 'feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type[project_name] == 'reg' else '','shuffle' if shuffle_labels else '')

            if not path_to_results.exists():
                continue
            
            random_seeds_test = [folder.name for folder in path_to_results.iterdir() if folder.is_dir()]

            if len(random_seeds_test) == 0:
                random_seeds_test = ['']
                
            for random_seed_test in random_seeds_test:
                files = [file for file in Path(path_to_results,random_seed_test).iterdir() if 'best_models_' in file.stem and 'dev' in file.stem and scoring in file.stem]
                
                if len(files) == 0:
                    continue
                try:
                    X_dev = pickle.load(open(Path(path_to_results,random_seed_test,'X_dev.pkl'),'rb'))

                    y_dev = pickle.load(open(Path(path_to_results,random_seed_test,'y_dev.pkl'),'rb'))
                    
                    IDs_dev = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_dev.pkl'),'rb'))

                    X_test = pickle.load(open(Path(path_to_results,random_seed_test,'X_test.pkl'),'rb'))
                    
                    y_test = pickle.load(open(Path(path_to_results,random_seed_test,'y_test.pkl'),'rb'))
                
                    IDs_test = pickle.load(open(Path(path_to_results,random_seed_test,'IDs_test.pkl'),'rb'))

                    all_features = [col for col in X_dev.columns if any(f'{x}_{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__')))]
                    
                    for file in files:
                        model_name = file.stem.split('_')[2]

                        print(model_name)
                        
                        if Path(file.parent,f'all_models_{model_name}_test.csv').exists():
                            continue
                        
                        results_dev = pd.read_excel(file) if file.suffix == '.xlsx' else pd.read_csv(file)
                        
                        if f'{extremo}_{scoring}' in results_dev.columns:
                            scoring_col = f'{extremo}_{scoring}'
                        elif f'{extremo}_{scoring}_dev' in results_dev.columns:
                            scoring_col = f'{extremo}_{scoring}_dev'
                        else:
                            scoring_col = f'{scoring}_{extremo}'

                        results_dev = results_dev.sort_values(by=scoring_col,ascending=ascending)
                        
                        if 'threshold' not in results_dev.columns:
                            results_dev['threshold'] = thresholds[project_name][0]

                        results = Parallel(n_jobs=-1)(delayed(test_models_bootstrap)(models_dict[problem_type[project_name]][model_name],results_dev.loc[r,:],scaler,imputer,X_dev,y_dev,
                                                                                    X_test,y_test,all_features,y_labels[project_name],metrics_names[problem_type[project_name]],IDs_test,boot_train,
                                                                                    boot_test,problem_type[project_name],threshold=results_dev.loc[r,'threshold']) 
                                                                                    for r in results_dev.index)
                        
                        results_test = pd.concat([pd.DataFrame(result[0],index=[0]) for result in results])
                        results_test['idx'] = results_dev['Unnamed: 0'].values

                        outputs_bootstrap = np.stack([result[1] for result in results],axis=0)
                        y_true_bootstrap = np.stack([result[2] for result in results],axis=0)
                        y_pred_bootstrap = np.stack([result[3] for result in results],axis=0)
                        IDs_test_bootstrap = np.stack([result[4] for result in results],axis=0)

                        results_test.to_csv(Path(file.parent,f'best_models_{model_name}_test_{scoring}.csv'))
                        
                        if not Path(file.parent,f'all_models_{model_name}_test.csv').exists():
                            with open(Path(file.parent,'y_test_bootstrap.pkl'),'wb') as f:
                                pickle.dump(y_true_bootstrap,f)
                            with open(Path(file.parent,f'IDs_test_bootstrap.pkl'),'wb') as f:   
                                pickle.dump(IDs_test_bootstrap,f)
                
                        with open(Path(file.parent,f'y_pred_bootstrap_{model_name}_{scoring}.pkl'),'wb') as f:
                            pickle.dump(y_pred_bootstrap,f)
                        
                except Exception as e:
                    logging.exception(e)