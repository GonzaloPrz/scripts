import numpy as np
import pandas as pd
from pathlib import Path
import math 
import logging, sys
import torch

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as xgboost
from xgboost import XGBRegressor as xgboostr
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm
import itertools,pickle,sys, json
from scipy.stats import loguniform, uniform, randint
from random import randint as randint_random 

from random import randint as randint_random 

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *
##---------------------------------PARAMETERS---------------------------------##

# Check if required arguments are provided
if len(sys.argv) < 2:
    print("Usage: python train_models.py <project_name> [hyp_opt] [filter_outliers] [shuffle_labels] [stratify] [k] [n_iter_features] [feature_sample_ratio] [scaler_name] [id_col] [n_seeds_train]")
    sys.exit(1)

# Parse arguments
project_name = sys.argv[1]
# Default values
hyp_opt = True
filter_outliers = True
shuffle_labels = False
stratify = True
n_folds = 5
n_iter = 50
n_iter_features = 50
feature_sample_ratio = 0.5
scaler_name = 'StandardScaler'
id_col = 'id'
n_seeds_train = 10

# Check and assign arguments if provided
if len(sys.argv) > 2:
    hyp_opt = bool(int(sys.argv[2]))
if len(sys.argv) > 3:
    filter_outliers = bool(int(sys.argv[3]))
if len(sys.argv) > 4:
    shuffle_labels = bool(int(sys.argv[4]))
if len(sys.argv) > 5:
    stratify = bool(int(sys.argv[5]))
if len(sys.argv) > 6:
    n_folds = int(sys.argv[6])
if len(sys.argv) > 7:
    n_iter = int(sys.argv[7])
if len(sys.argv) > 8:
    n_iter_features = int(sys.argv[8])
if len(sys.argv) > 9:
    feature_sample_ratio = float(sys.argv[9])
if len(sys.argv) > 10:
    scaler_name = sys.argv[10]
if len(sys.argv) > 11:
    id_col = sys.argv[11]
if len(sys.argv) > 12:
    n_seeds_train = int(sys.argv[12])

parallel = True

l2ocv = False

cmatrix = None 

random_seeds_train = [3**x for x in np.arange(1,n_seeds_train+1)] if n_seeds_train > 0 else ['']

thresholds = {'tell_classifier':[0.5],
              'MCI_classifier':[0.5],
                'Proyecto_Ivo':[0.5],
                'GeroApathy':[0.5],
                'GERO_Ivo':[None]}

test_size = {'tell_classifier':0.3,
             'MCI_classifier':0.3,
            'Proyecto_Ivo':0,
            'GeroApathy':0.3,
            'GERO_Ivo':0.3}

n_seeds_test_ = 0 if test_size[project_name] == 0 else 1

##---------------------------------PARAMETERS---------------------------------##

data_file = {'tell_classifier':'data_MOTOR-LIBRE.csv',
            'MCI_classifier':'features_data.csv',
            'Proyecto_Ivo':'data_total.csv',
            'GeroApathy':'all_data_agradable.csv',
            'GERO_Ivo':'all_data.csv'}

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['cog'],
         'GeroApathy':['agradable'],
         'GERO_Ivo':['fas','animales','fas__animales','grandmean']
         }

single_dimensions = {'tell_classifier':['voice-quality','talking-intervals','pitch'],
                     'MCI_classifier':['talking-intervals','psycholinguistic'],
                     'Proyecto_Ivo':{'Animales':['properties','timing','properties__timing','properties__vr','timing__vr','properties__timing__vr'],
                                     'P':['properties','timing','properties__timing','properties__vr','timing__vr','properties__timing__vr'],
                                     'Animales__P': ['properties','timing','properties__timing','properties__vr','timing__vr','properties__timing__vr'],
                                     'cog':['neuropsico_digits__neuropsico_tmt','neuropsico_tmt','neuropsico_digits'],
                                     'brain':['norm_brain_lit'],
                                     'AAL':['norm_AAL'],
                                     'conn':['connectivity']
                                     },
                        'GeroApathy':['formants','mfcc','pitch','talking-intervals'],
                        'GERO_Ivo':['psycholinguistic','speech-timing']
}

if scaler_name == 'StandardScaler':
    scaler = StandardScaler
elif scaler_name == 'MinMaxScaler':
    scaler = MinMaxScaler
else:
    scaler = None
imputer = KNNImputer

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

models_dict = {'clf': {'lr':LR,
                    'svc':SVC,
                    'knnc':KNNC,
                    'xgb':xgboost},
                'reg':{'lasso':Lasso,
                    'ridge':Ridge,
                    'elastic':ElasticNet,
                    #'knnr':KNNR,
                    'svr':SVR,
                    'xgb':xgboostr
                    }
                }

y_labels = {'tell_classifier':['target'],
            'MCI_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GeroApathy':['DASS_21_Depression_V_label','Depression_Total_Score_label','AES_Total_Score_label',
                          'MiniSea_MiniSea_Total_EkmanFaces_label','MiniSea_minisea_total_label'
                          ],
            'GERO_Ivo':[#'GM_norm','WM_norm','norm_vol_bilateral_HIP','norm_vol_mask_AD',
                        'MMSE_Total_Score','ACEIII_Total_Score','IFS_Total_Score','MoCA_Total_Boni_3'
                        ]
            }

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GeroApathy':'clf',
                'GERO_Ivo':'reg'}

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)
results_dir = Path(str(data_dir).replace('data','results'))

for y_label,task in itertools.product(y_labels[project_name],tasks[project_name]):
    dimensions = list()
    if isinstance(single_dimensions[project_name],list):
        for ndim in range(1,len(single_dimensions[project_name])+1):
            for dimension in itertools.combinations(single_dimensions[project_name],ndim):
                dimensions.append('__'.join(dimension))

    if len(dimensions) == 0:
        dimensions = single_dimensions[project_name][task]
    
    for dimension in dimensions:
        print(y_label,task,dimension)
        data = pd.read_excel(Path(data_dir,data_file[project_name])) if 'xlsx' in data_file else pd.read_csv(Path(data_dir,data_file[project_name]))

        if shuffle_labels:
            np.random.seed(15)
            if problem_type[project_name] == 'clf':
                #Randomly change some 1s to 0s and viceversa

                data[y_label] = np.where(np.random.rand(data.shape[0]) > 0.5,data[y_label],np.where(data[y_label] == 1,0,1))    
            else:
                #Randomly shuffle the values of the target variable
                data[y_label] = np.random.permutation(data[y_label].values)

        all_features = [col for col in data.columns if any(f'{x}_{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__'))) and not isinstance(data.loc[0,col],str) and 'timestamp' not in col]
        
        data = data[all_features + [y_label,id_col]]
        
        data = data.dropna(subset=[y_label])

        #Filter outliers for regression problems:
        if problem_type[project_name] == 'reg' and filter_outliers:
            data = data[np.abs(data[y_label]-data[y_label].mean()) <= (3*data[y_label].std())]

        features = all_features
        
        ID = data.pop(id_col)

        y = data.pop(y_label)

        for model in models_dict[problem_type[project_name]].keys():        
            print(model)
            held_out = True if n_iter > 0 or n_iter_features > 0 else False

            if held_out:
                if l2ocv:
                    n_folds = int((data.shape[0]*(1-test_size[project_name]))/2)
                n_seeds_test = n_seeds_test_
            else:
                if l2ocv:
                    n_folds = int(data.shape[0]/2)
                n_seeds_test = 1
            
            random_seeds_test = np.arange(n_seeds_test) if test_size[project_name] > 0 else ['']

            CV_type = StratifiedKFold(n_splits=n_folds,shuffle=True) if stratify else KFold(n_splits=n_folds,shuffle=True)

            path_to_save = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt' if n_iter > 0 else 'no_hyp_opt','feature_selection' if n_iter_features >0 else '','filter_outliers' if filter_outliers and problem_type[project_name] == 'reg' else '','shuffle' if shuffle_labels else '')

            print(path_to_save)

            path_to_save.mkdir(parents=True,exist_ok=True)

            if shuffle_labels:
                predefined_models = True if Path(path_to_save,random_seeds_test[0],f'all_models_{model}').exists() else False
            else:
                predefined_models = False

            config = {'n_iter':n_iter,
            'test_size':test_size[project_name],
            'n_feature_sets': n_iter_features,
            'feature_sample_ratio':feature_sample_ratio,
            'cmatrix':str(cmatrix)}
            
            if predefined_models == False:
                hyperp = {'lr': pd.DataFrame({'C': 1},index=[0]),
                            'lda':pd.DataFrame({'solver':'lsqr'},index=[0]),
                            'knnc': pd.DataFrame({'n_neighbors':5},index=[0]),
                            'svc': pd.DataFrame({'C': 1,
                                    'gamma': 'scale',
                                    'kernel':'rbf',
                                    'probability':True},index=[0]),
                            'xgb': pd.DataFrame({'n_estimators':100,
                                    'max_depth':6,
                                    'learning_rate':0.3
                                    },index=[0]),
                            'ridge': pd.DataFrame({'alpha': 1,
                                            'tol':.0001,
                                            'solver':'auto',
                                            'random_state':42},index=[0]),
                            'knnr': pd.DataFrame({'n_neighbors':5},index=[0]),
                            'lasso': pd.DataFrame({'alpha': 1,
                                                'tol':.0001,
                                                'random_state':42},index=[0]),
                            'elastic': pd.DataFrame({'alpha': 1,
                                                'l1_ratio':.5,
                                                'tol':.0001,
                                                'random_state':42},index=[0]),
                            'svr': pd.DataFrame({'C':1,
                                                'kernel':'rbf',
                                                'gamma':'scale'},index=[0])
                                        
                        }
                
                for n in range(n_iter):
                    new_combination = dict((key,{}) for key in models_dict[problem_type[project_name]].keys())
                    new_combination['lr'] = {'C': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))])}
                    new_combination['svc'] = {'C': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))]),
                                            'kernel': np.random.choice(['linear', 'rbf', 'sigmoid']),
                                            'gamma': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))]),
                                            'probability': True}
                    new_combination['knnc'] = {'n_neighbors': int(randint(1, int((n_folds - 1) / n_folds * (data.shape[0] * (1-test_size[project_name])))).rvs())}
                    new_combination['xgb'] = {'n_estimators': int(randint(10,1000).rvs()),
                                            'max_depth': randint(1, 10).rvs(),
                                            'learning_rate': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))])
                                            }
                    new_combination['ridge'] = {'alpha': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-3, 2))]),
                                            'tol': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-5, 0))]),
                                            'solver':'auto',
                                            'random_state':42}
                    new_combination['lasso'] = {'alpha': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-3, 2))]),
                                                'tol': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-5, 0))]),
                                                'random_state':42}
                    new_combination['elastic'] = {'alpha': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-3, 2))]),
                                                'l1_ratio': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-4, -1))]),
                                                'tol': np.random.choice([x*10**y for x,y in itertools.product(range(1, 10),range(-5, 0))]),
                                                'random_state':42}
                    new_combination['knnr'] = {'n_neighbors': randint(1, int((n_folds - 1) / n_folds * (data.shape[0] * (1-test_size[project_name])))).rvs()},
                    new_combination['svr'] = {'C': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))]),
                                            'kernel': np.random.choice(['linear', 'rbf', 'sigmoid']),
                                            'gamma': np.random.choice([x*10**y for x,y in itertools.product(range(1,10),range(-3, 2))])}
                                            
                    
                    for key in models_dict[problem_type[project_name]].keys():
                        hyperp[key].loc[len(hyperp[key].index),:] = new_combination[key]
                
                    hyperp[model].drop_duplicates(inplace=True)
                    hyperp[model] = hyperp[model].reset_index(drop=True)

                num_comb = 0

                for k in range(np.min((int(feature_sample_ratio*data.shape[0]*(1-test_size[project_name]))-1,len(features)-1))):
                    num_comb += math.comb(len(features),k+1)

                feature_sets = list()

                if n_iter_features > num_comb:
                    # Generate combinations of features with different lengths
                    for k in range(np.min((int(feature_sample_ratio * data.shape[0] * (1 - test_size[project_name])) - 1, len(features) - 1))):
                        for comb in itertools.combinations(features, k + 1):
                            feature_sets.append(list(comb))
                else:
                    # Generate random feature samples
                    feature_sets = [
                        list(np.unique(np.random.choice(features, int(feature_sample_ratio * data.shape[0] * (1 - test_size[project_name])), replace=True)))
                        for _ in range(n_iter_features)
                    ]

                # Add the full set of features
                feature_sets.append(features)

            features_df = pd.DataFrame(index=np.arange(len(feature_sets)),columns=features)
            for f, feature_set in enumerate(feature_sets):
                features_df.loc[f,features] = [1 if feature in feature_set else 0 for feature in features]
            
            features_df.drop_duplicates(inplace=True)
            features_df = features_df.reset_index(drop=True)

            feature_sets = list()
            
            for r in features_df.index:
                feature_sets.append(list(features_df.columns[features_df.loc[r,:]==1]))

            for random_seed_test in random_seeds_test:
                                                                                                                                                                                                                                                                                                                                
                if test_size[project_name] > 0:
                    path_to_save_final = Path(path_to_save,f'random_seed_{random_seed_test}')

                    X_train,X_test,y_train,y_test,ID_train,ID_test = train_test_split(data,y,ID,test_size=test_size[project_name],random_state=random_seed_test,shuffle=True,stratify=y if stratify else None)
                    X_train.reset_index(drop=True,inplace=True)
                    X_test.reset_index(drop=True,inplace=True)
                    y_train.reset_index(drop=True,inplace=True)
                    y_test.reset_index(drop=True,inplace=True)
                    ID_train.reset_index(drop=True,inplace=True)
                    ID_test.reset_index(drop=True,inplace=True)
                else:
                    X_train = data.reset_index(drop=True)
                    y_train = y.reset_index(drop=True)
                    ID_train = ID.reset_index(drop=True)

                    X_test = pd.DataFrame()
                    y_test = pd.Series()
                    ID_test = pd.Series()
                    path_to_save_final = path_to_save

                path_to_save_final.mkdir(parents=True,exist_ok=True)

                if predefined_models:
                    random_seed_test_predefined = [folder for folder in path_to_save.iterdir()  if 'random_seed_' in folder.name]
                    if len(random_seed_test_predefined) == 0:
                        random_seed_test_predefined = ['']

                    models = pd.read_csv(Path(str(Path(path_to_save,random_seed_test_predefined[0])).replace('shuffle',''),f'all_models_{model}.csv'))
                    
                    all_features = models[[col for col in models.columns if any(f'{x}_{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__')))]].drop_duplicates()

                    feature_sets = [list([col for col in all_features.columns if all_features.loc[r,col] == 1]) for r in all_features.index]
                    # Remove duplicates by converting to set and back to list of lists
                    feature_sets = [list(t) for t in set(tuple(sorted(fs)) for fs in feature_sets)]
                    hyperp = {model:models[[col for col in models.columns if all(x not in col for x in ['Unnamed: 0','threshold'] + list(all_features.columns))]]}
                    hyperp[model] = hyperp[model].drop_duplicates()
                    if 'gamma' in hyperp[model].columns:
                        for r in hyperp[model].index:
                            try:
                                hyperp[model].loc[r,'gamma'] = float(hyperp[model].loc[r,'gamma'])
                            except:
                                pass

                assert not set(ID_train).intersection(set(ID_test)), "Data leakeage detected between train and test sets!"

                if Path(path_to_save_final,f'all_models_{model}.csv').exists():
                    continue
                
                with open(Path(path_to_save_final,'config.json'),'w') as f:
                    json.dump(config,f)

                log_file = Path(results_dir,Path(__file__).stem + '.log')

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

                models,outputs,y_pred,y_dev,IDs_dev = CVT(models_dict[problem_type[project_name]][model],scaler,imputer,X_train, y_train,CV_type,random_seeds_train,hyperp[model],feature_sets,ID_train,thresholds[project_name],cmatrix=cmatrix,parallel=parallel,problem_type=problem_type[project_name])        
            
                all_models = pd.DataFrame()
                
                for model_index in range(models.shape[0]):
                    model_ = {}
                    for param in models.keys():
                        if param in [y_label,id_col]:
                            continue
                        model_[param] = models.iloc[model_index][param]

                    all_models = pd.concat([all_models,pd.DataFrame(model_,index=[0])],ignore_index=True,axis=0)
                
                all_models.to_csv(Path(path_to_save_final,f'all_models_{model}.csv'),index=False)

                with open(Path(path_to_save_final,f'X_dev.pkl'),'wb') as f:
                    pickle.dump(X_train,f)
                with open(Path(path_to_save_final,f'y_true_dev.pkl'),'wb') as f:
                    pickle.dump(y_dev,f)
                with open(Path(path_to_save_final,f'y_dev.pkl'),'wb') as f:
                    pickle.dump(y_train,f) 
                with open(Path(path_to_save_final,f'IDs_dev.pkl'),'wb') as f:
                    pickle.dump(IDs_dev,f)
                with open(Path(path_to_save_final,f'X_test.pkl'),'wb') as f:
                    pickle.dump(X_test,f)
                with open(Path(path_to_save_final,f'y_test.pkl'),'wb') as f:
                    pickle.dump(y_test,f)
                with open(Path(path_to_save_final,f'IDs_test.pkl'),'wb') as f:
                    pickle.dump(ID_test,f)
                
                with open(Path(path_to_save_final,f'outputs_{model}.pkl'),'wb') as f:
                    pickle.dump(outputs,f)