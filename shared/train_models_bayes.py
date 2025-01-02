import numpy as np
import pandas as pd
from pathlib import Path
import math 

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
from joblib import Parallel, delayed
from skopt.space import Real, Integer, Categorical
from random import randint as randint_random 

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

from expected_cost.ec import *
from expected_cost.utils import *

##---------------------------------PARAMETERS---------------------------------##
project_name = 'GeroApathy_reg'
n_folds = 5
filter_outliers = True
feature_selection = True
shuffle_labels = False
n_iter = 15
init_points = 10
scaler_name = 'StandardScaler'
id_col = 'id'
stratify = True

# Check if required arguments are provided
if len(sys.argv) > 1:
    project_name = sys.argv[1]
if len(sys.argv) > 2:
    feature_selection = bool(int(sys.argv[2]))
if len(sys.argv) > 3:
    filter_outliers = bool(int(sys.argv[3]))
if len(sys.argv) > 4:
    shuffle_labels = bool(int(sys.argv[4]))
if len(sys.argv) > 5:
    n_folds = int(sys.argv[5])
if len(sys.argv) > 6:
    n_iter = int(sys.argv[6])
if len(sys.argv) > 7:
    init_points = int(sys.argv[7])

l2ocv = False

cmatrix = None 

n_seeds_train = 10

random_seeds_train = [3**x for x in range(n_seeds_train)] if n_seeds_train > 0 else ['']

thresholds = {'tell_classifier':[np.log(0.5)],
              'MCI_classifier':[np.log(0.5)],
                'Proyecto_Ivo':[np.log(0.5)],
                'GeroApathy':[np.log(0.5)],
                'GeroApathy_reg':[None],
                'GERO_Ivo':[None]}

test_size = {'tell_classifier':0,
             'MCI_classifier':0,
            'Proyecto_Ivo':0,
            'GeroApathy':0,
            'GeroApathy_reg':0,
            'GERO_Ivo':0}

n_seeds_test_ = 0 if test_size[project_name] == 0 else 1

##---------------------------------PARAMETERS---------------------------------##
data_file = {'tell_classifier':'data_MOTOR-LIBRE.csv',
            'MCI_classifier':'features_data.csv',
            'Proyecto_Ivo':'data_total.csv',
            'GeroApathy':'data_matched_agradable',
            'GeroApathy_reg':'all_data_agradable.csv',
            'GERO_Ivo':'all_data.csv'}

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['Animales','P','Animales__P','cog','brain','AAL','conn'],
         'GeroApathy':['agradable'],
         'GeroApathy_reg':['agradable'],
         'GERO_Ivo':['animales','grandmean','fas__animales','fas']
         }

single_dimensions = {'tell_classifier':['voice-quality','talking-intervals','pitch'],
                     'MCI_classifier':['talking-intervals','psycholinguistic'],
                     'Proyecto_Ivo':{'Animales':['properties','timing','properties__timing','properties__vr','timing__vr','properties__timing__vr'],
                                     'P':['properties','timing','properties__timing','properties__vr','timing__vr','properties__timing__vr'],
                                     'Animales__P': ['properties','timing','properties__timing','properties__vr','timing__vr','properties__timing__vr'],
                                     'cog':['neuropsico','neuropsico_mmse'],
                                     'brain':['norm_brain_lit'],
                                     'AAL':['norm_AAL'],
                                     'conn':['connectivity']
                                     },
                        'GeroApathy':['mfcc','pitch','ratio','talking-intervals'],
                        'GeroApathy_reg':['mfcc','pitch','ratio','talking-intervals'],
                        'GERO_Ivo':['psycholinguistic','speech-timing']
}

scoring = {'tell_classifier':'norm_cross_entropy',
            'MCI_classifier':'norm_cross_entropy',
            'Proyecto_Ivo':'roc_auc_score',
            'GeroApathy':'roc_auc_score',
            'GeroApathy_reg':'mean_absolute_error',
            'GERO_Ivo':'r2_score'}

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
    kfold_folder = f'{n_folds}_folds'
models_dict = {'clf':{'lr':LR,
                    #'svc':SVC,
                    'knnc':KNNC,
                    'xgb':xgboost},
                
                'reg':{'lasso':Lasso,
                    'ridge':Ridge,
                    'elastic':ElasticNet,
                    'knnr':KNNR,
                    'svr':SVR,
                    'xgb':xgboostr
                    }
}

y_labels = {'tell_classifier':['target'],
            'MCI_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GeroApathy':[#'DASS_21_Depression_V_label','Depression_Total_Score_label','AES_Total_Score_label',
                            #'MiniSea_minisea_total_label',
                            'MiniSea_MiniSea_Total_EkmanFaces_label'
                          ],
            'GeroApathy_reg':['DASS_21_Depression_V','Depression_Total_Score','AES_Total_Score',
                                #'MiniSea_minisea_total','MiniSea_MiniSea_Total_EkmanFaces'
                          ],
            'GERO_Ivo':['MMSE_Total_Score','IFS_Total_Score','ACEIII_Total_Score']
}

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GeroApathy':'clf',
                'GeroApathy_reg':'reg',
                'GERO_Ivo':'reg'}

hyperp = {'lr':{'C':(1e-4,100)},
          'svc':{'C':(1e-4,100),
                 'gamma':(1e-4,1e4),
                 'probability':[True]},
            'knnc':{'n_neighbors':(1,40)},
            'xgb':{'max_depth':(1,10),
                   'n_estimators':(1,1000),
                   'learning_rate':(1e-4,1)},
            'lasso':{'alpha':(1e-4,1e4)},
            'ridge':{'alpha':(1e-4,1e4)},
            'elastic':{'alpha':(1e-4,1e4),
                       'l1_ratio':(0,1)},
            'knnr':{'n_neighbors':(1,40)},
            'svr':{'C':(1e-4,100),
                    'gamma':(1e-4,1e4)}
            }

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
        print(task,dimension)
        
        if problem_type[project_name] == 'clf':
            data = pd.read_csv(Path(data_dir,f'{data_file[project_name]}_{y_label}.csv'))
        else:
            data = pd.read_excel(Path(data_dir,data_file[project_name])) if 'xlsx' in data_file else pd.read_csv(Path(data_dir,data_file[project_name]))
        
        if problem_type[project_name] == 'reg' and filter_outliers:
            data = data[np.abs(data[y_label]-data[y_label].mean()) <= (3*data[y_label].std())]

        if shuffle_labels and problem_type[project_name] == 'clf':
            np.random.seed(0)
            zero_indices = np.where(y == 0)[0]
            one_indices = np.where(y == 1)[0]

            # Shuffle and select half of the indices for flipping
            zero_to_flip = np.random.choice(zero_indices, size=len(zero_indices) // 2, replace=False)
            one_to_flip = np.random.choice(one_indices, size=len(one_indices) // 2, replace=False)

            # Flip the values at the selected indices
            y[zero_to_flip] = 1
            y[one_to_flip] = 0

        elif shuffle_labels:
            np.random.seed(42)
            #Perform random permutations of the labels
            y = np.random.permutation(y)
    
        all_features = [col for col in data.columns if any(f'{x}_{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__'))) and isinstance(data.loc[0,col],(int,float)) and 'timestamp' not in col]
        
        data = data[all_features + [y_label,id_col]]
        
        features = all_features
        
        ID = data.pop(id_col)

        y = data.pop(y_label)

        for model in models_dict[problem_type[project_name]].keys():        
            print(model)
            
            random_seeds_test = np.arange(n_seeds_test_) if test_size[project_name] > 0 else ['']

            CV_type = StratifiedKFold(n_splits=n_folds,shuffle=True) if stratify and problem_type[project_name] == 'clf' else KFold(n_splits=n_folds,shuffle=True)

            path_to_save = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','bayes','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type[project_name] == 'reg' else '','shuffle' if shuffle_labels else '')
            
            path_to_save.mkdir(parents=True,exist_ok=True)

            config = {'n_iter':n_iter,
                      'init_points':init_points,
            'test_size':test_size[project_name],
            'cmatrix':str(cmatrix)}

            with open(Path(path_to_save,'config.json'),'w') as f:
                json.dump(config,f)
            
            for random_seed_test in random_seeds_test:
                if test_size[project_name] > 0:
                    path_to_save_final = Path(path_to_save,f'random_seed_{random_seed_test}')

                    X_train,X_test,y_train,y_test,ID_train,ID_test = train_test_split(data,y,ID,test_size=test_size[project_name],random_state=random_seed_test,shuffle=True,stratify=y if stratify and problem_type[project_name] == 'clf' else None)
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

                hyperp['knnc']['n_neighbors'] = (1,int(X_train.shape[0]*(1-test_size[project_name])*(1-1/n_folds)**2-1))
                hyperp['knnr']['n_neighbors'] = (1,int(X_train.shape[0]*(1-test_size[project_name])*(1-1/n_folds)**2-1))

                path_to_save_final.mkdir(parents=True,exist_ok=True)

                assert not set(ID_train).intersection(set(ID_test)), "Data leakeage detected between train and test sets!"

                if Path(path_to_save_final,f'all_models_{model}.csv').exists():
                    continue
                
                with open(Path(path_to_save_final,'config.json'),'w') as f:
                    json.dump(config,f)
                
                all_models,outputs_best,y_true,y_pred_best,IDs_val = nestedCVT(models_dict[problem_type[project_name]][model],scaler,imputer,X_train,y_train,n_iter,CV_type,CV_type,random_seeds_train,hyperp[model],ID_train,
                                                                               init_points=init_points,scoring=scoring[project_name],problem_type=problem_type[project_name],cmatrix=cmatrix,priors=None,
                                                                               threshold=thresholds[project_name],feature_selection=feature_selection)

                all_models.to_csv(Path(path_to_save_final,f'all_models_{model}.csv'),index=False)
                pickle.dump(outputs_best,open(Path(path_to_save_final,f'outputs_best_{model}.pkl'),'wb'))
                pickle.dump(y_true,open(Path(path_to_save_final,f'y_true_dev.pkl'),'wb'))
                pickle.dump(y_pred_best,open(Path(path_to_save_final,f'y_pred_best_{model}.pkl'),'wb'))
                pickle.dump(IDs_val,open(Path(path_to_save_final,f'IDs_val.pkl'),'wb'))