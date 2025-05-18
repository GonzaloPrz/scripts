import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier as xgboost
from xgboost import XGBRegressor as xgboostr
import itertools,pickle,sys, json
import logging,sys,os,argparse
from psrcal.calibration import AffineCalLogLoss

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

from expected_cost.ec import CostMatrix

##---------------------------------PARAMETERS---------------------------------##
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train models with hyperparameter optimization and feature selection'
    )
    parser.add_argument('--project_name', default='MCI_classifier',type=str,help='Project name')
    parser.add_argument('--stats', type=str, default='mean_std', help='Stats to be considered (default = all)')
    parser.add_argument('--shuffle_labels', type=int, default=0, help='Shuffle labels flag (1 or 0)')
    parser.add_argument('--stratify', type=int, default=1, help='Stratification flag (1 or 0)')
    parser.add_argument('--calibrate', type=int, default=0, help='Whether to calibrate models')
    parser.add_argument('--n_folds_outer', type=int, default=11, help='Number of folds for cross validation (outer loop)')
    parser.add_argument('--n_folds_inner', type=int, default=5, help='Number of folds for cross validation (inner loop)')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of hyperparameter iterations')
    parser.add_argument('--feature_selection',type=int,default=0,help='Whether to perform feature selection with RFE or not')
    parser.add_argument('--init_points', type=int, default=10, help='Number of random initial points to test during Bayesian optimization')
    parser.add_argument('--n_seeds_train',type=int,default=10,help='Number of seeds for cross-validation training')
    parser.add_argument('--n_seeds_shuffle',type=int,default=1,help='Number of seeds for shuffling')
    parser.add_argument('--scaler_name', type=str, default='StandardScaler', help='Scaler name')
    parser.add_argument('--id_col', type=str, default='id', help='ID column name')
    parser.add_argument('--n_boot',type=int,default=200,help='Number of bootstrap iterations')
    parser.add_argument('--n_boot_train',type=int,default=0,help='Number of bootstrap iterations for training')
    parser.add_argument('--n_boot_test',type=int,default=2000,help='Number of bootstrap iterations for testing')
    parser.add_argument('--shuffle_all',type=int,default=1,help='Whether to shuffle all models or only the best ones')
    parser.add_argument('--filter_outliers',type=int,default=0,help='Whether to filter outliers in regression problems')
    parser.add_argument('--early_fusion',type=int,default=1,help='Whether to perform early fusion')
    parser.add_argument('--overwrite',type=int,default=0,help='Whether to overwrite past results or not')
    parser.add_argument('--parallel',type=int,default=1,help='Whether to parallelize processes or not')
    parser.add_argument('--n_seeds_test',type=int,default=1,help='Number of seeds for testing')

    return parser.parse_args()

def load_configuration(args):
    # Global configuration dictionaries
    config = dict(
        project_name = args.project_name,
        stats = str(args.stats),
        shuffle_labels = bool(args.shuffle_labels),
        shuffle_all = bool(args.shuffle_all),
        stratify = bool(args.stratify),
        calibrate = bool(args.calibrate),
        n_folds_outer = float(args.n_folds_outer),
        n_folds_inner = float(args.n_folds_inner),
        n_iter = float(args.n_iter),
        feature_selection = bool(args.feature_selection),
        init_points = float(args.init_points),
        n_seeds_train = float(args.n_seeds_train) if args.n_folds_outer!= -1 else float(1),
        n_seeds_shuffle = float(args.n_seeds_shuffle) if args.shuffle_labels else float(0),
        scaler_name = args.scaler_name,
        id_col = args.id_col,
        n_boot = float(args.n_boot),
        n_boot_test = float(args.n_boot_test),
        n_boot_train = float(args.n_boot_train),
        filter_outliers = bool(args.filter_outliers),
        early_fusion = bool(args.early_fusion),
        overwrite = bool(args.overwrite),
        parallel = bool(args.parallel),
        n_seeds_test = float(args.n_seeds_test) if args.n_folds_outer!= -1 else float(0)
    )

    return config

args = parse_args()
config = load_configuration(args)
project_name = config['project_name']

logging.info('Configuration loaded. Starting training...')
logging.info('Training completed.')

##------------------ Configuration and Parameter Parsing ------------------##
home = Path(os.environ.get('HOME', Path.home()))
if 'Users/gp' in str(home):
    data_dir = home / 'data' / project_name
else:
    data_dir = Path('D:/CNC_Audio/gonza/data', project_name)

results_dir = Path(str(data_dir).replace('data', 'results'))

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
if not isinstance(scoring_metrics,list):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]
cmatrix = CostMatrix(np.array(main_config["cmatrix"][project_name])) if main_config["cmatrix"][project_name] is not None else None

config['test_size'] = float(test_size)
config['data_file'] = data_file
config['tasks'] = tasks
config['single_dimensions'] = single_dimensions    

config['scoring_metrics'] = scoring_metrics
config['problem_type'] = problem_type
config['y_labels'] = y_labels
config['avoid_stats'] = list(set(['min','max','median','skewness','kurtosis','std','mean']) - set(config['stats'].split('_'))) if config['stats'] != '' else []
config['stat_folder'] = '_'.join(sorted(config['stats'].split('_')))

config['random_seeds_train'] = [int(3**x) for x in np.arange(1, config['n_seeds_train']+1)]
config['random_seeds_test'] = [int(3**x) for x in np.arange(1, config['n_seeds_test']+1)] if config['test_size'] > 0 else ['']
config['random_seeds_shuffle'] = [float(3**x) for x in np.arange(1, config['n_seeds_shuffle']+1)] if config['shuffle_labels'] else ['']

if config['n_folds_outer'] == 0:
    config['kfold_folder'] = 'l2ocv'
elif config['n_folds_outer'] == -1:
    config['kfold_folder'] = 'loocv'
else:
    config['kfold_folder'] = f'{int(config["n_folds_outer"])}_folds'

if config['calibrate']:
    calmethod = AffineCalLogLoss
    calparams = {'bias':True, 'priors':None}
else:
    calmethod = None
    calparams = None

models_dict = {'clf':{'svc':SVC,
                    'lr':LR,
                    #'knnc':KNNC,
                    'xgb':xgboost},
                
                'reg':{'lasso':Lasso,
                    'ridge':Ridge,
                    'elastic':ElasticNet,
                    'knnr':KNNR,
                    'svr':SVR,
                    'xgb':xgboostr
                    }
}

hyperp = {'lr':{'C':(1e-4,100)},
          'svc':{'C':(1e-4,100),
                 'gamma':(1e-4,1e4)},
            'knnc':{'n_neighbors':(1,40)},
            'xgb':{'max_depth':(1,10),
                   'n_estimators':(1,2000),
                   'learning_rate':(1e-4,1)},
            'lasso':{'alpha':(1e-4,1e4)},
            'ridge':{'alpha':(1e-4,1e4)},
            'elastic':{'alpha':(1e-4,1e4),
                       'l1_ratio':(0,1)},
            'knnr':{'n_neighbors':(1,40)},
            'svr':{'C':(1e-4,100),
                    'gamma':(1e-4,1e4)}
            }

                
with open(Path(__file__).parent/'config.json', 'w') as f:
    json.dump(config, f, indent=4)

for y_label,task,scoring in itertools.product(y_labels,tasks,scoring_metrics):
    dimensions = list()
    if isinstance(single_dimensions,dict):
        single_dimensions_ = single_dimensions[task]
    else:
        single_dimensions_ = single_dimensions

    if isinstance(single_dimensions_,list):
        if config["early_fusion"]:
            for ndim in range(len(single_dimensions_)):
                for dimension in itertools.combinations(single_dimensions_,ndim+1):
                    dimensions.append('__'.join(dimension))
        else:
            dimensions = single_dimensions_
    
    else:
        dimensions = single_dimensions[task]

    for dimension in dimensions:
        print(task,dimension)
        
        if problem_type == 'clf':
            data = pd.read_csv(Path(data_dir,data_file))
        else:
            data = pd.read_excel(Path(data_dir,data_file)) if 'xlsx' in data_file else pd.read_csv(Path(data_dir,data_file))
        
        if problem_type == 'reg' and config['filter_outliers']:
            data = data[np.abs(data[y_label]-data[y_label].mean()) <= (3*data[y_label].std())]

        if config['shuffle_labels'] and problem_type == 'clf':
            np.random.seed(0)
            zero_indices = np.where(y == 0)[0]
            one_indices = np.where(y == 1)[0]

            # Shuffle and select half of the indices for flipping
            zero_to_flip = np.random.choice(zero_indices, size=len(zero_indices) // 2, replace=False)
            one_to_flip = np.random.choice(one_indices, size=len(one_indices) // 2, replace=False)

            # Flip the values at the selected indices
            y[zero_to_flip] = 1
            y[one_to_flip] = 0

        elif config['shuffle_labels']:
            np.random.seed(42)
            #Perform random permutations of the labels
            y = np.random.permutation(y)
    
        all_features = [col for col in data.columns if any(f'{x}__{y}__' in col for x,y in itertools.product(task.split('__'),dimension.split('__')))]
        
        data = data[all_features + [y_label,config['id_col']]]
        
        features = all_features
        
        ID = data.pop(config['id_col'])

        y = data.pop(y_label)

        for model_key, model_class in models_dict[problem_type].items():        
            print(model_key)
            
            held_out = (config['n_iter'] > 0 or config['n_iter_features'] > 0)
            n_folds_outer = int(config['n_folds_outer'])
            n_folds_inner = int(config['n_folds_inner'])
            if held_out:
                if n_folds_outer== 0:
                    n_folds_outer= int((data.shape[0]*(1 - config['test_size'])) / 2)
                elif n_folds_outer== -1:
                    n_folds_outer= int(data.shape[0]*(1 - config['test_size']))
                n_seeds_test = config['n_seeds_test']
            else:
                if n_folds_outer== 0:
                    n_folds_outer= int(data.shape[0]/2)
                elif n_folds_outer== -1:
                    n_folds_outer= data.shape[0]
                n_seeds_test = 1

            random_seeds_test = config['random_seeds_test']
            
            CV_outer = (StratifiedKFold(n_splits=int(n_folds_outer), shuffle=True)
                        if config['stratify'] and problem_type == 'clf'
                        else KFold(n_splits=n_folds_outer, shuffle=True))  
            
            CV_inner = (StratifiedKFold(n_splits=int(n_folds_inner), shuffle=True)
                        if config['stratify'] and problem_type == 'clf'
                        else KFold(n_splits=n_folds_inner, shuffle=True))            
            subfolders = [
                task, dimension, config['scaler_name'],
                config['kfold_folder'], y_label, config['stat_folder'],'bayes',scoring,
                'hyp_opt' if config['n_iter'] > 0 else '','feature_selection' if config['feature_selection'] else '',
                'filter_outliers' if config['filter_outliers'] and problem_type == 'reg' else '',
                'shuffle' if config['shuffle_labels'] else ''
            ]

            path_to_save = results_dir.joinpath(*[str(s) for s in subfolders if s])
            path_to_save.mkdir(parents=True, exist_ok=True)

            for random_seed_test in random_seeds_test:
                if test_size > 0:
                    X_train_, X_test_, y_train_, y_test_, ID_train_, ID_test_ = train_test_split(
                        data, y, ID,
                        test_size=config['test_size'],
                        random_state=int(random_seed_test),
                        shuffle=True,
                        stratify=y if (config['stratify'] and problem_type == 'clf') else None
                    )
                    # Reset indexes after split.
                    X_train_.reset_index(drop=True, inplace=True)
                    X_test_.reset_index(drop=True, inplace=True)
                    y_train_ = y_train_.reset_index(drop=True)
                    y_test_ = y_test_.reset_index(drop=True)
                    ID_train_ = ID_train_.reset_index(drop=True)
                    ID_test_ = ID_test_.reset_index(drop=True)
                else:
                    X_train_, y_train_, ID_train_ = data.reset_index(drop=True), y.reset_index(drop=True), ID.reset_index(drop=True)
                    X_test_, y_test_, ID_test_ = pd.DataFrame(), pd.Series(), pd.Series()

                hyperp['knnc']['n_neighbors'] = (1,int(X_train_.shape[0]*(1-test_size)*(1-1/n_folds_outer)**2-1))
                hyperp['knnr']['n_neighbors'] = (1,int(X_train_.shape[0]*(1-test_size)*(1-1/n_folds_outer)**2-1))

                # Check for data leakage.
                assert set(ID_train_).isdisjoint(set(ID_test_)), 'Data leakage detected between train and test sets!'
                
                if (Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '', f'all_models_{model_key}.csv').exists() and config['calibrate'] == False) or (Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '', f'cal_outputs_{model_key}.pkl').exists() and config['calibrate']):
                    if not bool(config['overwrite']):
                        print(f'Results already exist for {task} - {y_label} - {model_key}. Skipping...')
                        continue
                
                print(f'Training model: {model_key}')
    
                all_models,outputs_best,y_dev,y_pred_best,IDs_dev = utils.nestedCVT(model_class=models_dict[problem_type][model_key],
                                                                                     scaler=StandardScaler if config['scaler_name'] == 'StandardScaler' else MinMaxScaler,
                                                                                     imputer=KNNImputer,
                                                                                     X=X_train_,
                                                                                     y=y_train_,
                                                                                     n_iter=int(config['n_iter']),
                                                                                     iterator_outer=CV_outer,
                                                                                     iterator_inner=CV_inner,
                                                                                     random_seeds_outer=config['random_seeds_train'],
                                                                                     hyperp_space=hyperp[model_key],
                                                                                     IDs=ID_train_,
                                                                                     init_points=int(config['init_points']),
                                                                                     scoring=scoring,
                                                                                     problem_type=problem_type,
                                                                                     cmatrix=cmatrix,priors=None,
                                                                                     threshold=thresholds,
                                                                                     parallel=config['parallel'],
                                                                                     calmethod=calmethod,
                                                                                     calparams=calparams)

                Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '').mkdir(parents=True, exist_ok=True)
                with open(Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '','config.json'),'w') as f:
                    json.dump(config,f)
            
                all_models.to_csv(Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '',f'all_models_{model_key}.csv'),index=False)
                result_files = {
                    'X_train.pkl': X_train_,
                    'y_train.pkl': y_train_,
                    'IDs_train.pkl': ID_train_,
                    'y_dev.pkl': y_dev,
                    'IDs_dev.pkl': IDs_dev,
                    f'outputs_{model_key}.pkl': outputs_best}
                
                if test_size > 0:
                    result_files.update({
                        'X_test.pkl': X_test_,
                        'y_test.pkl': y_test_,
                        'IDs_test.pkl': ID_test_,
                    })
                for fname, obj in result_files.items():
                    with open(Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '', fname), 'wb') as f:
                        pickle.dump(obj, f)

                with open(Path(path_to_save,f'random_seed_{int(random_seed_test)}' if config['test_size'] else '', 'config.json'), 'w') as f:
                    json.dump(config, f, indent=4)