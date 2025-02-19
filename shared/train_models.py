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
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm
import itertools,pickle,sys, json
from scipy.stats import loguniform, uniform, randint
from random import randint as randint_random 
import warnings,argparse,os

warnings.filterwarnings("ignore")

from random import randint as randint_random 

sys.path.append(str(Path(Path.home(),"scripts_generales"))) if "Users/gp" in str(Path.home()) else sys.path.append(str(Path(Path.home(),"gonza","scripts_generales")))

import utils

cmatrix = None
parallel = True

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train models with hyperparameter optimization and feature selection"
    )
    parser.add_argument("--project_name", type=str,help="Project name")
    parser.add_argument("--all_stats", type=int, default=1, help="All stats flag (1 or 0)")
    parser.add_argument("--shuffle_labels", type=int, default=0, help="Shuffle labels flag (1 or 0)")
    parser.add_argument("--stratify", type=int, default=1, help="Stratification flag (1 or 0)")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross validation")
    parser.add_argument("--n_iter", type=int, default=0, help="Number of hyperparameter iterations")
    parser.add_argument("--n_iter_features", type=int, default=0, help="Number of feature sets to try and select from")
    parser.add_argument("--feature_sample_ratio", type=float, default=0.5, help="Feature-to-sample ratio: number of features in each feature set = ratio * number of samples in the training set")
    parser.add_argument("--n_seeds_train",type=int,default=10,help="Number of seeds for cross-validation training")
    parser.add_argument("--n_seeds_shuffle",type=int,default=10,help="Number of seeds for shuffling")
    parser.add_argument("--scaler_name", type=str, default="StandardScaler", help="Scaler name")
    parser.add_argument("--id_col", type=str, default="id", help="ID column name")
    parser.add_argument("--n_models",type=float,default=0,help="Number of hyperparameter combinatios to try and select from  to train")
    parser.add_argument("--n_boot",type=int,default=200,help="Number of features to select")
    parser.add_argument("--bayesian",type=int,default=0,help="Whether to calculate bayesian credible intervals or bootstrap confidence intervals")
    parser.add_argument("--shuffle_all",type=int,default=1,help="Whether to shuffle all models or only the best ones")
    parser.add_argument("--filter_outliers",type=int,default=0,help="Whether to filter outliers in regression problems")
    parser.add_argument("--early_fusion",type=int,default=1,help="Whether to perform early fusion")

    return parser.parse_args()

def load_configuration(args):
    # Global configuration dictionaries
    config = dict(
        project_name = args.project_name,
        all_stats = bool(args.all_stats),
        shuffle_labels = bool(args.shuffle_labels),
        shuffle_all = bool(args.shuffle_all),
        stratify = bool(args.stratify),
        n_folds = float(args.n_folds),
        n_iter = float(args.n_iter),
        n_iter_features = float(args.n_iter_features),
        feature_sample_ratio = args.feature_sample_ratio,
        n_seeds_train = float(args.n_seeds_train) if args.n_folds != -1 else float(1),
        n_seeds_shuffle = float(args.n_seeds_shuffle) if args.shuffle_labels else float(0),
        scaler_name = args.scaler_name,
        id_col = args.id_col,
        n_models = float(args.n_models),
        n_boot = float(args.n_boot),
        bayesian = bool(args.bayesian),
        filter_outliers = bool(args.filter_outliers),
        early_fusion = bool(args.early_fusion)
    )

    return config

args = parse_args()
config = load_configuration(args)
project_name = config["project_name"]

logging.info("Configuration loaded. Starting training...")
logging.info("Training completed.")

##------------------ Configuration and Parameter Parsing ------------------##
home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    data_dir = home / "data" / project_name
else:
    data_dir = Path("D:/CNC_Audio/gonza/data", project_name)

results_dir = Path(str(data_dir).replace("data", "results"))

main_config = json.load(Path(Path(__file__).parent,"main_config.json").open())

y_labels = main_config["y_labels"][project_name]
tasks = main_config["tasks"][project_name]
test_size = main_config["test_size"][project_name]
single_dimensions = main_config["single_dimensions"][project_name]
data_file = main_config["data_file"][project_name]
thresholds = main_config["thresholds"][project_name]
scoring_metrics = main_config["scoring_metrics"][project_name]
problem_type = main_config["problem_type"][project_name]

config["test_size"] = float(test_size)
config["n_seeds_test"] = float(0) if config["test_size"]== 0 else float(1)
config["data_file"] = data_file
config["tasks"] = tasks
config["single_dimensions"] = single_dimensions        
config["scoring_metrics"] = scoring_metrics
config["problem_type"] = problem_type
# Model dictionaries. Note: KNNR and other regressors can be added as needed.    
config["y_labels"] = y_labels

# Determine which stats to avoid (if not all stats)
config["avoid_stats"] = ["min","max","median","skewness","kurtosis"] if not config["all_stats"] else []
config["stat_folder"] = "_".join(sorted(list(set(["mean","std","min","max","median","kurtosis","skewness"]) - set(config["avoid_stats"])))) if not config["all_stats"] else ""

config["random_seeds_train"] = [float(3**x) for x in np.arange(1, config["n_seeds_train"]+1)]
config["random_seeds_shuffle"] = config["random_seeds_train"][:int(config["n_seeds_shuffle"])] if config["shuffle_labels"] else [""]

if config["n_folds"] == 0:
    config["kfold_folder"] = "l2ocv"
elif config["n_folds"] == -1:
    config["kfold_folder"] = "loocv"
else:
    config["kfold_folder"] = f"{int(config['n_folds'])}_folds"

models_dict = {
        "clf": {
            "lr": LR,
            "svc": SVC,
            "knnc": KNNC,
            "xgb": xgboost,
            #"nb":GaussianNB
        },
        "reg": {
            "lasso": Lasso,
            "ridge": Ridge,
            "elastic": ElasticNet,
            "svr": SVR,
            "xgb": xgboostr
        }
    }

with Path(Path(__file__).parent,"default_hp.json").open("rb") as f:
    default_hp = json.load(f)

#C: inverse of regularization strength, 10**x, x in [-5,5]
hp_ranges = {
        "lr": {"C": [x*10**y for x,y in itertools.product(range(1,9),range(-3, 2))]},
        "svc": {"C": [x*10**y for x,y in itertools.product(range(1,9),range(-3, 2))], "gamma": ["scale", "auto"], "kernel": ["rbf", "linear", "poly", "sigmoid"], "probability": [True]},
        "knnc": {"n_neighbors": [x for x in range(1, 21)]},
        "xgb": {"n_estimators": [x*10**y for x,y in itertools.product(range(1,9),range(1,3))], "max_depth": [1, 2, 3, 4], "learning_rate": [0.1, 0.3, 0.5, 0.7, 0.9]},
        "nb": {"priors":[None]},
        "ridge": {"alpha": [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                  "tol": [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                  "solver": ["auto"],
                  "max_iter": [5000],
                  "random_state": [42]},
        "lasso": {"alpha": [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                  "tol": [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                  "max_iter": [5000], 
                  "random_state": [42]},
        "elastic": {"alpha": [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                    "l1_ratio": [x**-1 for x in range(1,10)], 
                    "tol": [x*10**y for x,y in itertools.product(range(1,9),range(-4, 0))], 
                  "max_iter": [5000],
                  "random_state": [42]},
        "svr": {"C": [x*10**y for x,y in itertools.product(range(1,9),range(-3, 2))], "kernel": ["rbf", "linear", "poly", "sigmoid"], "gamma": ["scale", "auto"]}
}
##------------------ Main Model Training Loop ------------------##

for y_label, task in itertools.product(y_labels, tasks):
    print(y_label)
    print(task)
    # Determine feature dimensions. For projects with a dictionary, pick based on the task.
    dimensions = []
    single_dims = single_dimensions
    if isinstance(single_dims, list) and config["early_fusion"]:
        for ndim in range(1, len(single_dims)+1):
            for dimension in itertools.combinations(single_dims, ndim):
                dimensions.append("__".join(dimension))
    else:
        dimensions = single_dims
    
    for dimension in dimensions:
        print(dimension)
        logging.info(f"Processing: y_label={y_label}, task={task}, dimension={dimension}")
        # Load dataset. Use CSV or Excel based on file extension.
        data_file = data_file
        data_path = data_dir / (data_file if data_file.endswith(".csv") else data_file)
        if problem_type == "clf":
            data = pd.read_csv(data_path if data_file.endswith(".csv") else data_path.with_suffix(".csv"))
        else:
            # For regression: Excel if available; default to CSV.
            data = pd.read_excel(data_path) if data_path.suffix in [".xlsx", ".xls"] else pd.read_csv(data_path)
        
        data.dropna(axis=1,how='all',inplace=True)

        # Identify feature columns (avoid stats and other unwanted columns)
        features = [col for col in data.columns if any(f"{x}__{y}__" in col 
                    for x,y in itertools.product(task.split("__"), dimension.split("__"))) 
                    and not isinstance(data.iloc[0][col], str) 
                    and all(f'_{x}' not in col for x in config["avoid_stats"] + ["query", "timestamp"])]
        # Select only the desired features along with the target and id
        data = data[features + [y_label, config["id_col"]]]
        data = data.dropna(subset=[y_label])
        
        # For regression, optionally filter outliers
        if problem_type == "reg" and config["filter_outliers"]:
            data = data[np.abs(data[y_label]-data[y_label].mean()) <= (3*data[y_label].std())]
        
        # Separate features, target and ID.
        ID = data.pop(config["id_col"])
        y = data.pop(y_label)
        
        # Iterate over each model defined for this problem type.
        for model_key, model_class in models_dict[problem_type].items():
            # Determine held-out settings based on hyperparameter or feature iterations.
            held_out = (config["n_iter"] > 0 or config["n_iter_features"] > 0)
            n_folds = int(config["n_folds"])
            if held_out:
                if n_folds == 0:
                    n_folds = int((data.shape[0]*(1 - config["test_size"])) / 2)
                elif n_folds == -1:
                    n_folds = int(data.shape[0]*(1 - config["test_size"]))
                n_seeds_test = config["n_seeds_test"]
            else:
                if n_folds == 0:
                    n_folds = int(data.shape[0]/2)
                elif n_folds == -1:
                    n_folds = data.shape[0]
                n_seeds_test = 1

            random_seeds_test = np.arange(n_seeds_test) if config["test_size"] > 0 else [""]
            # Choose cross-validation iterator
            CV_type = (StratifiedKFold(n_splits=int(n_folds), shuffle=True)
                        if config["stratify"] and problem_type == "clf"
                        else KFold(n_splits=n_folds, shuffle=True))
            
            # Construct a path to save results (with clear folder names)
            subfolders = [
                task, dimension, config["scaler_name"],
                config["kfold_folder"], y_label, config["stat_folder"],
                "hyp_opt" if config["n_iter"] > 0 else "no_hyp_opt",
                "feature_selection" if config["n_iter_features"] > 0 else "",
                "filter_outliers" if config["filter_outliers"] and problem_type == "reg" else "",
                "shuffle" if config["shuffle_labels"] else ""
            ]
            path_to_save = results_dir.joinpath(*[str(s) for s in subfolders if s])
            path_to_save.mkdir(parents=True, exist_ok=True)
            
            hyperp = utils.initialize_hyperparameters(model_key, config, data.shape, default_hp, hp_ranges)
            feature_sets = utils.generate_feature_sets(features, config, data.shape)
            
            for random_seed_test in random_seeds_test:
                Path(path_to_save,f"random_seed_{int(random_seed_test)}" if config["test_size"] else "").mkdir(exist_ok=True,parents=True)
                X_dev, y_dev, IDs_dev, outputs, X_test, y_test, IDs_test = np.empty(0),np.empty(0),np.empty(0),np.empty(0),np.empty(0),np.empty(0),np.empty(0)

                if test_size > 0:
                    X_train_, X_test_, y_train_, y_test_, ID_train_, ID_test_ = train_test_split(
                        data, y, ID,
                        test_size=config["test_size"],
                        random_state=int(random_seed_test),
                        shuffle=True,
                        stratify=y if (config["stratify"] and problem_type == "clf") else None
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
                
                # If shuffling is requested, perform label shuffling before training.
                for rss, random_seed_shuffle in enumerate(config["random_seeds_shuffle"]):
                    if config["shuffle_labels"]:
                        np.random.seed(int(random_seed_shuffle))
                        #For binary classification, swap half of the labels.
                        if problem_type == "clf" and len(np.unique(y_train_)) == 2:
                            zero_indices = np.where(y_train_ == 0)[0]
                            one_indices = np.where(y_train_ == 1)[0]
                            zero_to_flip = np.random.choice(zero_indices, size=len(zero_indices) // 2, replace=False)
                            one_to_flip = np.random.choice(one_indices, size=len(one_indices) // 2, replace=False)
                            y_train_.iloc[zero_to_flip] = 1
                            y_train_.iloc[one_to_flip] = 0
                        else:
                            y_train_ = pd.Series(np.random.permutation(y_train_.values))

                        all_models = pd.read_csv(Path(str(Path(path_to_save,f"random_seed_{int(random_seed_test)}" if config["test_size"] else "")).replace("shuffle",""), f"all_models_{model_key}.csv"))

                        if config["shuffle_all"]:
                            feature_names = [col for col in all_models.columns if any(x in col for x in dimension.split("__"))]
                            param_names = list(set(all_models.columns) - set(feature_names) - set(["threshold"]))
                            hyperp = all_models[param_names]
                            feature_sets = []
                            for r,row in all_models.iterrows():
                                feature_sets.append([col for col in all_models.columns if col in feature_names and row[col] == 1])
                            
                            #Drop repeated feature sets
                            feature_sets = [list(x) for x in set(tuple(x) for x in feature_sets)]
                            hyperp = hyperp.drop_duplicates()     
                        else:
                            best_models_file_name = Path(results_dir,f"best_models_{scoring_metrics}_{config['kfold_folder']}_{config['scaler_name']}__hyp_opt_feature_selection.csv")
                            if config["n_iter"] == 0:
                                best_models_file_name = Path(str(best_models_file_name).replace("hyp_opt","no_hyp_opt"))
                            if config["n_iter_features"] == 0:
                                best_models_file_name = Path(str(best_models_file_name).replace("_feature_selection",""))
                            
                            best_models = pd.read_csv(best_models_file_name)
                            best_models = best_models[best_models["y_label"] == y_label]
                            best_models = best_models[best_models["task"] == task]
                            best_models = best_models[best_models["dimension"] == dimension]

                            model_type = best_models["model_type"].values[0]
                            model_index = best_models["model_index"].values[0]

                            if model_type != model_key:
                                continue

                            feature_names = [col for col in all_models.columns if any(x in col for x in task.split("__"))]
                            param_names = list(set([col for col in all_models.columns if col not in feature_names]) - set(["threshold"]))
                            hyperp = pd.DataFrame(all_models.loc[model_index][param_names]).T
                            feature_sets = [[col for col in all_models.columns if col in feature_names and all_models.loc[model_index][col] == 1]]

                    # Check for data leakage.
                    assert set(ID_train_).isdisjoint(set(ID_test_)), "Data leakage detected between train and test sets!"
                    
                    # Save configuration.
                    with open(Path(__file__).parent/"config.json", "w") as f:
                        json.dump(config, f, indent=4)
                    
                    if Path(path_to_save,f"random_seed_{int(random_seed_test)}" if config["test_size"] else "", f"all_models_{model_key}.csv").exists():
                        print(f"Results already exist for {task} - {y_label} - {model_key}. Skipping...")
                        continue
                    
                    print(f"Training model: {model_key}")

                    # Call CVT from utils to perform cross-validation training and tuning.
                    all_models, outputs_, y_dev_, IDs_dev_ = utils.CVT(
                        model=model_class,
                        scaler=(StandardScaler if config["scaler_name"] == "StandardScaler" else MinMaxScaler),
                        imputer=KNNImputer,
                        X=X_train_,
                        y=y_train_,
                        iterator=CV_type,
                        random_seeds_train=config["random_seeds_train"],
                        hyperp=hyperp,
                        feature_sets=feature_sets,
                        IDs=ID_train_,
                        thresholds=thresholds,
                        cmatrix=cmatrix,
                        parallel=parallel,
                        problem_type=problem_type
                    )

                    if rss == 0:
                        X_dev = np.empty((len(config["random_seeds_shuffle"]),int(config["n_seeds_train"]),X_train_.shape[0],X_train_.shape[1]))
                        y_dev = np.empty((len(config["random_seeds_shuffle"]),int(config["n_seeds_train"]),y_train_.shape[0]))
                        IDs_dev = np.empty((len(config["random_seeds_shuffle"]),int(config["n_seeds_train"]),ID_train_.shape[0]),dtype=object)
                        outputs = np.empty((len(config["random_seeds_shuffle"]),)+ outputs_.shape)
                        X_test = np.empty((len(config["random_seeds_shuffle"]),int(config["n_seeds_test"]),X_test_.shape[0],X_test_.shape[1]))
                        y_test = np.empty((len(config["random_seeds_shuffle"]),int(config["n_seeds_test"]),y_test_.shape[0]))
                        IDs_test = np.empty((len(config["random_seeds_shuffle"]),int(config["n_seeds_test"]),ID_test_.shape[0]),dtype=object)

                    X_dev[rss] = X_train_
                    y_dev[rss] = y_dev_
                    IDs_dev[rss] = IDs_dev_
                    outputs[rss] = outputs_
                    X_test[rss] = X_test_
                    y_test[rss] = y_test_
                    IDs_test[rss] = ID_test_

                    # Save results.
                all_models.to_csv(Path(path_to_save,f"random_seed_{int(random_seed_test)}" if config["test_size"] else "", f"all_models_{model_key}.csv"),index=False)

                if outputs.shape[0] == 0:
                    continue

                result_files = {
                    "X_dev.pkl": X_dev,
                    "y_dev.pkl": y_dev,
                    "IDs_dev.pkl": IDs_dev,
                    f"outputs_{model_key}.pkl": outputs,
                }
                if test_size > 0:
                    result_files.update({
                        "X_test.pkl": X_test_,
                        "y_test.pkl": y_test_,
                        "IDs_test.pkl": ID_test_,
                    })
                for fname, obj in result_files.items():
                    with open(Path(path_to_save,f"random_seed_{int(random_seed_test)}" if config["test_size"] else "", fname), "wb") as f:
                        pickle.dump(obj, f)
                
                with open(Path(path_to_save,f"random_seed_{int(random_seed_test)}" if config["test_size"] else "", "config.json"), "w") as f:
                    json.dump(config, f, indent=4)
                logging.info(f"Results saved to {path_to_save}")

##----------------------------------------------------------------------------##