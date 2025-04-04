import numpy as np

import pandas as pd
from pathlib import Path
import itertools,pickle
import sys,json,os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import LeaveOneOut as LOOCV

from scipy.stats import chi2_contingency

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

config = json.load(Path(Path(__file__).parent,"config.json").open())

project_name = config["project_name"]
scaler_name = config["scaler_name"]
kfold_folder = config["kfold_folder"]
n_folds = int(config["n_folds"])
shuffle_labels = config["shuffle_labels"]
calibrate = config["calibrate"]
stat_folder = config["stat_folder"]
hyp_opt = True if config["n_iter"] > 0 else False
feature_selection = True if config["n_iter_features"] > 0 else False
filter_outliers = config["filter_outliers"]
n_boot = int(config["n_boot"])
random_seed_train = list(config["random_seeds_train"])[0]
stratify = bool(config["stratify"])
problem_type = config["problem_type"]

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / "results" / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,"main_config.json").open())

scoring_metrics = main_config["scoring_metrics"][project_name]
metrics_names = main_config["metrics_names"][main_config["problem_type"][project_name]]

if not isinstance(scoring_metrics,list):
    scoring_metrics = [scoring_metrics]

for scoring in scoring_metrics:
    best_classifiers = pd.read_csv(Path(results_dir,f'best_models_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection.csv'))
    
    stats = pd.DataFrame(columns=['comparison','z','p_value'])
    y_labels = best_classifiers['y_label'].unique()
    tasks = best_classifiers['task'].unique()
    dimensions = best_classifiers['dimension'].unique()

    for y_label,task,dimension in itertools.product(y_labels,tasks,dimensions):
        print(y_label,task,dimension)
        row = best_classifiers[(best_classifiers['task'] == task) & (best_classifiers['dimension'] == dimension)]
        if row.empty:
            continue

        model_name = row['model_type'].values[0]

        y_true = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt','feature_selection',f'y_dev.pkl'),'rb'))[0,0]
        X_dev = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt','feature_selection',f'X_dev.pkl'),'rb'))[0,0]
        
        y_true_shuffle = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt','feature_selection','shuffle',f'y_dev.pkl'),'rb'))[0,0]
        X_dev_shuffle = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt','feature_selection','shuffle',f'X_dev.pkl'),'rb'))[0,0]

        model = pickle.load(open(Path(results_dir,f'final_model_{scoring}',task,dimension,y_label,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '','final_model.pkl'),'rb'))
        scaler = pickle.load(open(Path(results_dir,f'final_model_{scoring}',task,dimension,y_label,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '','scaler.pkl'),'rb'))
        imputer = pickle.load(open(Path(results_dir,f'final_model_{scoring}',task,dimension,y_label,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '','imputer.pkl'),'rb'))

        if n_folds == -1:
            CV = LOOCV()
        elif n_folds == 0:
            n_folds = int(len(y_true)/2)
        
        CV = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=int(random_seed_train)) if stratify else KFold(n_splits=n_folds,shuffle=True,random_state=int(random_seed_train))
        
        y_scores = np.zeros((X_dev.shape[0],len(np.unique(y_true)) if problem_type=='clf' else 1)).squeeze()
        y_scores_shuffle = np.zeros((X_dev.shape[0],len(np.unique(y_true)) if problem_type=='clf' else 1)).squeeze()

        try:
            for train_index,test_index in CV.split(X_dev,y_true):
                m = utils.Model(model,type(scaler),type(imputer))

                m.train(pd.DataFrame(X_dev[train_index]),y_true[train_index])

                y_scores[test_index] = m.eval(pd.DataFrame(X_dev[test_index]),problem_type=problem_type)

                m.train(pd.DataFrame(X_dev_shuffle[train_index]),y_true_shuffle[train_index])

                y_scores_shuffle[test_index] = m.eval(pd.DataFrame(X_dev_shuffle[test_index]),problem_type=problem_type)
            
            if problem_type == 'clf':
                _, y_pred = utils.get_metrics_clf(y_scores, y_true, [])
                _, y_pred_shuffle = utils.get_metrics_clf(y_scores_shuffle, y_true_shuffle, [])
            else:
                _, y_pred = utils.get_metrics_reg(y_scores, y_true, [])
                _, y_pred_shuffle = utils.get_metrics_reg(y_scores_shuffle, y_true_shuffle, [])
            
            tb1 = pd.crosstab(y_true, y_pred)
            tb2 = pd.crosstab(y_true_shuffle, y_pred_shuffle)
            
            #Compare contingency tables
            combined_table = np.vstack([tb1, tb2])
            chi2, p_value, dof, expected = chi2_contingency(combined_table)

            stats_append = {'comparison':f'{task} vs {dimension}','stat':chi2,'p_value':p_value}
            if stats.empty:
                stats = pd.DataFrame(stats_append,index=[0])
            else:
                stats = pd.concat((stats,pd.DataFrame(stats_append,index=[0])),ignore_index=True)
        except:
            continue
stats.to_csv(Path(results_dir,'stats_comparison_mcnemar_shuffle.csv'),index=False)
