import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import pandas as pd
from pathlib import Path
import itertools,pickle
import sys,json,os
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import LeaveOneOut as LOOCV

def delong_auc_test(y_true, y_pred_1, y_pred_2):
    """
    Perform DeLong's test to compare AUCs of two models.

    Parameters:
        y_true (array): True binary labels (0 or 1)
        y_pred_1 (array): Predicted probabilities from Model 1
        y_pred_2 (array): Predicted probabilities from Model 2
    
    Returns:
        p-value: Significance level of the AUC difference
    """
    def compute_auc_var(y_true, y_scores):
        """ Compute AUC variance using DeLong method """
        n = len(y_true)
        auc = roc_auc_score(y_true, y_scores[:,1])
        pos = y_scores[y_true == 1,1]
        neg = y_scores[y_true == 0,1]
        n_pos, n_neg = len(pos), len(neg)
        v = np.var([np.mean(pos > neg[j]) for j in range(n_neg)])
        return auc, v / n

    # Compute AUC and variance for both models
    auc1, var1 = compute_auc_var(y_true, y_pred_1)
    auc2, var2 = compute_auc_var(y_true, y_pred_2)

    # Compute Z-score
    se = np.sqrt(var1 + var2)
    z_score = (auc1 - auc2) / se
    p_value = 2 * norm.sf(abs(z_score))  # Two-tailed test

    return z_score, p_value

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

            z_score,p_value = delong_auc_test(y_true,y_scores,y_scores_shuffle)
            z_score = np.round(z_score,3)
            p_value = np.round(p_value,3)

            stats_append = {'comparison':f'{task} - {dimension}','z_score':z_score,'p_value':p_value}
            
            if stats.empty:
                stats = pd.DataFrame(stats_append,index=[0])
            else:
                stats = pd.concat((stats,pd.DataFrame(stats_append,index=[0])),ignore_index=True)
        except:
            print(f"Error in DeLong's test for {task} - {dimension}.")
            continue
stats.to_csv(Path(results_dir,'stats_comparison_delong_shuffle.csv'),index=False)