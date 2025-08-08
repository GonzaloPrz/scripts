import pandas as pd
import sys, pickle, os, json,itertools
from pathlib import Path
import numpy as np
from expected_cost.utils import plot_hists

config = json.load(Path(Path(__file__).parent,"config.json").open())

project_name = config["project_name"]
scaler_name = config["scaler_name"]
kfold_folder = config["kfold_folder"]
shuffle_labels = config["shuffle_labels"]
stat_folder = config["stat_folder"]
hyp_opt = True if config["n_iter"] > 0 else False
feature_selection = True if config["n_iter_features"] > 0 else False
filter_outliers = config["filter_outliers"]
n_boot = int(config["n_boot"])

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / "results" / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,"main_config.json").open())

scoring_metrics = main_config["scoring_metrics"][project_name]

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

for scoring in scoring_metrics:
    best_models_filename = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection_calibrated.csv'.replace('__','_')
    if not hyp_opt:
        best_models_filename = best_models_filename.replace('_hyp_opt','')
    if not feature_selection:
        best_models_filename = best_models_filename.replace('_feature_selection','')
    
    best_models = pd.read_csv(Path(results_dir,best_models_filename))
    y_labels = best_models['y_label'].unique()
    tasks = best_models['task'].unique()
    dimensions = best_models['dimension'].unique()

    for r, row in best_models.iterrows():
        task = row['task']
        dimension = row['dimension']
        y_label = row['y_label']
        random_seed = row['random_seed_test']
        model_index = row['model_index']
        model_type = row['model_type']
        
        if str(random_seed) == 'nan':
            random_seed = ''
        
        path_to_scores = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers else '',random_seed,'shuffle' if shuffle_labels else '')

        uncalibrated_scores_ = pickle.load(open(Path(path_to_scores,f'outputs_{model_type}.pkl'),'rb'))[0,model_index]
        uncalibrated_scores = np.concatenate([uncalibrated_scores_[r] for r in range(uncalibrated_scores_.shape[0])])  

        calibrated_scores_ = pickle.load(open(Path(path_to_scores,f'cal_outputs_{model_type}.pkl'),'rb'))[0,model_index]
        calibrated_scores = np.concatenate([calibrated_scores_[r] for r in range(calibrated_scores_.shape[0])])

        y_true_ = pickle.load(open(Path(path_to_scores, f"y_dev.pkl"), "rb"))
        y_true = np.concatenate([y_true_[0,r] for r in range(y_true_.shape[1])])

        Path(results_dir,"plots",task,dimension,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '').mkdir(exist_ok=True)
        
        ax = plot_hists(y_true, uncalibrated_scores, outfile=None, nbins=50, group_by='score', style='-', label_prefix='uncalibrated ', axs=None)

        filename_to_save = f"best_calibrated_uncalibrated_{model_type}_logpost.png"
        plot_hists(y_true, calibrated_scores, outfile=Path(results_dir,"plots",task,dimension,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '',filename_to_save), nbins=50, group_by='score', style='--', label_prefix='calibrated ', axs=ax)