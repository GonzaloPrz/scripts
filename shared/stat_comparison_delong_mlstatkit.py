import numpy as np
from MLstatkit.stats import Delong_test

import pandas as pd
from pathlib import Path
import itertools,pickle
import sys,json,os
from scipy.stats import chi2

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

planned_comparisons = [['Animales__P___properties','brain___norm_brain_lit'],
                       ['Animales__P___properties','cog___neuropsico_digits__neuropsico_tmt']
                       ]

config = json.load(Path(Path(__file__).parent,"config.json").open())

project_name = config["project_name"]
scaler_name = config["scaler_name"]
kfold_folder = config["kfold_folder"]
shuffle_labels = config["shuffle_labels"]
calibrate = config["calibrate"]
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
metrics_names = main_config["metrics_names"][main_config["problem_type"][project_name]]

if not isinstance(scoring_metrics,list):
    scoring_metrics = [scoring_metrics]

for scoring in scoring_metrics:
    best_classifiers = pd.read_csv(Path(results_dir,f'best_models_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection.csv'))

    stats = pd.DataFrame(columns=['comparison','z','p_value'])
    y_labels = best_classifiers['y_label'].unique()
    
    for y_label,comparison in itertools.product(y_labels,planned_comparisons):
        model1 = comparison[0]
        model2 = comparison[1]

        task1 = model1.split('___')[0]
        dimension1 = model1.split('___')[1]

        task2 = model2.split('___')[0]
        dimension2 = model2.split('___')[1]

        model_name1 = best_classifiers[(best_classifiers['task'] == task1) & (best_classifiers['dimension'] == dimension1)]['model_type'].values[0]
        model_name2 = best_classifiers[(best_classifiers['task'] == task2) & (best_classifiers['dimension'] == dimension2)]['model_type'].values[0]

        model_index1 = best_classifiers[(best_classifiers['task'] == task1) & (best_classifiers['dimension'] == dimension1)]['model_index'].values[0]
        model_index2 = best_classifiers[(best_classifiers['task'] == task2) & (best_classifiers['dimension'] == dimension2)]['model_index'].values[0]

        outputs_1 = pickle.load(open(Path(results_dir,task1,dimension1,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection',f'outputs_{model_name1}_calibrated.pkl' if calibrate else f'outputs_{model_name1}.pkl'),'rb'))[:,model_index1]
        outputs_2 = pickle.load(open(Path(results_dir,task2,dimension2,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection',f'outputs_{model_name2}_calibrated.pkl' if calibrate else f'outputs_{model_name2}.pkl'),'rb'))[:,model_index2]

        y_true = pickle.load(open(Path(results_dir,task1,dimension1,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt','feature_selection',f'y_dev.pkl'),'rb'))

        z_score, p_value = Delong_test(np.concatenate([y_true[j,r] for j,r in itertools.product(range(y_true.shape[0]),range(y_true.shape[1]))]), 
                                       np.concatenate([outputs_1[j,r] for j,r in itertools.product(range(outputs_1.shape[0]),range(outputs_1.shape[1]))])[:,1], 
                                       np.concatenate([outputs_2[j,r] for j,r in itertools.product(range(outputs_2.shape[0]),range(outputs_2.shape[1]))])[:,1])
        
        stats_append = {'comparison':f'{model1} vs {model2}','z_score':np.round(z_score,3),'p_value':np.round(p_value,3)}
        
        if stats.empty:
            stats = pd.DataFrame(stats_append,index=[0])
        else:
            stats = pd.concat((stats,pd.DataFrame(stats_append,index=[0])),ignore_index=True)

stats.to_csv(Path(results_dir,'stats_comparison_delong.csv'),index=False)


