import numpy as np
from MLstatkit.stats import Delong_test

import pandas as pd
from pathlib import Path
import itertools,pickle
import sys,json,os
from scipy.stats import chi2

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

planned_comparisons = [['Animales___properties','brain___norm_brain_lit'],
                       ['Animales___properties','cog___neuropsico_digits']
                       ]

config = json.load(Path(Path(__file__).parent,"config.json").open())

project_name = config["project_name"]
scaler_name = config["scaler_name"]
kfold_folder = config["kfold_folder"]
shuffle_labels = config["shuffle_labels"]
calibrate = config["calibrate"]
stat_folder = config["stat_folder"]
hyp_opt = True if config["n_iter"] > 0 else False
feature_selection = config["feature_selection"]
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
    extremo = 1 if 'norm' in scoring else 0
    ascending = True if extremo == 1 else False

    filename = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_{config["bootstrap_method"]}_hyp_opt_feature_selection_shuffled_calibrated.csv'.replace('__','_')
    if not hyp_opt:
        filename = filename.replace('_hyp_opt','')
    if not feature_selection:
        filename = filename.replace('_feature_selection','')
    if not shuffle_labels:
        filename = filename.replace('_shuffled','')
    if not calibrate:
        filename = filename.replace('_calibrated','')

    best_models = pd.read_csv(Path(results_dir,filename))

    scoring_col = f'{scoring}_extremo'
    best_models[scoring_col] = best_models[scoring].apply(lambda x: x.split('(')[1].replace(')','').split(', ')[extremo])
    best_models = best_models[best_models[scoring_col].astype(str) != 'nan'].reset_index(drop=True)
    
    best_models = best_models.sort_values(by=scoring_col,ascending=ascending).reset_index(drop=True)

    stats = pd.DataFrame(columns=['comparison','z','p_value'])
    y_labels = best_models['y_label'].unique()
    
    for y_label,comparison in itertools.product(y_labels,planned_comparisons):
        model1 = comparison[0]
        model2 = comparison[1]

        task1 = model1.split('___')[0]
        dimension1 = model1.split('___')[1]

        task2 = model2.split('___')[0]
        dimension2 = model2.split('___')[1]

        model_name1 = best_models[(best_models['task'] == task1) & (best_models['dimension'] == dimension1)]['model_type'].values[0]
        model_name2 = best_models[(best_models['task'] == task2) & (best_models['dimension'] == dimension2)]['model_type'].values[0]

        if 'model_index' in best_models:
            model_index1 = best_models[(best_models['task'] == task1) & (best_models['dimension'] == dimension1)]['model_index'].values[0]
            model_index2 = best_models[(best_models['task'] == task2) & (best_models['dimension'] == dimension2)]['model_index'].values[0]

            bayes = False
        else:
            bayes = True

        outputs_1 = pickle.load(open(Path(results_dir,task1,dimension1,scaler_name,kfold_folder,y_label,'bayes' if bayes else '',scoring if bayes else '','hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',f'outputs_{model_name1}_calibrated.pkl' if calibrate else f'outputs_{model_name1}.pkl'),'rb'))
        outputs_2 = pickle.load(open(Path(results_dir,task2,dimension2,scaler_name,kfold_folder,y_label,'bayes' if bayes else '',scoring if bayes else '','hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',f'outputs_{model_name2}_calibrated.pkl' if calibrate else f'outputs_{model_name2}.pkl'),'rb'))

        if not bayes:
            outputs_1 = outputs_1[:,model_index1]
            outputs_2 = outputs_2[:,model_index2]

        y_true = pickle.load(open(Path(results_dir,task1,dimension1,scaler_name,kfold_folder,y_label,stat_folder,'bayes' if bayes else '',scoring if bayes else '','hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',f'y_dev.pkl'),'rb'))

        while y_true.ndim < 3:
            y_true = y_true[np.newaxis,:]
        
        while outputs_1.ndim < 4:
            outputs_1 = outputs_1[np.newaxis,:]
            outputs_2 = outputs_2[np.newaxis,:]

        z_score, p_value = Delong_test(y_true.ravel(), 
                                       outputs_1.reshape(-1,outputs_1.shape[-1])[:,1],
                                       outputs_2.reshape(-1,outputs_2.shape[-1])[:,1])                                       
        
        stats_append = {'comparison':f'{model1} vs {model2}','z_score':np.round(z_score,4),'p_value':np.round(p_value,4)}
        
        if stats.empty:
            stats = pd.DataFrame(stats_append,index=[0])
        else:
            stats = pd.concat((stats,pd.DataFrame(stats_append,index=[0])),ignore_index=True)

stats.to_csv(Path(results_dir,'stats_comparison_delong.csv'),index=False)


