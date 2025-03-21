import pickle,json,os,sys,itertools
import pandas as pd
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

sys.path.append(str(Path(Path.home(),"scripts_generales"))) if "Users/gp" in str(Path.home()) else sys.path.append(str(Path(Path.home(),"gonza","scripts_generales")))

import utils

tasks_list = [
              ['Animales__P','cog'],
              ['Animales__P','brain']

]
dimensions_list = [
                   ['properties','neuropsico_digits__neuropsico_tmt'],
                   ['properties','norm_brain_lit']
]

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
n_models = int(config["n_models"])
n_boot = int(config["n_boot"])
early_fusion = bool(config["early_fusion"])
problem_type = config["problem_type"]

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
scoring_metrics = [main_config['scoring_metrics'][project_name]]
problem_type = main_config['problem_type'][project_name]
metrics_names = main_config["metrics_names"][main_config["problem_type"][project_name]]

diff_ci = pd.DataFrame()

for scoring in scoring_metrics:

    best_models_file = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection_shuffled.csv'.replace('__','_')
    if not hyp_opt:
        best_models_file = best_models_file.replace('_hyp_opt','_no_hyp_opt')
    if not feature_selection:
        best_models_file = best_models_file.replace('_feature_selection','')
    if not shuffle_labels:
        best_models_file = best_models_file.replace('_shuffled','')

    best_models = pd.read_csv(Path(results_dir,best_models_file))
    for tasks, dimensions in zip(tasks_list, dimensions_list):

        for y_label in y_labels:
            best_models_task1 = best_models[(best_models.task == tasks[0]) & (best_models.dimension == dimensions[0]) & (best_models.y_label == y_label)]
            best_models_task2 = best_models[(best_models.task == tasks[1]) & (best_models.dimension == dimensions[1]) & (best_models.y_label == y_label)]

            if best_models_task1.empty or best_models_task2.empty:
                continue

            model_type_1 = best_models_task1.model_type.values[0]
            model_type_2 = best_models_task2.model_type.values[0]

            model_index_1 = best_models_task1.model_index.values[0]
            model_index_2 = best_models_task2.model_index.values[0]

            outputs1 = pickle.load(open(Path(results_dir,tasks[0], dimensions[0], scaler_name, kfold_folder,
                                            y_label, "hyp_opt" if hyp_opt else "no_hyp_opt",
                                            "feature_selection" if feature_selection else "", 
                                            "filter_outliers" if filter_outliers and problem_type == 'reg' else '',f'outputs_{model_type_1}.pkl'),"rb"))[:,model_index_1,...]

            outputs1 = outputs1[:,np.newaxis,...]

            outputs2 = pickle.load(open(Path(results_dir,tasks[1], dimensions[1], scaler_name, kfold_folder,
                                            y_label, "hyp_opt" if hyp_opt else "no_hyp_opt",
                                            "feature_selection" if feature_selection else "", 
                                            "filter_outliers" if filter_outliers and problem_type == 'reg' else '',f'outputs_{model_type_2}.pkl'),"rb"))[:,model_index_2,...]

            outputs2 = outputs2[:,np.newaxis,...]

            y_dev1 = pickle.load(open(Path(results_dir,tasks[0], dimensions[0], scaler_name, kfold_folder,
                                            y_label, "hyp_opt" if hyp_opt else "no_hyp_opt",
                                            "feature_selection" if feature_selection else "", 
                                            "filter_outliers" if filter_outliers and problem_type == 'reg' else '',f'y_dev.pkl'),"rb"))
            
            IDs1 = pickle.load(open(Path(results_dir,tasks[0], dimensions[0], scaler_name, kfold_folder,
                                            y_label, "hyp_opt" if hyp_opt else "no_hyp_opt",
                                            "feature_selection" if feature_selection else "", 
                                            "filter_outliers" if filter_outliers and problem_type == 'reg' else '','IDs_dev.pkl'),"rb"))

            y_dev2 = pickle.load(open(Path(results_dir,tasks[1], dimensions[1], scaler_name, kfold_folder,
                                            y_label, "hyp_opt" if hyp_opt else "no_hyp_opt",
                                            "feature_selection" if feature_selection else "", 
                                            "filter_outliers" if filter_outliers and problem_type == 'reg' else '',f'y_dev.pkl'),"rb"))
            
            IDs2 = pickle.load(open(Path(results_dir,tasks[1], dimensions[1], scaler_name, kfold_folder,
                                            y_label, "hyp_opt" if hyp_opt else "no_hyp_opt",
                                            "feature_selection" if feature_selection else "", 
                                            "filter_outliers" if filter_outliers and problem_type == 'reg' else '',f'IDs_dev.pkl'),"rb"))
            
            results1 = Parallel(n_jobs=-1)(delayed(utils.compute_metrics)(j,model_index, r, outputs1, y_dev1, IDs1, metrics_names, n_boot, problem_type, cmatrix=None, priors=None, threshold=None) for j,model_index, r in itertools.product(range(outputs1.shape[0]),range(outputs1.shape[1]),range(outputs1.shape[2])))
            results2 = Parallel(n_jobs=-1)(delayed(utils.compute_metrics)(j,model_index, r, outputs2, y_dev2, IDs2, metrics_names, n_boot, problem_type, cmatrix=None, priors=None, threshold=None) for j,model_index, r in itertools.product(range(outputs2.shape[0]),range(outputs2.shape[1]),range(outputs2.shape[2])))
            
            metrics = dict((metric, np.empty((outputs1.shape[0], outputs1.shape[1], outputs1.shape[2], n_boot))) for metric in metrics_names)
            metrics_shuffle = dict((metric, np.empty((outputs2.shape[0], outputs2.shape[1], outputs2.shape[2], n_boot))) for metric in metrics_names)
            metrics_diff = dict((metric, np.empty((outputs1.shape[0], outputs1.shape[1], outputs1.shape[2], n_boot))) for metric in metrics_names)

            for metric in metrics_names:
                for j, model_index, r, metrics_result, sorted_IDs1 in results1:
                    metrics[metric][j, model_index, r, :] = metrics_result[metric]
                for j, model_index, r, metrics_result, sorted_IDs2 in results2:
                    metrics_shuffle[metric][j, model_index, r, :] = metrics_result[metric]

                metrics[metric] = metrics[metric].flatten()
                metrics_shuffle[metric] = metrics_shuffle[metric].flatten()
                metrics_diff[metric] = metrics[metric] - metrics_shuffle[metric]
                
                if diff_ci.empty:
                    diff_ci = pd.DataFrame({"tasks": f'[{tasks[0]}, {tasks[1]}]', "dimensions": f'[{dimensions[0]}, {dimensions[1]}]', "y_label": y_label, "metric": metric, "mean": np.nanmean(metrics_diff[metric]), "ci_low": np.nanpercentile(metrics_diff[metric], 2.5), "ci_high": np.nanpercentile(metrics_diff[metric], 97.5)}, index=[0])
                else:
                    diff_ci = pd.concat((diff_ci,pd.DataFrame({"tasks": f'[{tasks[0]}, {tasks[1]}]', "dimensions": f'[{dimensions[0]}, {dimensions[1]}]', "y_label": y_label, "metric": metric, "mean": np.nanmean(metrics_diff[metric]), "ci_low": np.nanpercentile(metrics_diff[metric], 2.5), "ci_high": np.nanpercentile(metrics_diff[metric], 97.5)}, index=[0])))

    diff_ci.to_csv(Path(results_dir,f'ic_diff_{scoring}.csv'),index=False)