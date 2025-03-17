import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings, json, os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.append(str(Path(Path.home(),"scripts_generales"))) if "Users/gp" in str(Path.home()) else sys.path.append(str(Path(Path.home(),"gonza","scripts_generales")))

import utils

warnings.filterwarnings("ignore")

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

y_labels = main_config["y_labels"][project_name]
tasks = main_config["tasks"][project_name]
test_size = main_config["test_size"][project_name]
single_dimensions = main_config["single_dimensions"][project_name]
thresholds = main_config["thresholds"][project_name]
scoring_metrics = main_config["scoring_metrics"][project_name]
metrics_names = main_config["metrics_names"][main_config["problem_type"][project_name]]

diff_ci = pd.DataFrame(columns=["task", "dimension", "y_label", "metric", "mean", "ci_low", "ci_high"])

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

problem_type = main_config["problem_type"][project_name]

# Set the style for the plots
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
    "figure.titlesize": 16
})

diff_ci = pd.DataFrame()

for scoring in scoring_metrics:
    extremo = "sup" if "norm" in scoring else "inf"
    ascending = True if extremo == "sup" else False

    best_models = pd.read_csv(Path(results_dir,f"best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection.csv".replace("__","_")))

    for r, row in best_models.iterrows():   
        print(row["task"], row["dimension"])
        path_to_results = Path(results_dir, row.task, row.dimension, scaler_name, kfold_folder, row.y_label, stat_folder,"hyp_opt" if hyp_opt else "", "feature_selection" if feature_selection else "", 'filter_outliers' if filter_outliers and problem_type == 'reg' else '')

        model_name = row.model_type
        if str(row.random_seed_test) == "nan":
            random_seed = ""
        else:
            random_seed = row.random_seed_test

        try:
            model_index = pd.read_csv(Path(path_to_results, random_seed, f"all_models_{model_name}_dev_bca.csv")).sort_values(f"{scoring}_{extremo}", ascending=ascending).index[0]
            threshold = pd.read_csv(Path(path_to_results, random_seed, f"all_models_{model_name}_dev_bca.csv")).sort_values(f"{scoring}_{extremo}", ascending=ascending)['threshold'][0]
        except:
            model_index = pd.read_csv(Path(path_to_results, random_seed, f"best_models_{scoring}_{model_name}_dev_bca.csv")).sort_values(f"{scoring}_{extremo}", ascending=ascending).index[0]
            threshold = pd.read_csv(Path(path_to_results, random_seed, f"best_models_{scoring}_{model_name}_dev_bca.csv")).sort_values(f"{scoring}_{extremo}", ascending=ascending)['threshold'][0]

        if Path(path_to_results, 'shuffle', random_seed, f"all_models_{model_name}_dev_bca.csv").exists():

            model_index_shuffle = pd.read_csv(Path(path_to_results, 'shuffle', random_seed, f"all_models_{model_name}_dev_bca.csv")).sort_values(f"{scoring}_{extremo}", ascending=ascending).index[0]
            threshold = pd.read_csv(Path(path_to_results, 'shuffle', random_seed, f"all_models_{model_name}_dev_bca.csv")).sort_values(f"{scoring}_{extremo}", ascending=ascending)['threshold'][0]
        elif Path(path_to_results, 'shuffle', random_seed, f"best_models_{model_name}_dev_bca.csv").exists():
            model_index_shuffle = pd.read_csv(Path(path_to_results, 'shuffle', random_seed, f"best_models_{scoring}_{model_name}_dev_bca.csv")).sort_values(f"{scoring}_{extremo}", ascending=ascending).index[0]
            threshold = pd.read_csv(Path(path_to_results, 'shuffle', random_seed, f"best_models_{scoring}_{model_name}_dev_bca.csv")).sort_values(f"{scoring}_{extremo}", ascending=ascending)['threshold'][0]
        else:
            pass
        
        outputs_ = pickle.load(open(Path(path_to_results, random_seed, f"outputs_{model_name}.pkl"), "rb"))[:,model_index]

        #Add missing dimensions: model_index, j
        outputs_ =outputs_[:,np.newaxis, ...]
        
        y_true_ = pickle.load(open(Path(path_to_results,random_seed, f"y_dev.pkl"), "rb"))
        IDs_ = pickle.load(open(Path(path_to_results,random_seed, f"IDs_dev.pkl"), "rb"))
        results = Parallel(n_jobs=1)(delayed(utils.compute_metrics)(j,model_index, r, outputs_, y_true_,IDs_,metrics_names, n_boot, problem_type, cmatrix=None, priors=None, threshold=threshold) for j,model_index, r in itertools.product(range(outputs_.shape[0]),range(outputs_.shape[1]),range(outputs_.shape[2])))
        metrics = dict((metric, np.empty((outputs_.shape[0], outputs_.shape[1], outputs_.shape[2], n_boot))) for metric in metrics_names)

        for metric in metrics_names:
            for j, model_index, r, metrics_result, IDs in results:
                metrics[metric][j, model_index, r, :] = metrics_result[metric]
            metrics[metric] = metrics[metric].flatten()

        try:
            outputs_shuffle = pickle.load(open(Path(path_to_results,'shuffle',random_seed, f"outputs_{model_name}.pkl"), "rb"))[:,model_index_shuffle, :, :, :]
            #Expand dimensions to match outputs_

            outputs_shuffle = outputs_shuffle[:,np.newaxis, ...]
            y_true_shuffle = pickle.load(open(Path(path_to_results, 'shuffle',random_seed, f"y_dev.pkl"), "rb"))
            IDs_shuffle = pickle.load(open(Path(path_to_results, 'shuffle',random_seed, f"IDs_dev.pkl"), "rb"))

            results_shuffle = Parallel(n_jobs=1)(delayed(utils.compute_metrics)(j,model_index, r, outputs_shuffle, y_true_shuffle,IDs_shuffle,metrics_names, n_boot, problem_type, cmatrix=None, priors=None, threshold=threshold) for j,model_index, r in itertools.product(range(outputs_shuffle.shape[0]),range(outputs_shuffle.shape[1]),range(outputs_shuffle.shape[2])))
            metrics_shuffle = dict((metric, np.empty((outputs_shuffle.shape[0], outputs_shuffle.shape[1], outputs_shuffle.shape[2], n_boot))) for metric in metrics_names)
            
            for metric in metrics_names:
                for j, model_index, r, metrics_result, IDs_shuffle_ in results_shuffle:
                    metrics_shuffle[metric][j, model_index, r, :] = metrics_result[metric]
                metrics_shuffle[metric] = metrics_shuffle[metric].flatten()

            metrics_diff = dict((metric, np.empty((outputs_.shape[0], outputs_.shape[1], outputs_.shape[2], n_boot))) for metric in metrics_names)
        
                #Concatenate metrics as many times as necessary to match the length of metrics_shuffle
            if len(metrics[metric]) < len(metrics_shuffle[metric]):
                metrics[metric] = np.concatenate([metrics[metric] for _ in range(len(metrics_shuffle[metric]) // len(metrics[metric]))])

            metrics_diff[metric] = metrics[metric] - metrics_shuffle[metric]
            if diff_ci.empty:
                diff_ci = pd.DataFrame({"task": row.task, "dimension": row.dimension, "y_label": row.y_label, "metric": metric, "mean": np.nanmean(metrics_diff[metric]), "ci_low": np.nanpercentile(metrics_diff[metric], 2.5), "ci_high": np.nanpercentile(metrics_diff[metric], 97.5)}, index=[0])
            else:
                diff_ci = pd.concat((diff_ci,pd.DataFrame({"task": row.task, "dimension": row.dimension, "y_label": row.y_label, "metric": metric, "mean": np.nanmean(metrics_diff[metric]), "ci_low": np.nanpercentile(metrics_diff[metric], 2.5), "ci_high": np.nanpercentile(metrics_diff[metric], 97.5)}, index=[0])))
        except:
            pass

        y_true = np.concatenate([y_true_[0,r,:] for r in range(y_true_.shape[1])])
        outputs = np.concatenate([outputs_[0,0,r,:,1] for r in range(outputs_.shape[2])]).flatten()
        
        Path(results_dir,"plots").mkdir(parents=True, exist_ok=True)

        plt.figure()

        h, e = np.histogram(np.log(np.exp(outputs[y_true == 0]) / (1 - np.exp(outputs[y_true == 0]))), bins=50, density=True)
        centers = (e[:-1] + e[1:]) / 2
        plt.plot(centers, h, label="Class 0")
        for cl in set(np.unique(y_true)) - {0}:
            h, e = np.histogram(np.log(np.exp(outputs[y_true == cl]) / (1 - np.exp(outputs[y_true == cl]))), bins=50, density=True)
            centers = (e[:-1] + e[1:]) / 2
            plt.plot(centers, h, label=f"Class {cl}")

        plt.xlabel("Log-odds")
        plt.ylabel("Density")
        plt.title(f"Log-odds Distribution for {model_name}")
        plt.tight_layout()
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(results_dir,"plots", f"best_{row.task}_{row.dimension}_{model_name}_log_odds.png"))
        
        try:
            for metric in metrics_names:
                plt.figure()
                sns.violinplot(data=[metrics[metric], metrics_shuffle[metric]], inner=None, palette="muted")
                plt.xticks([0, 1], ["Real", "Shuffle"])
                plt.ylabel(metric.replace("_", " ").upper())
                plt.title(f"{metric.replace('_', ' ').upper()} Distribution for {model_name}")
                plt.tight_layout()
                plt.ylim(0, 1)
                plt.grid(True)
                plt.savefig(Path(results_dir,"plots", f"best_{row.task}_{row.dimension}_{model_name}_{metric}.png"))
                plt.savefig(Path(results_dir,"plots", f"best_{row.task}_{row.dimension}_{model_name}_{metric}.svg"))
                plt.close()
        except:
            pass
    try:
        diff_ci.to_csv(Path(results_dir,f"diff_ci_shuffle_{scoring}.csv"), index=False)
        print("Done!")
    except:
        pass