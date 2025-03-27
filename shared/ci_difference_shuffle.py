import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings, json, os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from expected_cost.utils import plot_hists
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

warnings.filterwarnings('ignore')

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config['project_name']
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
rewrite = bool(config['rewrite'])
test_size = float(config['test_size'])
n_boot = int(config['n_boot'])

home = Path(os.environ.get('HOME', Path.home()))
if 'Users/gp' in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path('D:/CNC_Audio/gonza/results', project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

scoring_metrics = main_config['scoring_metrics'][project_name]
metrics_names = main_config['metrics_names'][main_config['problem_type'][project_name]]
tasks = main_config['tasks'][project_name]
dimensions = main_config['dimensions'][project_name]
y_labels = main_config['y_labels'][project_name]

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]

# Set the style for the plots
sns.set(style='whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

if problem_type == 'clf':
    for scoring in scoring_metrics:
        extremo = 'sup' if 'norm' in scoring else 'inf'
        ascending = True if extremo == 'sup' else False

        for task,dimension,y_label in itertools.product(tasks,dimensions,y_labels):
            data_to_plot = pd.DataFrame()

            Path(results_dir,'plots',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '').mkdir(parents=True, exist_ok=True)

            print(task, dimension)
            path_to_results = Path(results_dir, task, dimension, scaler_name, kfold_folder, y_label, stat_folder,'hyp_opt' if hyp_opt else '', 'feature_selection' if feature_selection else '', 'filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '')

            best_models_filename = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection.csv'.replace('__','_')
            if not hyp_opt:
                best_models_filename = best_models_filename.replace('_hyp_opt','')
            if not feature_selection:
                best_models_filename = best_models_filename.replace('_feature_selection','')

            best_models = pd.read_csv(Path(results_dir,best_models_filename))
            
            row = best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label)]
            if row.empty or Path(path_to_results, file).exists() == False:
                continue
                
            model_name = row['model_name'].values[0]
            
            file = f'all_models_{model_name}_dev_bca.csv'
            
            if config['n_models'] != 0:
                file = file.replace('all_models', 'best_models').replace('.csv', f'_{scoring}.csv')
            
            if str(row['random_seed_test']) == 'nan':
                random_seed = ''
            else:
                random_seed = row.random_seed_test
    
            df_filename = pd.read_csv(Path(path_to_results, random_seed, file)).sort_values(f'{scoring}_{extremo}', ascending=ascending)
            model_index = df_filename.index[0]

            if 'threshold' in df_filename.columns:
                threshold = df_filename['threshold'][0]
            else:
                threshold = None
                
            df_filename_shuffle = pd.read_csv(Path(path_to_results, 'shuffle', random_seed, f'all_models_{model_name}_dev_bca.csv')).sort_values(f'{scoring}_{extremo}', ascending=ascending)
            model_index_shuffle = df_filename_shuffle.index[0]
            if 'threshold' in df_filename_shuffle.columns:
                threshold_shuffle = df_filename_shuffle['threshold'][0]
            else:
                threshold_shuffle = None
                
            outputs_filename = f'outputs_{model_name}.pkl'
            outputs_ = pickle.load(open(Path(path_to_results, random_seed, outputs_filename), 'rb'))[:,model_index]
            y_true_ = pickle.load(open(Path(path_to_results,random_seed, f'y_dev.pkl'), 'rb'))
            IDs_ = pickle.load(open(Path(path_to_results,random_seed, f'IDs_dev.pkl'), 'rb'))

            results = Parallel(n_jobs=1)(delayed(utils.compute_metrics)(j,model_index, r, outputs_, y_true_,IDs_,metrics_names, n_boot, problem_type, cmatrix=None, priors=None, threshold=threshold) for j,model_index, r in itertools.product(range(outputs_.shape[0]),range(outputs_.shape[1]),range(outputs_.shape[2])))
            metrics = dict((metric, np.empty((outputs_.shape[0], outputs_.shape[1], outputs_.shape[2], n_boot))) for metric in metrics_names)
            metrics_diff = dict((metric, np.empty((outputs_.shape[0], outputs_.shape[1], outputs_.shape[2], n_boot))) for metric in metrics_names)
            
            #Check whether IDs and IDs_shuffle are the same

            for metric in metrics_names:
                for j, model_index, r, metrics_result, IDs in results:
                    metrics[metric][j, model_index, r, :] = metrics_result[metric]
                metrics[metric] = metrics[metric].flatten()

                data_to_plot[metric] = metrics[metric]
 
            outputs_shuffle = pickle.load(open(Path(path_to_results,'shuffle',random_seed, outputs_filename), 'rb'))[:,model_index_shuffle, :, :, :]
            #Expand dimensions to match outputs_

            outputs_shuffle = outputs_shuffle[:,np.newaxis, ...]
            y_true_shuffle = pickle.load(open(Path(path_to_results, 'shuffle',random_seed, f'y_dev.pkl'), 'rb'))
            IDs_shuffle = pickle.load(open(Path(path_to_results, 'shuffle',random_seed, f'IDs_dev.pkl'), 'rb'))

            results_shuffle = Parallel(n_jobs=1)(delayed(utils.compute_metrics)(j,model_index, r, outputs_shuffle, y_true_shuffle,IDs_shuffle,metrics_names, n_boot, problem_type, cmatrix=None, priors=None, threshold=threshold) for j,model_index, r in itertools.product(range(outputs_shuffle.shape[0]),range(outputs_shuffle.shape[1]),range(outputs_shuffle.shape[2])))

            metrics_shuffle = dict((metric, np.empty((outputs_shuffle.shape[0], outputs_shuffle.shape[1], outputs_shuffle.shape[2], n_boot))) for metric in metrics_names)
            
            #Check whether IDs and IDs_shuffle are the same

            for metric in metrics_names:
                for j, model_index, r, metrics_result, IDs_shuffle_ in results_shuffle:
                    metrics_shuffle[metric][j, model_index, r, :] = metrics_result[metric]
                metrics_shuffle[metric] = metrics_shuffle[metric].flatten()

                data_to_plot[f'{metric}_shuffled'] = metrics_shuffle[metric]
                    #Concatenate metrics as many times as necessary to match the length of metrics_shuffle
                if len(metrics[metric]) < len(metrics_shuffle[metric]):
                    metrics[metric] = np.concatenate([metrics[metric] for _ in range(len(metrics_shuffle[metric]) // len(metrics[metric]))])

                metrics_diff[metric] = metrics[metric] - metrics_shuffle[metric]

                wilc, p_value = wilcoxon(metrics[metric], metrics_shuffle[metric])
                non_param_stat_to_append = {'task': task, 'dimension': dimension, 'y_label': y_label, 'metric': metric, 'mean': np.nanmean(metrics_diff[metric]), 'ci_low': np.nanpercentile(metrics_diff[metric], 2.5), 'ci_high': np.nanpercentile(metrics_diff[metric], 97.5), 'wilcoxon_stat': wilc, 'p_value': p_value}
                
                if nonparam_stats.empty:
                    nonparam_stats = pd.DataFrame(non_param_stat_to_append, index=[0])
                else:
                    nonparam_stats = pd.concat((nonparam_stats,pd.DataFrame(non_param_stat_to_append, index=[0])))
                                            
                t_stat, p_value = ttest_rel(metrics[metric], metrics_shuffle[metric])
                param_stat_to_append = {'task': task, 'dimension': dimension, 'y_label': y_label, 'metric': metric, 'mean': np.nanmean(metrics_diff[metric]), 'ci_low': np.nanpercentile(metrics_diff[metric], 2.5), 'ci_high': np.nanpercentile(metrics_diff[metric], 97.5), 't_stat': t_stat, 'p_value': p_value}

                if param_stats.empty:
                    param_stats = pd.DataFrame(param_stat_to_append, index=[0])
                else:
                    param_stats = pd.concat((param_stats,pd.DataFrame(param_stat_to_append, index=[0])))
                
                if diff_ci.empty:
                    diff_ci = pd.DataFrame({'task': task, 'dimension': dimension, 'y_label': y_label, 'metric': metric, 'mean': np.nanmean(metrics_diff[metric]), 'ci_low': np.nanpercentile(metrics_diff[metric], 2.5), 'ci_high': np.nanpercentile(metrics_diff[metric], 97.5)}, index=[0])
                else:
                    diff_ci = pd.concat((diff_ci,pd.DataFrame({'task': task, 'dimension': dimension, 'y_label': y_label, 'metric': metric, 'mean': np.nanmean(metrics_diff[metric]), 'ci_low': np.nanpercentile(metrics_diff[metric], 2.5), 'ci_high': np.nanpercentile(metrics_diff[metric], 97.5)}, index=[0])))
            
            with open(Path(path_to_results,'shuffle',random_seed,f'metrics_best_model_{model_name}_shuffled.pkl'),'wb') as f:
                pickle.dump(metrics,f)

#        data_to_plot.to_csv(Path(results_dir,filename_to_save), index=False)
        
        if Path(path_to_results,'shuffle',random_seed, outputs_filename).exists():
            for metric in metrics_names:
                plt.figure()
                sns.violinplot(data=[metrics[metric], metrics_shuffle[metric]], inner=None, palette='muted')
                plt.xticks([0, 1], ['Real', 'Shuffle'])
                plt.ylabel(metric.replace('_', ' ').upper())
                plt.title(f"{metric.replace('_', ' ').upper()} Distribution for {model_name}")
                plt.tight_layout()
                plt.ylim(0, 1)
                plt.grid(True)
                plt.savefig(Path(results_dir,'plots',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',filename_to_save.replace('log_odds',f'{metric}_violin')))
                plt.savefig(Path(results_dir,'plots',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',filename_to_save.replace('log_odds.png',f'{metric}_violin.svg')))
                plt.close()

    if Path(path_to_results,'shuffle',random_seed, outputs_filename).exists():
        filename_to_save = f'diff_ci_shuffle_{scoring}_cal.csv' if calibrate else f'diff_ci_shuffle_{scoring}.csv'
        
        diff_ci.to_csv(Path(results_dir,filename_to_save), index=False)
        
        filename_to_save = filename_to_save.replace('diff_ci','nonparam_stats')
        nonparam_stats.to_csv(Path(results_dir,filename_to_save), index=False)
        filename_to_save = filename_to_save.replace('nonparam_stats','param_stats')
        param_stats.to_csv(Path(results_dir,filename_to_save), index=False)

        print('Done!')