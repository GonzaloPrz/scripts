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

        best_models_filename = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection_calibrated.csv'.replace('__','_')
        if not hyp_opt:
            best_models_filename = best_models_filename.replace('_hyp_opt','')
        if not feature_selection:
            best_models_filename = best_models_filename.replace('_feature_selection','')
        if not calibrate:
            best_models_filename = best_models_filename.replace('_calibrated','')
        
        if not Path(results_dir,best_models_filename).exists():
            continue

        best_models = pd.read_csv(Path(results_dir,best_models_filename))

        for r,row in best_models.iterrows():
            task = row.task
            y_label = row.y_label
            dimension = row.dimension
            model_name = row.model_type
            random_seed = row.random_seed_test  
            path_to_results = Path(results_dir, task, dimension, scaler_name, kfold_folder, y_label, stat_folder,'hyp_opt' if hyp_opt else '', 'feature_selection' if feature_selection else '', 'filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '')

            for calibrate in [False,True]:                
                Path(results_dir,'plots',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed).mkdir(parents=True, exist_ok=True)
                
                file = f'all_models_{model_name}_dev_bca_calibrated.csv' if calibrate else f'all_models_{model_name}_dev_bca.csv'

                if config['n_models'] != 0:
                    file = file.replace('all_models', 'best_models').replace('.csv', f'_{scoring}.csv')
                
                if not Path(path_to_results,random_seed,file).exists():
                    continue
                
                df_filename = pd.read_csv(Path(path_to_results, random_seed, file)).sort_values(f'{scoring}_{extremo}'.replace('_score',''), ascending=ascending)
                model_index = df_filename.index[0]

                if 'threshold' in df_filename.columns:
                    threshold = df_filename['threshold'][0]
                else:
                    threshold = None
                    
                if Path(path_to_results, 'shuffle', random_seed,file).exists():
                    df_filename_shuffle = pd.read_csv(Path(path_to_results, 'shuffle', random_seed, f'all_models_{model_name}_dev_bca.csv')).sort_values(f'{scoring}_{extremo}'.replace('_score',''), ascending=ascending)
                    model_index_shuffle = df_filename_shuffle.index[0]
                    if 'threshold' in df_filename_shuffle.columns:
                        threshold_shuffle = df_filename_shuffle['threshold'][0]
                    else:
                        threshold_shuffle = None
                    
                outputs_filename = f'cal_outputs_{model_name}.pkl' if calibrate else f'outputs_{model_name}.pkl'
                
                ax = None

                if Path(path_to_results,random_seed,outputs_filename.replace('outputs','outputs_test')).exists():
                    outputs_test = pickle.load(open(Path(path_to_results,random_seed,outputs_filename.replace('outputs','outputs_test')), 'rb'))[model_index]
                    #Add missing dimensions: model_index, j
                    y_test = pickle.load(open(Path(path_to_results,random_seed,f'y_test.pkl'), 'rb'))
                    IDs_test = pickle.load(open(Path(path_to_results,random_seed,f'IDs_test.pkl'), 'rb'))
                    
                    if outputs_test.shape[0] > 1:
                        y_test = np.concatenate([y_test for _ in range(outputs_test.shape[0])])
                        IDs_test = np.concatenate([IDs_test for _ in range(outputs_test.shape[0])])

                    scores = np.concatenate([outputs_test[r] for r in range(outputs_test.shape[0])])
                    
                    plot_hists(y_test, scores, outfile=Path(results_dir,'plots',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed,f'best_{model_name}_cal_logpost.png' if calibrate else '' + f'best_{model_name}_logpost_test.png'), nbins=50, group_by='score', style='--', label_prefix='test ', axs=None)
                
                outputs_ = pickle.load(open(Path(path_to_results, random_seed, outputs_filename), 'rb'))[:,model_index]

                #Add missing dimensions: model_index, j
                outputs_ =outputs_[:,np.newaxis, ...]
                
                y_true_ = pickle.load(open(Path(path_to_results,random_seed, f'y_dev.pkl'), 'rb'))
                IDs = pickle.load(open(Path(path_to_results,random_seed, f'IDs_dev.pkl'), 'rb'))
                
                scores = np.concatenate([outputs_[0,0,r] for r in range(outputs_.shape[2])])
                y_true = np.concatenate([y_true_[0,r] for r in range(y_true_.shape[1])])
                
                plot_hists(y_true, scores, outfile=Path(results_dir,'plots',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '',random_seed,f'best_{model_name}_cal_logpost.png' if calibrate else '' + f'best_{model_name}_logpost.png'), nbins=50, group_by='score', style='-', label_prefix='dev ', axs=ax)
                