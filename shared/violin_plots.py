import seaborn as sns
import pandas as pd
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings, json, os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

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
dimensions = main_config['single_dimensions'][project_name]
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

for scoring in scoring_metrics:
    extremo = 'sup' if 'norm' in scoring else 'inf'
    ascending = True if extremo == 'sup' else False
    data_to_plot = pd.DataFrame()
    best_models_filename = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection.csv'.replace('__','_')
    best_models = pd.read_csv(Path(results_dir,best_models_filename))

    for r, row in best_models.iterrows():
        task = row.task
        dimension = row.dimension
        y_label = row.y_label
        model_name = row.model_type

        data_append = pd.DataFrame()
        Path(results_dir,'plots',task,dimension,y_label,stat_folder,scoring,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '').mkdir(parents=True, exist_ok=True)

        print(task, dimension)
        path_to_results = Path(results_dir, task, dimension, scaler_name, kfold_folder, y_label, stat_folder,'hyp_opt' if hyp_opt else '', 'feature_selection' if feature_selection else '', 'filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '')

        if not hyp_opt:
            best_models_filename = best_models_filename.replace('_hyp_opt','')
        if not feature_selection:
            best_models_filename = best_models_filename.replace('_feature_selection','')
                            
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
                            
        outputs_filename = f'outputs_{model_name}.pkl'
        outputs_ = pickle.load(open(Path(path_to_results, random_seed, outputs_filename), 'rb'))[:,model_index]
        y_true_ = pickle.load(open(Path(path_to_results,random_seed, f'y_dev.pkl'), 'rb'))
        IDs_ = pickle.load(open(Path(path_to_results,random_seed, f'IDs_dev.pkl'), 'rb'))

        results = Parallel(n_jobs=1)(delayed(compute_metrics)(j,model_index, r, outputs_, y_true_,IDs_,metrics_names, n_boot, problem_type, cmatrix=None, priors=None, threshold=threshold) for j,model_index, r in itertools.product(range(outputs_.shape[0]),range(outputs_.shape[1]),range(outputs_.shape[2])))
        metrics = dict((metric,) for metric in metrics_names)
        
        #Check whether IDs and IDs_shuffle are the same

        for metric in metrics_names:
            for j, model_index, r, metrics_result, IDs in results:
                metrics[metric][j, model_index, r, :] = metrics_result[metric]
            metrics[metric] = metrics[metric].flatten()

            data_append[metric] = metrics[metric]
            data_append['dimension'] = dimension
            data_append['task'] = task
    
    if data_to_plot.empty:
        data_to_plot = data_append
    else:
        data_to_plot = pd.concat(data_to_plot,data_append,axis=0)
#        data_to_plot.to_csv(Path(results_dir,filename_to_save), index=False)
    
    for metric in metrics_names:
        plt.figure()
        sns.violinplot(data=data_to_plot,x=task,y=metric, palette='muted')
        plt.ylabel(metric.replace('_', ' ').upper())
        plt.title(f"{metric.replace('_', ' ').upper()} Distribution for {model_name}")
        plt.tight_layout()
        plt.grid(True)
        filename_to_save = f'violin_{metric}_best_model_{task}_{dimension}_{y_label}_{stat_folder}_{scoring}_hyp_opt_feature_selection_shuffle'
        if not hyp_opt:
            filename_to_save = filename_to_save.replace(filename_to_save,'_hyp_opt','')
        if not feature_selection:
            filename_to_save = filename_to_save.replace(filename_to_save,'_feature_selection','')
        if not shuffle_labels:
            filename_to_save = filename_to_save.replace(filename_to_save,'_shuffle','')

        plt.savefig(Path(results_dir,'plots',f'{filename_to_save}.png'))
        plt.savefig(Path(results_dir,'plots',f'{filename_to_save}.svg'))
        plt.close()