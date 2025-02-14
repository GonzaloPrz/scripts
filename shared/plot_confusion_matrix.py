import pickle, sys, json, os, itertools
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

parallel = True 
cmatrix = None

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
id_col = config['id_col']
bayesian = config['bayesian']

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path(r"D:/",r"CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = [main_config['scoring_metrics'][project_name]]
problem_type = main_config['problem_type'][project_name]

##---------------------------------PARAMETERS---------------------------------##
results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for scoring in scoring_metrics:
    best_models_file = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection_shuffle.csv'.replace('__','_')

    if not feature_selection:
        best_models_file = best_models_file.replace('feature_selection_','')
    if not shuffle_labels:
        best_models_file = best_models_file.replace('_shuffle','')
    if not hyp_opt:
        best_models_file = best_models_file.replace('hyp_opt','no_hyp_opt')

    best_models = pd.read_csv(Path(results_dir,best_models_file))

    tasks = best_models['task'].unique()
    y_labels = best_models['y_label'].unique()
    dimensions = best_models['dimension'].unique()  
    random_seeds_test = best_models['random_seed_test'].unique()
    
    for task,y_label in itertools.product(tasks,y_labels):
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            print(task,dimension)
            best_model = best_models[(best_models['task'] == task) & (best_models['y_label'] == y_label) & (best_models['dimension'] == dimension)]

            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection','shuffle')
            if not feature_selection:
                path = str(path).replace('feature_selection_','')
            if not shuffle_labels:
                path = str(path).replace('shuffle','')
            
            for random_seed in random_seeds_test:
                if random_seed == np.nan:
                    random_seed = ''
                
                model_name = best_model['model_type'].values[0]
                model_index = best_model['model_index'].values[0]

                with open(Path(path,random_seed,'bayesian' if bayesian else '','y_dev.pkl'),'rb') as f:
                    y_dev = pickle.load(f)
                with open(Path(path,random_seed,'bayesian' if bayesian else '','y_test.pkl'),'rb') as f:
                    y_test = pickle.load(f)
                with open(Path(path,random_seed,'bayesian' if bayesian else '',f'outputs_{model_name}.pkl'),'rb') as f:
                    outputs_dev = pickle.load(f)
                with open(Path(path,random_seed,'bayesian' if bayesian else '',f'outputs_test_{model_name}.pkl'),'rb') as f:
                    outputs_test = pickle.load(f)
                
                _, y_pred_dev = utils.get_metrics_clf(outputs_dev.squeeze(), y_dev, [])
                _, y_pred_test = utils.get_metrics_clf(outputs_test.squeeze(), y_test, [])

                cmatrix_dev = confusion_matrix(y_dev.flatten(), y_pred_dev[model_index].flatten(),normalize='all')
                cmatrix_test = confusion_matrix(y_test.values.flatten(), y_pred_test[model_index].flatten(),normalize='all')

                #Plot confusion matrix and save as fig
                fig, ax = plt.subplots(1,2,figsize=(10,5))
                ConfusionMatrixDisplay(cmatrix_dev).plot(ax=ax[0])
                ax[0].set_title('X-val')
                ConfusionMatrixDisplay(cmatrix_test).plot(ax=ax[1])
                ax[1].set_title('Test')

                fig.suptitle(f'{task} - {dimension} - {y_label} - {model_name} - {random_seed}')
                plt.savefig(Path(path,random_seed,f'confusion_matrix_{model_name}.png'))


                

                


