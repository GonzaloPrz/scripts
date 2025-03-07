import seaborn as sns

import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

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
early_fusion = bool(config["early_fusion"])
bayesian = bool(config["bayesian"])
n_boot_test = int(config["n_boot_test"])
n_boot_train = int(config["n_boot_train"])

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]

scoring = 'roc_auc'
extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if extremo == 'sup' else False

best_models = pd.read_csv(Path(results_dir,f'best_models_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection.csv'))

tasks = best_models['task'].unique()
dimensions = best_models.dimension.unique()
y_labels = best_models.y_label.unique()

for r,row in best_models.iterrows():
    print(row['task'],row['dimension'])
    for y_label in y_labels:
        path_to_results = results_dir / row.task / row.dimension / scaler_name / kfold_folder / y_label / 'no_hyp_opt' / 'feature_selection'
        
        path_to_results = Path(str(path_to_results).replace('no_hyp_opt', 'hyp_opt')) if hyp_opt else path_to_results
        path_to_results = Path(str(path_to_results).replace('feature_selection', '')) if not feature_selection else path_to_results

        model_name = row.model_type
        if np.isnan(row.random_seed_test):
            random_seed = ''
        else:
            random_seed = row.random_seed_test

        try:
            model_index = pd.read_csv(Path(path_to_results,random_seed,f'all_models_{model_name}_dev_bca.csv')).sort_values(f'{scoring}_{extremo}',ascending=ascending).index[0]
        except:
            model_index = pd.read_csv(Path(path_to_results,random_seed,f'best_models_{scoring}_{model_name}_dev_bca.csv')).sort_values(f'{scoring}_{extremo}',ascending=ascending).index[0]

        outputs = pickle.load(open(Path(path_to_results,random_seed,f'outputs_{model_name}.pkl'),'rb')).squeeze()[model_index,:,:,1]
        y_true = pickle.load(open(Path(path_to_results,random_seed,f'y_true_dev.pkl'),'rb')).squeeze()

        y_true = np.concatenate([y_true[r,:] for r in range(y_true.shape[0])])
        outputs = np.concatenate([outputs[r,:] for r in range(outputs.shape[0])])

        plt.figure()

        h, e = np.histogram(np.log(np.exp(outputs[y_true==0])/(1-np.exp(outputs[y_true==0]))), bins=50, density=True)
        centers = (e[:-1] + e[1:]) / 2
        plt.plot(centers, h, label='0',color='b')

        h, e = np.histogram(np.log(np.exp(outputs[y_true==1])/(1-np.exp(outputs[y_true==1]))), bins=50, density=True)
        centers = (e[:-1] + e[1:]) / 2
        plt.plot(centers, h, label='1',color='r')
        
        plt.xlabel('Log-odds')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.legend()
        plt.savefig(Path(path_to_results,random_seed,f'best_{model_name}_log_odds.png'))