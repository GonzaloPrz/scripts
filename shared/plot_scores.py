import seaborn as sns

import pandas as pd
import numpy as np
from pathlib import Path
import itertools, sys, pickle, tqdm, warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

project_name = 'tell_classifier'

l2ocv = False

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:/','CNC_Audio','gonza','results',project_name)

scaler_name = 'StandardScaler'

y_labels = ['target']

hyp_tuning_list = [True]
feature_selection_list = [True]

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

scoring = 'norm_cross_entropy'
extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if extremo == 'sup' else False

best_classifiers = pd.read_csv(Path(results_dir,f'best_classifiers_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection.csv'))

tasks = best_classifiers['task'].unique()
dimensions = best_classifiers.dimension.unique()

for r,row in best_classifiers.iterrows():
    print(row['task'],row['dimension'])
    for y_label,hyp_opt,feature_selection in itertools.product(y_labels,hyp_tuning_list,feature_selection_list):
        path_to_results = results_dir / row.task / row.dimension / scaler_name / kfold_folder / y_label / 'no_hyp_opt' / 'feature_selection'
        
        path_to_results = Path(str(path_to_results).replace('no_hyp_opt', 'hyp_opt')) if hyp_opt else path_to_results
        path_to_results = Path(str(path_to_results).replace('feature_selection', '')) if not feature_selection else path_to_results

        model_name = row.model_type
        model_index = pd.read_csv(Path(path_to_results,row.random_seed_test,f'all_models_{model_name}_dev.csv')).sort_values(f'{extremo}_{scoring}',ascending=ascending).index[0]

        outputs = pickle.load(open(Path(path_to_results,row.random_seed_test,f'outputs_{model_name}.pkl'),'rb'))[model_index,:,:,1]
        y_true = pickle.load(open(Path(path_to_results,row.random_seed_test,f'y_true_dev.pkl'),'rb'))

        y_true = np.concatenate([y_true[r,:] for r in range(y_true.shape[0])])
        outputs = np.concatenate([outputs[r,:] for r in range(outputs.shape[0])])

        plt.figure()

        h, e = np.histogram(np.log(np.exp(outputs[y_true==0])/(1-np.exp(outputs[y_true==0]))), bins=50, density=True)
        centers = (e[:-1] + e[1:]) / 2
        plt.plot(centers, h, label='HC',color='b')

        h, e = np.histogram(np.log(np.exp(outputs[y_true==1])/(1-np.exp(outputs[y_true==1]))), bins=50, density=True)
        centers = (e[:-1] + e[1:]) / 2
        plt.plot(centers, h, label='PD',color='r')
        
        plt.xlabel('Log-odds')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.legend()
        plt.savefig(Path(path_to_results,row.random_seed_test,f'best_{model_name}_log_odds.png'))