import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

dimensions = ['properties','timing','properties_timing',
              'properties_vr','timing_vr','properties_timing_vr']
n_seeds = 10
n_folds = 5
scaler_name = 'StandardScaler'
y_label = 'Grupo'

tasks = ['Animales','both','P']

base_dir = Path(__file__).parent

for task,dimension in itertools.product(tasks,dimensions):
    path_to_results = base_dir / task / dimension / scaler_name / f'{n_folds}_folds' / f'{n_seeds}_seeds' / y_label / 'bt' / 'hyp_opt' / 'held_out'
    path_to_save = Path(path_to_results,'plots')

    files = [file for file in path_to_results.iterdir() if 'all_results' in file.stem]

    for file in files:
        model = file.stem.split('_')[-1]

        Path(path_to_save,model).mkdir(parents=True, exist_ok=True)
        Path(path_to_save,model,'norm_cross_entropy').mkdir(parents=True, exist_ok=True)
        Path(path_to_save,model,'norm_expected_cost').mkdir(parents=True, exist_ok=True)

        results = pd.read_excel(file)
        
        n_boot = len(results['bootstrap'].unique())
        n_iter = int(results.shape[0]/(n_seeds*np.max((1,n_boot))))

        param_names = results.columns[:-9]
        
        for i in range(n_iter):            
            results_iter = results.iloc[i*n_seeds*np.max((1,n_boot)):(i+1)*n_seeds*np.max((1,n_boot)),:]

            metrics_iter = results_iter.iloc[:,-7:]
        
            norm_cross_entropy = metrics_iter.pop('norm_cross_entropy')
            norm_expected_cost = metrics_iter.pop('norm_expected_cost')
        
            title = ''
            for param in param_names:
                title += f'{param}: {results_iter[param].iloc[0]} || '
            title = title[:-4]

            fig, ax = plt.subplots(figsize=(10, 6))

            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            sns.kdeplot(data=metrics_iter, ax=ax,fill=True, common_norm=False, linewidth=2.5)
            roc_auc_inf = metrics_iter['roc_auc'].quantile(0.025)
            roc_auc_sup = metrics_iter['roc_auc'].quantile(0.975)
            ax.axvline(roc_auc_inf, color='red', linestyle='--', label='AUC-ROC 95% CI')
            ax.axvline(roc_auc_sup, color='red', linestyle='--')
            ax.set_xlabel('metric')
            ax.set_ylabel('density')
            #Show confidence interval values in axvline
            ax.set_title(title)

            plt.tight_layout()
        
            plt.savefig(path_to_save / model / f'iter_{i}.png')
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))

            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            sns.kdeplot(data=norm_cross_entropy, ax=ax,fill=True, common_norm=False, linewidth=2.5)
            norm_cross_entropy_inf = norm_cross_entropy.quantile(0.025)
            norm_cross_entropy_sup = norm_cross_entropy.quantile(0.975)
            ax.axvline(norm_cross_entropy_inf, color='red', linestyle='--', label='Norm cross entropy 95% CI')
            ax.axvline(norm_cross_entropy_sup, color='red', linestyle='--')
            ax.legend()
            ax.set_xlabel('Norm cross entropy')
            ax.set_ylabel('density')
 
            ax.set_title(title)

            plt.tight_layout()
        
            plt.savefig(path_to_save / model / 'norm_cross_entropy' / f'norm_cross_entropy_iter_{i}.png')
            plt.close()

            fig, ax = plt.subplots(figsize=(10, 6))

            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            sns.kdeplot(data=norm_expected_cost, ax=ax,fill=True, common_norm=False, linewidth=2.5)
            norm_expected_cost_inf = norm_expected_cost.quantile(0.025)
            norm_expected_cost_sup = norm_expected_cost.quantile(0.975)

            ax.axvline(norm_expected_cost_inf, color='red', linestyle='--', label='Norm expected cost 95% CI')
            ax.axvline(norm_expected_cost_sup, color='red', linestyle='--')
            #
            ax.set_xlabel('Norm expected cost')
            ax.set_ylabel('density')
 
            ax.set_title(title)

            plt.tight_layout()
        
            plt.savefig(path_to_save / model / 'norm_expected_cost' / f'norm_expected_cost_{i}.png')
            plt.close()

