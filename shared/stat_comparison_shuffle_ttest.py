import pandas as pd
from pathlib import Path
import numpy as np
from mlxtend.evaluate import cochrans_q
import itertools,pickle
import sys
from scipy.stats import ttest_rel

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

l2ocv = False

project_name = 'Proyecto_Ivo'

scaler_name = 'StandardScaler'
scoring = 'roc_auc'

models = {'Proyecto_Ivo':['Animales___properties__vr',
                        'cog___neuropsico_mmse',
                        'cog___neuropsico',
                        'brain___norm_brain_lit',
                        'AAL___norm_AAL',
                        'conn___connectivity',
                        'Animales___properties__timing']}

y_labels = {'Proyecto_Ivo':['target']}
            
if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

best_classifiers = pd.read_csv(Path(results_dir,f'best_classifiers_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection.csv'))
best_classifiers_shuffle = pd.read_csv(Path(results_dir,f'best_classifiers_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection_shuffled.csv'))

stats = pd.DataFrame(columns=['comparison','t_statistic','p_value'])

for y_label,model in itertools.product(y_labels[project_name],models[project_name]):
   
    task = model.split('___')[0]
    dimension = model.split('___')[1]

    model_name = best_classifiers[(best_classifiers['task'] == task) & (best_classifiers['dimension'] == dimension)]['model_type'].values[0]
    model_name_shuffle = best_classifiers_shuffle[(best_classifiers_shuffle['task'] == task) & (best_classifiers_shuffle['dimension'] == dimension)]['model_type'].values[0]

    model_index = best_classifiers[(best_classifiers['task'] == task) & (best_classifiers['dimension'] == dimension)]['model_index'].values[0]
    model_index_shuffle = best_classifiers_shuffle[(best_classifiers_shuffle['task'] == task) & (best_classifiers_shuffle['dimension'] == dimension)]['model_index'].values[0]

    metrics = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection',f'metrics_bootstrap_{model_name}.pkl'),'rb'))[scoring][:,model_index,:].flatten()
    metrics_shuffle = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection','shuffle',f'metrics_bootstrap_{model_name_shuffle}.pkl'),'rb'))[scoring][:,model_index_shuffle,:].flatten()

    t_statistic,p_value = ttest_rel(metrics,metrics_shuffle)

    stats_append = {'comparison':f'{task}_{dimension}','t_statistic':np.round(t_statistic,3),'p_value':np.round(p_value,3)}
    
    if stats.empty:
        stats = pd.DataFrame(stats_append,index=[0])
    else:
        stats = pd.concat((stats,pd.DataFrame(stats_append,index=[0])),ignore_index=True)

stats.to_csv(Path(results_dir,'stats_comparison_ttest_shuffle.csv'),index=False)