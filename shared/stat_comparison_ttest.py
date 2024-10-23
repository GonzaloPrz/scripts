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

planned_comparisons = {'Proyecto_Ivo':(('Animales___properties__vr','cog___neuropsico_mmse'),
                                        ('Animales___properties__vr','cog___neuropsico'),
                                        ('Animales___properties__vr','brain___norm_brain_lit'),
                                        ('Animales___properties__vr','AAL___norm_AAL'),
                                        ('Animales___properties__vr','conn___connectivity'),
                                        ('Animales___properties__timing','cog___neuropsico_mmse'),
                                        ('Animales___properties__timing','cog___neuropsico'),
                                        ('Animales___properties__timing','brain___norm_brain_lit'),
                                        ('Animales___properties__timing','AAL___norm_AAL'),
                                        ('Animales___properties__timing','conn___connectivity'))}
y_labels = {'Proyecto_Ivo':['target']}
            
if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

best_classifiers = pd.read_csv(Path(results_dir,f'best_classifiers_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection.csv'))

stats = pd.DataFrame(columns=['comparison','q_statistic','p_value'])

for y_label,comparison in itertools.product(y_labels[project_name],planned_comparisons[project_name]):
    model1 = comparison[0]
    model2 = comparison[1]

    task1 = model1.split('___')[0]
    dimension1 = model1.split('___')[1]

    task2 = model2.split('___')[0]
    dimension2 = model2.split('___')[1]

    model_name1 = best_classifiers[(best_classifiers['task'] == task1) & (best_classifiers['dimension'] == dimension1)]['model_type'].values[0]
    model_name2 = best_classifiers[(best_classifiers['task'] == task2) & (best_classifiers['dimension'] == dimension2)]['model_type'].values[0]

    model_index1 = best_classifiers[(best_classifiers['task'] == task1) & (best_classifiers['dimension'] == dimension1)]['model_index'].values[0]
    model_index2 = best_classifiers[(best_classifiers['task'] == task2) & (best_classifiers['dimension'] == dimension2)]['model_index'].values[0]

    outputs_1 = pickle.load(open(Path(results_dir,task1,dimension1,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection',f'outputs_bootstrap_{model_name1}.pkl'),'rb'))[:,model_index1,:,:,:]
    outputs_2 = pickle.load(open(Path(results_dir,task2,dimension2,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection',f'outputs_bootstrap_{model_name2}.pkl'),'rb'))[:,model_index2,:,:,:]
    
    metrics_1 = pickle.load(open(Path(results_dir,task1,dimension1,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection',f'metrics_bootstrap_{model_name1}.pkl'),'rb'))[scoring][:,model_index1,:].flatten()
    metrics_2 = pickle.load(open(Path(results_dir,task2,dimension2,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection',f'metrics_bootstrap_{model_name2}.pkl'),'rb'))[scoring][:,model_index2,:].flatten()

    t_statistic,p_value = ttest_rel(metrics_1,metrics_2)

    stats_append = {'comparison':f'{model1} vs {model2}','t_statistic':np.round(t_statistic,3),'p_value':np.round(p_value,3)}
    
    if stats.empty:
        stats = pd.DataFrame(stats_append,index=[0])
    else:
        stats = pd.concat((stats,pd.DataFrame(stats_append,index=[0])),ignore_index=True)

stats.to_csv(Path(results_dir,'stats_comparison_ttest.csv'),index=False)