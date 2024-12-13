import pandas as pd
from pathlib import Path
import numpy as np
from mlxtend.evaluate import cochrans_q
import itertools,pickle
import sys
from scipy.stats import ttest_rel
from pingouin import mixed_anova, anova, rm_anova

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

l2ocv = False

output = False

project_name = 'Proyecto_Ivo'

scaler_name = 'StandardScaler'
scoring = 'roc_auc'

planned_comparisons = {'Proyecto_Ivo':('Animales___properties__timing','cog___neuropsico_mmse','cog___neuropsico',
                                        'brain___norm_brain_lit','AAL___norm_AAL','conn___connectivity')
                    }
y_labels = {'Proyecto_Ivo':['target']}
            
if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

best_classifiers = pd.read_csv(Path(results_dir,f'best_classifiers_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection.csv'))

best_classifiers_shuffled = pd.read_csv(Path(results_dir,f'best_classifiers_{scoring}_{kfold_folder}_{scaler_name}_hyp_opt_feature_selection_shuffled.csv'))

data = pd.DataFrame(columns=['dimension','order','roc_auc','accuracy','outputs','idx'])

for y_label,task_dimension in itertools.product(y_labels[project_name],planned_comparisons[project_name]):
    task = task_dimension.split('___')[0]
    dimension = task_dimension.split('___')[1]

    model_name = best_classifiers[(best_classifiers['task'] == task) & (best_classifiers['dimension'] == dimension)]['model_type'].values[0]
    
    model_index = best_classifiers[(best_classifiers['task'] == task) & (best_classifiers['dimension'] == dimension)]['model_index'].values[0]
    
    outputs = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection',f'outputs_{model_name}.pkl'),'rb'))[model_index,:,:,1].flatten()
    
    roc_auc = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection',f'metrics_bootstrap_{model_name}.pkl'),'rb'))['roc_auc'][:,model_index,:].flatten()
    accuracy = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection',f'metrics_bootstrap_{model_name}.pkl'),'rb'))['accuracy'][:,model_index,:].flatten()

    df = pd.DataFrame({'idx': np.arange(len(accuracy)),
                       'roc_auc':roc_auc,'accuracy':accuracy,
                       #'outputs':outputs,
                       'dimension':task_dimension,'order':'normal'})
    
    data = pd.concat([data,df],axis=0)
    
    model_name_shuffled = best_classifiers_shuffled[(best_classifiers_shuffled['task'] == task) & (best_classifiers_shuffled['dimension'] == dimension)]['model_type'].values[0]

    model_index_shuffled = best_classifiers_shuffled[(best_classifiers_shuffled['task'] == task) & (best_classifiers_shuffled['dimension'] == dimension)]['model_index'].values[0]

    outputs_shuffled = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection','shuffle',f'outputs_{model_name_shuffled}.pkl'),'rb'))[model_index_shuffled,:,:,1].flatten()

    roc_auc_shuffled = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection','shuffle',f'metrics_bootstrap_{model_name_shuffled}.pkl'),'rb'))['roc_auc'][:,model_index_shuffled,:].flatten()
    accuracy_shuffled = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt','feature_selection','shuffle',f'metrics_bootstrap_{model_name_shuffled}.pkl'),'rb'))['accuracy'][:,model_index_shuffled,:].flatten()

    df = pd.DataFrame({'idx':np.arange(len(accuracy)),
                       'roc_auc':roc_auc_shuffled,'accuracy':accuracy_shuffled,
                       #'outputs':outputs_shuffled,
                       'dimension':task_dimension,'order':'shuffled'})

    data = pd.concat([data,df],axis=0)

#Mixed ANOVA

print('roc_auc:')

anova_results = rm_anova(data=data,dv='roc_auc',within=['order','dimension'],subject='idx',detailed=True)

print(anova_results)

print('accuracy:')

anova_results = rm_anova(data=data,dv='accuracy',within=['order','dimension'],subject='idx',detailed=True)
print(anova_results)

#print('outputs:')

#anova_results = rm_anova(data=data,dv='outputs',within=['order','dimension'],subject='idx',detailed=True)
#print(anova_results)
