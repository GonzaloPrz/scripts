import pandas as pd
import numpy as np
from pathlib import Path
import itertools

results_dir = Path(Path.home(),'results','Proyecto_Ivo')

tasks = ['Animales','P','Animales_P','cog','brain']

best_classifiers = pd.DataFrame(columns=['task','dimension','model_type',
                                         'AUC_bootstrap_dev','AUC_oob_dev',
                                         'accuracy_bootstrap_dev','accuracy_oob_dev']
                                         )
pd.options.mode.copy_on_write = True 

l2ocv = False
n_seeds_train = 10
n_seeds_test = 1

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

hyp_opt = True
y_label = 'Grupo'

feature_selection = True
bootstrap = True

scoring = 'roc_auc'
extremo = 'sup' if 'norm' in scoring else 'inf'

for task in tasks:
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(task,dimension)
        path = Path(results_dir,task,dimension,'StandardScaler','all_features',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection','bootstrap')
        path = Path(str(path).replace('feature_selection','')) if not feature_selection else path 
        path = Path(str(path).replace('bootstrap','')) if not bootstrap else path

        files = [file for file in path.iterdir() if 'all_performances_' in file.stem and 'test' not in file.stem]
        best = None
        for file in files:
            df = pd.read_csv(file)
            if best is None:
                best = df.loc[0,:]
                model_type = file.stem.split('_')[-1].replace('.csv','')
                best['model_type'] = model_type
                best_file = file
            else:
                if df.loc[0,f'{extremo}_{scoring}_bootstrap'] > best[f'{extremo}_{scoring}_bootstrap']:
                    best = df.loc[0,:]
                    model_type = file.stem.split('_')[-1].replace('.csv','')
                    best['model_type'] = model_type
                    best_file = file
        
        print(best['model_type'])
        AUC_bootstrap = f'[ {best[f'inf_roc_auc_bootstrap'].round(2)}, {best[f"mean_roc_auc_bootstrap"].round(2)}, {best[f"sup_roc_auc_bootstrap"].round(2)}]'
        
        accuracy_bootstrap = f'[ {best["inf_accuracy_bootstrap"].round(2)}, {best["mean_accuracy_bootstrap"].round(2)}, {best["sup_accuracy_bootstrap"].round(2)}]'
    
        AUC_oob = f'[ {best[f'inf_roc_auc_oob'].round(2)}, {best[f"mean_roc_auc_oob"].round(2)}, {best[f"sup_roc_auc_oob"].round(2)}]'
    
        accuracy_oob = f'[ {best["inf_accuracy_oob"].round(2)}, {best["mean_accuracy_oob"].round(2)}, {best["sup_accuracy_oob"].round(2)}]'

        model_type = file
        best_classifiers.loc[len(best_classifiers),:] = pd.Series({'task':task,'dimension':dimension,'model_type':best['model_type'],
                                                                    'AUC_bootstrap_dev':AUC_bootstrap,
                                                                    'AUC_oob_dev':AUC_oob,
                                                                    'accuracy_bootstrap_dev':accuracy_bootstrap,
                                                                    'accuracy_oob_dev':accuracy_oob,
                                                                    })

filename_to_save = f'best_classifiers_{kfold_folder}_hyp_opt.csv' if hyp_opt else f'best_classifiers_{kfold_folder}_no_hyp_opt.csv'
best_classifiers.to_csv(Path(results_dir,filename_to_save),index=False)
