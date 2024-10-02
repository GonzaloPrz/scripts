import pandas as pd
import numpy as np
from pathlib import Path
import itertools

def new_best(current_best,value,ascending):
    if ascending:
        return value < current_best
    else:
        return value > current_best
    
tasks = ['fas','animales','fas__animales','grandmeam']
project_name = 'MCI_classifier'

best_classifiers = pd.DataFrame(columns=['task','dimension','model_type','random_seed_test',
                                         'AUC_bootstrap_dev',#'AUC_oob_dev',
                                         'AUC_bootstrap_holdout',
                                         'accuracy_bootstrap_dev',#'accuracy_oob_dev',
                                         'accuracy_bootstrap_holdout'])

pd.options.mode.copy_on_write = True 

l2ocv = False
n_seeds_train = 10

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

hyp_opt = True
n_seeds_test = 1
y_label = 'group'

feature_selection = True
bootstrap = True

random_seeds_test = np.arange(n_seeds_test)

scoring = 'roc_auc'
extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if 'norm' in scoring else False

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)
for task in tasks:
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(task,dimension)
        path = Path(results_dir,task,dimension,'StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection','bootstrap')
        path = Path(str(path).replace('feature_selection','')) if not feature_selection else path 
        path = Path(str(path).replace('bootstrap','')) if not bootstrap else path
        for random_seed_test in random_seeds_test:
            
            files = [file for file in Path(path,f'random_seed_{random_seed_test}').iterdir() if 'all_performances_' in file.stem and 'test' not in file.stem]
            best = None
            for file in files:
                
                df = pd.read_csv(file)
                df = df.sort_values(by=f'{extremo}_{scoring}_bootstrap',ascending=ascending).reset_index(drop=True)
                print(f'{file.stem.split('_')[-1]}:{df.loc[0,f'{extremo}_{scoring}_bootstrap']}')
                if best is None:
                    best = df.loc[0,:]

                    model_type = file.stem.split('_')[-1].replace('.csv','')
                    best['model_type'] = model_type
                    best_file = file
                else:
                    if new_best(best[f'{extremo}_{scoring}_bootstrap'],df.loc[0,f'{extremo}_{scoring}_bootstrap'],ascending):
                        best = df.loc[0,:]
                        model_type = file.stem.split('_')[-1].replace('.csv','')
                        best['model_type'] = model_type
                        best_file = file
            if best is None:
                continue
            print(best['model_type'])
            AUC_bootstrap = f'[ {best[f'inf_roc_auc_bootstrap'].round(2)}, {best[f"mean_roc_auc_bootstrap"].round(2)}, {best[f"sup_roc_auc_bootstrap"].round(2)}]'
            
            accuracy_bootstrap = f'[ {best["inf_accuracy_bootstrap"].round(2)}, {best["mean_accuracy_bootstrap"].round(2)}, {best["sup_accuracy_bootstrap"].round(2)}]'
        
            AUC_oob = f'[ {best[f'inf_roc_auc_oob'].round(2)}, {best[f"mean_roc_auc_oob"].round(2)}, {best[f"sup_roc_auc_oob"].round(2)}]'
        
            accuracy_oob = f'[ {best["inf_accuracy_oob"].round(2)}, {best["mean_accuracy_oob"].round(2)}, {best["sup_accuracy_oob"].round(2)}]'

            AUC_bootstrap_test = 'NA'
            accuracy_bootstrap_test = 'NA'
            if Path(best_file.parent,f'best_10_{best['model_type']}_test.csv').exists():
                
                best_test = pd.read_csv(Path(best_file.parent,f'best_10_{best['model_type']}_test.csv')).loc[0,:]
                AUC_bootstrap_test = f'[ {best_test[f'inf_roc_auc_bootstrap_test'].round(2)}, {best_test[f"mean_roc_auc_bootstrap_test"].round(2)}, {best_test[f"sup_roc_auc_bootstrap_test"].round(2)}]'
            
                accuracy_bootstrap_test = f'[ {best_test["inf_accuracy_bootstrap_test"].round(2)}, {best_test["mean_accuracy_bootstrap_test"].round(2)}, {best_test["sup_accuracy_bootstrap_test"].round(2)}]'
            
            model_type = file
            best_classifiers.loc[len(best_classifiers),:] = pd.Series({'task':task,'dimension':dimension,'model_type':best['model_type'],'random_seed_test':random_seed_test,
                                                                        'AUC_bootstrap_dev':AUC_bootstrap,
                                                                        #'AUC_oob_dev':AUC_oob,
                                                                        'AUC_bootstrap_holdout':AUC_bootstrap_test,
                                                                        'accuracy_bootstrap_dev':accuracy_bootstrap,
                                                                        #'accuracy_oob_dev':accuracy_oob,
                                                                        'accuracy_bootstrap_holdout':accuracy_bootstrap_test,
                                                                        })

filename_to_save = f'best_classifiers_{kfold_folder}_hyp_opt.csv' if hyp_opt else f'best_classifiers_{kfold_folder}_no_hyp_opt.csv'
best_classifiers.to_csv(Path(results_dir,filename_to_save),index=False)
