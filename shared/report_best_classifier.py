import pandas as pd
import numpy as np
from pathlib import Path

def new_best(current_best,value,ascending):
    if ascending:
        return value < current_best
    else:
        return value > current_best
    
project_name = 'MCI_classifier'

tasks = [#'fas',
    'animales','fas__animales','grandmean'] 

scaler_name = 'StandardScaler'

best_classifiers = pd.DataFrame(columns=['task','dimension','model_type','random_seed_test',
                                         'AUC_dev',#'AUC_oob_dev',
                                         'AUC_holdout',
                                         'accuracy_dev',#'accuracy_oob_dev',
                                         'accuracy_holdout'])

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
y_label = 'target'

feature_selection_list = [True]
bootstrap = False

random_seeds_test = np.arange(n_seeds_test)

scoring = 'roc_auc'
extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if 'norm' in scoring else False

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)
for feature_selection in feature_selection_list:
    for task in tasks:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            print(task,dimension)
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection','bootstrap')
            path = Path(str(path).replace('feature_selection','')) if not feature_selection else path 
            path = Path(str(path).replace('bootstrap','')) if not bootstrap else path
            
            random_seeds_test = [folder.name for folder in path.iterdir() if folder.is_dir()]

            if len(random_seeds_test) == 0:
                random_seeds_test = ['']

            for random_seed_test in random_seeds_test:
                
                files = [file for file in Path(path,random_seed_test).iterdir() if 'best_performances_' in file.stem and 'svc' not in file.stem]
                best = None
                for file in files:
                    
                    df = pd.read_csv(file)
                    df = df.sort_values(by=f'{extremo}_{scoring}',ascending=ascending).reset_index(drop=True)
                    print(f'{file.stem.split("_")[-1]}:{df.loc[0,f"{extremo}_{scoring}"]}')
                    if best is None:
                        best = df.loc[0,:]

                        model_type = file.stem.split('_')[-1].replace('.csv','')
                        best['model_type'] = model_type
                        best_file = file
                    else:
                        if new_best(best[f'{extremo}_{scoring}'],df.loc[0,f'{extremo}_{scoring}'],ascending):
                            best = df.loc[0,:]
                            model_type = file.stem.split('_')[-1].replace('.csv','')
                            best['model_type'] = model_type
                            best_file = file
                if best is None:
                    continue
                print(best['model_type'])
                AUC = f'[ {best[f"inf_roc_auc"].round(2)}, {best[f"mean_roc_auc"].round(2)}, {best[f"sup_roc_auc"].round(2)}]'
                
                accuracy = f'[ {best["inf_accuracy"].round(2)}, {best["mean_accuracy"].round(2)}, {best["sup_accuracy"].round(2)}]'
                    
                AUC_test = 'NA'
                accuracy_test = 'NA'
                if Path(best_file.parent,f'best_10_{best["model_type"]}_test.csv').exists():
                    
                    best_test = pd.read_csv(Path(best_file.parent,f'best_10_{best["model_type"]}_test.csv')).loc[0,:]
                    AUC_test = f'[ {best_test[f"inf_roc_auc_test"].round(2)}, {best_test[f"mean_roc_auc_test"].round(2)}, {best_test[f"sup_roc_auc_test"].round(2)}]'
                
                    accuracy_test = f'[ {best_test["inf_accuracy_test"].round(2)}, {best_test["mean_accuracy_test"].round(2)}, {best_test["sup_accuracy_test"].round(2)}]'
                
                model_type = file
                best_classifiers.loc[len(best_classifiers),:] = pd.Series({'task':task,'dimension':dimension,'model_type':best['model_type'],'random_seed_test':random_seed_test,
                                                                            'AUC_dev':AUC,
                                                                            #'AUC_oob_dev':AUC_oob,
                                                                            'AUC_holdout':AUC_test,
                                                                            'accuracy_dev':accuracy,
                                                                            #'accuracy_oob_dev':accuracy_oob,
                                                                            'accuracy_holdout':accuracy_test,
                                                                            })

        filename_to_save = f'best_classifiers_{kfold_folder}_{scaler_name}_no_hyp_opt_feature_selection.csv'

    if hyp_opt:
        filename_to_save = filename_to_save.replace('no_hyp_opt','hyp_opt')
    if not feature_selection:
        filename_to_save = filename_to_save.replace('feature_selection','')

    best_classifiers.to_csv(Path(results_dir,filename_to_save),index=False)
