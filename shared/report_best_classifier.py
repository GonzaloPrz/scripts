import pandas as pd
import numpy as np
from pathlib import Path

def new_best(current_best,value,ascending):
    if ascending:
        return value < current_best
    else:
        return value > current_best
    
project_name = 'tell_classifier'
l2ocv = False

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean']}

scaler_name = 'StandardScaler'

best_classifiers = pd.DataFrame(columns=['task','dimension','model_type','random_seed_test',
                                         'AUC_dev',
                                         'AUC_holdout',
                                         'accuracy_dev',
                                         'accuracy_holdout'])

pd.options.mode.copy_on_write = True 

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

hyp_opt = True
y_label = 'target'

feature_selection_list = [True]

scoring = 'roc_auc'
extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if 'norm' in scoring else False

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)
for feature_selection in feature_selection_list:
    for task in tasks[project_name]:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            print(task,dimension)
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection')
            path = Path(str(path).replace('feature_selection','')) if not feature_selection else path 
            
            if not path.exists():
                continue
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
                AUC = f'[ {np.round(best[f"inf_roc_auc"],2)}, {np.round(best[f"mean_roc_auc"],2)}, {np.round(best[f"sup_roc_auc"],2)}]'
                
                accuracy = f'[ {np.round(best["inf_accuracy"],2)}, {np.round(best["mean_accuracy"],2)}, {np.round(best["sup_accuracy"],2)}]'
                    
                AUC_test = 'NA'
                accuracy_test = 'NA'
                if Path(best_file.parent,f'best_10_{best["model_type"]}_test.csv').exists():
                    
                    best_test = pd.read_csv(Path(best_file.parent,f'best_10_{best["model_type"]}_test.csv')).loc[0,:]
                    AUC_test = f'[ {np.round(best_test[f"inf_roc_auc_test"],2)}, {np.round(best_test[f"mean_roc_auc_test"],2)}, {np.round(best_test[f"sup_roc_auc_test"],2)}]'
                
                    accuracy_test = f'[ {np.round(best_test["inf_accuracy_test"],2)}, {np.round(best_test["mean_accuracy_test"],2)}, {np.round(best_test["sup_accuracy_test"],2)}]'
                
                model_type = file
                best_classifiers.loc[len(best_classifiers),:] = pd.Series({'task':task,'dimension':dimension,'model_type':best['model_type'],'random_seed_test':random_seed_test,
                                                                            'AUC_dev':AUC,
                                                                            'AUC_holdout':AUC_test,
                                                                            'accuracy_dev':accuracy,
                                                                            'accuracy_holdout':accuracy_test,
                                                                            })

        filename_to_save = f'best_classifiers_{kfold_folder}_{scaler_name}_no_hyp_opt_feature_selection.csv'

    if hyp_opt:
        filename_to_save = filename_to_save.replace('no_hyp_opt','hyp_opt')
    if not feature_selection:
        filename_to_save = filename_to_save.replace('feature_selection','')

    best_classifiers.to_csv(Path(results_dir,filename_to_save),index=False)
