import pandas as pd
import numpy as np
from pathlib import Path

def new_best(current_best,value,ascending):
    if ascending:
        return value < current_best
    else:
        return value > current_best

##---------------------------------PARAMETERS---------------------------------##
    
project_name = 'GeroApathy'

l2ocv = False

shuffle_labels = False

hyp_opt = True

feature_selection_list = [True]

scaler_name = 'StandardScaler'

n_folds = 5
##---------------------------------PARAMETERS---------------------------------##

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['Animales','P','Animales__P','cog','brain','AAL','conn'],
         'GeroApathy':['Fugu']}

metrics_names = {'tell_classifier':['roc_auc','accuracy','norm_expected_cost','norm_cross_entropy','recall','f1'],
                    'MCI_classifier':['roc_auc','accuracy','norm_expected_cost','norm_cross_entropy','recall','f1'],
                    'Proyecto_Ivo':['roc_auc','accuracy','norm_expected_cost','norm_cross_entropy','recall','f1'],
                    'GeroApathy':['r2_score','mean_absolute_error']}

scoring = {'tell_classifier':'norm_cross_entropy',
            'MCI_classifier':'norm_cross_entropy',
            'Proyecto_Ivo':'roc_auc',
            'GeroApathy':'r2_score'}

stats = {'tell_classifier':'',
            'MCI_classifier':'',
            'Proyecto_Ivo':'',
            'GeroApathy':''}

best_models = pd.DataFrame(columns=['task','dimension','y_label','model_type','model_index','random_seed_test'] + [f'{metric}_dev' for metric in metrics_names[project_name]] + [f'{metric}_holdout' for metric in metrics_names[project_name]])

pd.options.mode.copy_on_write = True 

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    kfold_folder = f'{n_folds}_folds'

extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if 'norm' in scoring else False

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)
for feature_selection in feature_selection_list:
    for task in tasks[project_name]:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            print(task,dimension)
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder)

            if not path.exists():
                continue
            
            y_labels = [folder.name for folder in path.iterdir() if folder.is_dir() and folder.name != 'mean_std']
            for y_label in y_labels:
                path = Path(results_dir,task,dimension,scaler_name,kfold_folder,stats[project_name],y_label,'hyp_opt' if hyp_opt else 'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
            
                random_seeds_test = [folder.name for folder in path.iterdir() if folder.is_dir() if 'random_seed' in folder.name]

                if len(random_seeds_test) == 0:
                    random_seeds_test = ['']

                for random_seed_test in random_seeds_test:
                    
                    files = [file for file in Path(path,random_seed_test).iterdir() if f'all_models_' in file.stem and 'test' in file.stem and any(x in file.stem for x in metrics_names[project_name])]
                    if len(files) == 0:
                        files = [file for file in Path(path,random_seed_test).iterdir() if f'all_models_' in file.stem and 'dev' in file.stem]

                    if len(files) == 0:
                        continue

                    best = None
                    for file in files:
                        
                        df = pd.read_csv(file)
                        
                        if f'{extremo}_{scoring[project_name]}' in df.columns:
                            scoring_col = f'{extremo}_{scoring[project_name]}'
                        else:
                            scoring_col = f'{extremo}_{scoring[project_name]}'

                        df = df.sort_values(by=scoring_col,ascending=ascending)
                        
                        print(f'{file.stem.split("_")[-2]}:{df.loc[0,scoring_col]}')
                        if best is None:
                            best = df.loc[0,:]
                            
                            model_type = file.stem.split('_')[-2]
                            best['y_label'] = y_label
                            best['model_type'] = model_type
                            best['model_index'] = df.index[0]
                            best_file = file
                        else:
                            if new_best(best[scoring_col],df.loc[0,scoring_col],ascending):
                                best = df.loc[0,:]

                                model_type = file.stem.split('_')[-2]
                                best['y_label'] = y_label
                                best['model_type'] = model_type
                                best['model_index'] = df.index[0]
                                best_file = file
                    if best is None:
                        continue
                    
                    print(best['model_type'])
                    for metric in metrics_names[project_name]:
                        try:
                            best[f'{metric}_dev'] = f'[{np.round(best[f"inf_{metric}_dev"],2)}, {np.round(best[f"mean_{metric}_dev"],2)}, {np.round(best[f"sup_{metric}_dev"],2)}]'
                        except:
                            best[f'{metric}_dev'] = f'[{np.round(best[f"inf_{metric}"],2)}, {np.round(best[f"mean_{metric}"],2)}, {np.round(best[f"sup_{metric}"],2)}]'

                        best[f'{metric}_holdout'] = np.nan
                        try:
                            mean = np.round(best[f'mean_{metric}_test'],2)
                            inf = np.round(best[f'inf_{metric}_test'],2)
                            sup = np.round(best[f'sup_{metric}_test'],2)
                            best[f'{metric}_holdout'] = f'[ {inf}, {mean}, {sup}]'
                        except:
                            continue

                    model_type = file
                    
                    dict_append = {'task':task,'dimension':dimension,'y_label':y_label,'model_type':best['model_type'],'model_index':best['model_index'],'random_seed_test':random_seed_test}
                    dict_append.update(dict((f'{metric}_dev',best[f'{metric}_dev']) for metric in metrics_names[project_name]))
                    dict_append.update(dict((f'{metric}_holdout',best[f'{metric}_holdout']) for metric in metrics_names[project_name]))

                    best_models.loc[len(best_models),:] = pd.Series(dict_append)

            filename_to_save = f'best_models_{scoring[project_name]}_{kfold_folder}_{scaler_name}_no_hyp_opt_feature_selection_shuffled.csv'

    if hyp_opt:
        filename_to_save = filename_to_save.replace('no_hyp_opt','hyp_opt')
    if not feature_selection:
        filename_to_save = filename_to_save.replace('_feature_selection','')
    if not shuffle_labels:
        filename_to_save = filename_to_save.replace('_shuffled','')

    best_models.to_csv(Path(results_dir,filename_to_save),index=False)
