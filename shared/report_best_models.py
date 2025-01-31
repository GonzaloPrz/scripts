import pandas as pd
import numpy as np
from pathlib import Path
import itertools,sys

def new_best(current_best,value,ascending):
    if ascending:
        return value < current_best
    else:
        return value > current_best

##---------------------------------PARAMETERS---------------------------------##
    
project_name = 'ad_mci_hc'

stat_folder = 'mean_std'

n_folds = 5

shuffle_labels = False

hyp_opt = True

feature_selection_list = [False]

scaler_name = 'StandardScaler'

##---------------------------------PARAMETERS---------------------------------##

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['cog','connectivity','Animales','P','brain','Animales__P'],
         'GeroApathy':['agradable'],
         'GERO_Ivo':['fas','animales','fas__animales','grandmean'],
         'MPLS':['Estado General'],
         'AKU':['picture_description','pleasant_memory',
                'routine','video_retelling'],
         'AKU_outliers_as_nan':['picture_description','pleasant_memory',
                'routine','video_retelling'],
          'arequipa':['dia_tipico','lamina1','lamina2','fugu','testimonio'],
          'ad_mci_hc':['fugu']
                }

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GeroApathy':'clf',
                'GERO_Ivo':'reg',
                'MPLS':'reg',
                'AKU':'reg',
                'AKU_outliers_as_nan':'reg',
                'arequipa':'clf',
                'ad_mci_hc':'clf'}

metrics_names = {'tell_classifier': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'MCI_classifier': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'ad_mci_hc':['accuracy','norm_expected_cost','norm_cross_entropy'],
                 'Proyecto_Ivo': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'GeroApathy': ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'GeroApthy_reg':['r2_score','mean_absolute_error','mean_squared_error'],
                 'GERO_Ivo':['r2_score','mean_absolute_error','mean_squared_error']
}
stats = {'tell_classifier':'',
            'MCI_classifier':'',
            'Proyecto_Ivo':'',
            'GeroApathy':'',
            'GERO_Ivo':'',
            'MPLS':'',
            'AKU':'',
            'AKU_outliers_as_nan':'',
            'arequipa':'',
            'ad_mci_hc':''}

scoring_metrics = {'MCI_classifier':['norm_cross_entropy'],
           'tell_classifier':['norm_cross_entropy'],
           'Proyecto_Ivo':['roc_auc'],
           'GeroApathy':['norm_cross_entropy','roc_auc'],
           'arequipa':['roc_auc'],
           'GeroApathy_reg':['r2_score','mean_absolute_error'],
           'GERO_Ivo':['r2_score','mean_absolute_error'],
           'MPLS':['r2_score'],
           'AKU':['r2_score'],
           'AKU_outliers_as_nan':['r2_score'],
           'ad_mci_hc':['norm_cross_entropy']}

best_models = pd.DataFrame(columns=['task','dimension','y_label','model_type','model_index','random_seed_test'] + [f'{metric}_mean_dev' for metric in metrics_names[project_name]] 
                           + [f'{metric}_ic_dev' for metric in metrics_names[project_name]] 
                           + [f'{metric}_mean_holdout' for metric in metrics_names[project_name]]
                           + [f'{metric}_ic_holdout' for metric in metrics_names[project_name]])

pd.options.mode.copy_on_write = True 

if n_folds == 0:
    kfold_folder = 'l2ocv'
elif n_folds == -1:
    kfold_folder = 'loocv'
else:
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:/','CNC_Audio','gonza','results',project_name)
for scoring,feature_selection in itertools.product(scoring_metrics[project_name],feature_selection_list):
    for task in tasks[project_name]:
        extremo = 'sup' if any(x in scoring for x in ['error','norm']) else 'inf'
        ascending = True if extremo == 'sup' else False

        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        for dimension in dimensions:
            print(task,dimension)
            path = Path(results_dir,task,dimension,scaler_name,kfold_folder)

            if not path.exists():
                continue
            
            y_labels = [folder.name for folder in path.iterdir() if folder.is_dir() and folder.name != 'mean_std']
            for y_label in y_labels:
                path = Path(results_dir,task,dimension,scaler_name,kfold_folder,stats[project_name],y_label,stat_folder,'hyp_opt' if hyp_opt else 'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '','shuffle' if shuffle_labels else '')
                if not path.exists():
                    continue
                random_seeds_test = [folder.name for folder in path.iterdir() if folder.is_dir() if 'random_seed' in folder.name]
                if len(random_seeds_test) == 0:
                    random_seeds_test = ['']

                for random_seed_test in random_seeds_test:
                    
                    files = [file for file in Path(path,random_seed_test).iterdir() if f'all_models_' in file.stem and 'test' in file.stem] 
                    if len(files) == 0:
                        files = [file for file in Path(path,random_seed_test).iterdir() if f'all_models_' in file.stem and 'dev' in file.stem and 'bca' in file.stem]

                    if len(files) == 0:
                        files = [file for file in Path(path,random_seed_test).iterdir() if f'best_models_' in file.stem and 'test' in file.stem and scoring in file.stem] 

                    if len(files) == 0:
                        files = [file for file in Path(path,random_seed_test).iterdir() if f'best_models_' in file.stem and 'dev' in file.stem and 'bca' in file.stem and scoring in file.stem] 

                    best = None
                    for file in files:
                        #if 'svc' in file.stem or 'svr' in file.stem:
                        #    continue
                        
                        df = pd.read_csv(file)
                        
                        if f'{extremo}_{scoring}' in df.columns:
                            scoring_col = f'{extremo}_{scoring}'
                        elif f'{extremo}_{scoring}_dev' in df.columns:
                            scoring_col = f'{extremo}_{scoring}_dev'
                        else:
                            scoring_col = f'{scoring}_{extremo}'

                        try:
                            df = df.sort_values(by=scoring_col,ascending=ascending)
                        except:
                            continue
                        
                        print(f'{file.stem.split("_")[2]}:{df.iloc[0,:][scoring_col]}')
                        if best is None:
                            best = df.iloc[0,:]
                            
                            model_type = file.stem.split('_')[-2]
                            best['y_label'] = y_label
                            best['model_type'] = model_type
                            best['random_seed_test'] = random_seed_test
                            if 'idx' in df.columns:
                                best['model_index'] = df['idx'][0]
                            else:
                                best['model_index'] = df.index[0]
                        
                            best_file = file
                        else:
                            if new_best(best[scoring_col],df.iloc[0,:][scoring_col],ascending):
                                best = df.iloc[0,:]

                                model_type = file.stem.split('_')[2]
                                best['y_label'] = y_label
                                best['model_type'] = model_type
                                best['random_seed_test'] = random_seed_test
                                if 'idx' in df.columns:
                                    best['model_index'] = df['idx'][0]
                                else:
                                    best['model_index'] = df.index[0]

                                best_file = file
                    
                    for metric in metrics_names[project_name]:
                        print(metric)
                        if f'inf_{metric}_dev' in best.index:
                            best[f'{metric}_mean_dev'] = np.round(best[f"mean_{metric}_dev"],5)
                            best[f'{metric}_ic_dev'] = f'[{np.round(best[f"inf_{metric}_dev"],5)} - {np.round(best[f"sup_{metric}_dev"],5)}]'
                        elif f'inf_{metric}' in best.index:
                            best[f'{metric}_mean_dev'] = np.round(best[f"mean_{metric}"],5)
                            best[f'{metric}_ic_dev'] = f'[{np.round(best[f"inf_{metric}"],5)}, {np.round(best[f"sup_{metric}"],5)}]'
                        else:
                            best[f'{metric}_mean_dev'] = np.round(best[f"{metric}_mean"],5)
                            best[f'{metric}_ic_dev'] = f'[{np.round(best[f"{metric}_inf"],5)}, {np.round(best[f"{metric}_sup"],5)}]'

                        best[f'{metric}_test'] = np.nan

                        mean = np.round(best[f'mean_{metric}_test'],5)
                        inf = np.round(best[f'inf_{metric}_test'],5)
                        sup = np.round(best[f'sup_{metric}_test'],5)
                        best[f'{metric}_mean_holdout'] = mean
                        best[f'{metric}_ic_holdout'] = f'[{inf}, {sup}]'
                    model_type = file
                    
                    dict_append = {'task':task,'dimension':dimension,'y_label':y_label,'model_type':best['model_type'],'model_index':best['model_index'],'random_seed_test':random_seed_test}
                    dict_append.update(dict((f'{metric}_mean_dev',best[f'{metric}_mean_dev']) for metric in metrics_names[project_name]))
                    dict_append.update(dict((f'{metric}_ic_dev',best[f'{metric}_ic_dev']) for metric in metrics_names[project_name]))
                    dict_append.update(dict((f'{metric}_mean_hodlout',np.nan) for metric in metrics_names[project_name]))
                    dict_append.update(dict((f'{metric}_ic_holdout',np.nan) for metric in metrics_names[project_name]))
                    
                    try:
                        dict_append.update(dict((f'{metric}_mean_holdout',best[f'{metric}_mean_holdout']) for metric in metrics_names[project_name]))
                        dict_append.update(dict((f'{metric}_ic_holdout',best[f'{metric}_ic_holdout']) for metric in metrics_names[project_name]))
                    except:
                        pass
                    best_models.loc[len(best_models),:] = dict_append

    filename_to_save = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_no_hyp_opt_feature_selection_shuffled.csv'

    if hyp_opt:
        filename_to_save = filename_to_save.replace('no_hyp_opt','hyp_opt')
    if not feature_selection:
        filename_to_save = filename_to_save.replace('_feature_selection','')
    if not shuffle_labels:
        filename_to_save = filename_to_save.replace('_shuffled','')

    #best_models.dropna(subset=['model_index'],inplace=True)
    best_models.to_csv(Path(results_dir,filename_to_save),index=False)
