import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path
import itertools, os

import numpy as np

bootstrap = True
held_out_default_list = [False]
shuffle_labels_list = [False]
feature_selection_list = [False]
base_dir = os.getcwd()
hyp_tuning_list = [True,False]
y_labels = ['Grupo']
n_seeds = 10
k = 5

tasks = ['Animales','both','P','cog','AAL']
scaler = 'StandardScaler'

results = pd.DataFrame(columns=['task','dataset','model','calibration'])

filename_to_save = f'all_results_bt_{k}_folds' if bootstrap else f'all_results_{k}_folds'

for task in tasks:
    print(task)
    base_path = Path(base_dir,task)
    dimensions = [file.stem for file in base_path.iterdir() if file.is_dir()]
    
    for y_label,hyp_tuning,dimension,held_out_default,feature_selection in itertools.product(y_labels,hyp_tuning_list,dimensions,held_out_default_list,feature_selection_list):
        print(dimension)
        if hyp_tuning == True:
            held_out = True
            subfolder = Path(f'hyp_opt')
        else:
            held_out = held_out_default
            subfolder = Path(f'no_hyp_opt')
            
        print('\n','*'*20,f'held_out: {held_out}, hyperparameter tuning: {hyp_tuning}, feature_selection: {feature_selection}','*'*20,'\n')
        path_to_data = Path(base_path,dimension,scaler,f'{k}_folds',f'{n_seeds}_seeds',y_label,'bt',subfolder,'held_out')

        if not held_out:
            path_to_data = Path(str(path_to_data).replace('held_out',''))

        if not bootstrap:
            path_to_data = Path(str(path_to_data).replace('bt',''))

        elif feature_selection:
            path_to_data = Path(path_to_data,'feature_selection')
        else:
            path_to_data = path_to_data
        
        if not path_to_data.exists():
            continue
                
        filename_to_save = f'{filename_to_save}_feature_selection' if feature_selection and 'feature_selection' not in filename_to_save else filename_to_save
            
        files = [file for file in path_to_data.iterdir() if 'results' in file.stem and file.suffix == '.xlsx' and 'conf_int' not in file.stem]
        
        for file in files:
            
            df = pd.read_excel(file)
            df.replace([np.inf, -np.inf], np.nan,inplace=True)
            
            if 'random_seed' in df.columns:
                df.drop(columns=['random_seed'],axis=1,inplace=True)

            df_append = {}
            
            df_append['task'] = task
            df_append['dimension'] = dimension
            df_append['model'] = file.stem.split('_')[-1]
            if 'val' in file.stem and held_out:
                df_append['dataset'] = 'xval'
            elif held_out:
                df_append['dataset'] = 'held_out'
            else:
                df_append['dataset'] = ''
                
            df_append['calibration'] = 'No'
            df_append['hyperparameter_tuning'] = 'Yes' if hyp_tuning else 'No'
            df_append['y_label'] = y_label 
            metrics = df.columns[-7:]
            
            for metric in metrics:
                if metric == 'best_params':
                    continue

                df_append[f'{metric}_mean'] = np.nanmean(df[metric]).round(3)
                df_append[f'{metric}_std'] = np.nanstd(df[metric]).round(3)
                df_append[f'{metric}_nan_values'] = df[metric].isna().sum()

                if df[metric].isna().sum() > 0:
                    print(f'{metric} has {df[metric].isna().sum()} nan values in {file.stem}')

            if results.empty:
                results = pd.DataFrame(columns=df_append.keys())
                
            results.loc[len(results.index),:] = df_append
        
            if Path(file.parent,'calibrated',file.stem + '.xlsx').exists():
                df = pd.read_excel(Path(file.parent,'calibrated',file.stem + '.xlsx'))
                df.drop(columns=['random_seed'],axis=1,inplace=True)

                df.replace([np.inf, -np.inf], np.nan,inplace=True)
                
                df_append = {}
                
                df_append['task'] = task
                df_append['dimension'] = dimension
                df_append['model'] = file.stem.split('_')[-1]

                if 'val' in file.stem and held_out:
                    df_append['dataset'] = 'xval'
                elif held_out:
                    df_append['dataset'] = 'held_out'
                else:
                    df_append['dataset'] = ''
                    
                df_append['calibration'] = 'Yes'
                df_append['hyperparameter_tuning'] = 'Yes' if hyp_tuning else 'No'
                df_append['y_label'] = y_label 
                metrics = df.columns[:-1]
                
                for metric in metrics:
                    df_append[f'{metric}_mean'] = np.nanmean(df[metric]).round(3)
                    df_append[f'{metric}_std'] = np.nanstd(df[metric]).round(3)
                    df_append[f'{metric}_nan_values'] = df[metric].isna().sum()
                    
                    if df[metric].isna().sum() > 0:
                        print(f'{metric} has {df[metric].isna().sum()} nan values in {file.stem}')

                results.loc[len(results.index),:] = df_append


print('\n','*'*20,filename_to_save,'*'*20,'\n')

results = results.sort_values(by=['task','dataset','roc_auc_mean'],ascending=False)

results.to_excel(Path(base_dir,f'{filename_to_save}.xlsx'),index=False)