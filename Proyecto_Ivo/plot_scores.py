import pandas as pd
from pathlib import Path
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import numpy as np

base_dir = Path(Path(__file__).parent)

loocv = True

tasks = ['Animales','P','both']

dimensions = ['properties','timing','properties_timing']

scaler_name = 'StandardScaler'  

if loocv:
    kfold_folder = 'loocv'
else:
    kfold_folder = '5_folds'

for task,dim in itertools.product(tasks,dimensions):
    parent_path = Path(base_dir,task,dim,scaler_name,kfold_folder)
    
    n_seeds_train = [folder.name for folder in Path(base_dir,task,dim,scaler_name,kfold_folder).iterdir() if folder.is_dir()]
    
    for n_seed_train in n_seeds_train:
        n_seeds_test = [folder.name for folder in Path(base_dir,task,dim,scaler_name,kfold_folder,n_seed_train).iterdir() if folder.is_dir()]

        for n_seed_test in n_seeds_test:
            y_labels = [folder.name for folder in Path(base_dir,task,dim,scaler_name,kfold_folder,n_seed_train,n_seed_test).iterdir() if folder.is_dir()]

            for y_label in y_labels:
                hyp_opt_list = [folder.name for folder in Path(base_dir,task,dim,scaler_name,kfold_folder,n_seed_train,n_seed_test,y_label).iterdir() if folder.is_dir()]

                for hyp_opt in hyp_opt_list:
                    path_to_results = Path(base_dir,task,dim,scaler_name,kfold_folder,n_seed_train,n_seed_test,y_label,hyp_opt)
                    files = [file for file in path_to_results.iterdir() if 'all_scores' in file.stem]
                    
                    for file in files:
                        scores = pd.read_csv(file) if 'csv' in file.suffix else pd.read_excel(file)

                        model = file.stem.split('_')[-1]
                        
                        random_seeds_train = scores['random_seed_train'].unique()
                        random_seeds_test = scores['random_seed_test'].unique()

                        for random_seed_train,random_seed_test in itertools.product(random_seeds_train,random_seeds_test):
                            path_to_save = Path(path_to_results,f'random_test_{random_seed_test}','score_plots',model,'val' if 'val' in file.stem else 'holdout') 
                            path_to_save.mkdir(parents=True,exist_ok=True)
                            
                            scores_seed = scores[(scores['random_seed_train'] == random_seed_train) & (scores['random_seed_test'] == random_seed_test)]
                            
                            params = set(scores_seed.columns) - set(['random_seed_train','random_seed_test','bootstrap','y_true','y_pred','logpost','ID','y_scores'])

                            unique_combinations = scores_seed[list(params)].drop_duplicates()

                            for _,combination in unique_combinations.iterrows():
                                scores_combination = scores_seed[(scores_seed[list(params)] == combination).all(axis=1)]
                                
                                if 'y_scores' in scores_combination.columns:
                                    scores_combination['logpost'] = scores_combination['y_scores'] 
                                    scores_combination.drop(columns='y_scores',inplace=True)    

                                param_combination_str = ' '.join([f'{param}={comb}' for param,comb in combination.items()])

                                param_combination_str += ' xval' if  'val' in file.stem else ' holdout'
                                
                                fig = plt.figure(figsize=(10,10))

                                sns.displot(scores_combination,x='logpost',hue='y_true',kind='kde',fill=True)
                                
                                plt.title(param_combination_str)
                                plt.tight_layout()
                                plt.savefig(Path(path_to_save,f'{param_combination_str.replace('=','_').replace(' ','_')}.png'))
                                plt.close(fig)