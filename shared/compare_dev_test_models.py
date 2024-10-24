import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

project_name = 'tell_classifier'
l2ocv = False
scoring = 'norm_cross_entropy'

tasks = {'tell_classifier':['MOTOR-LIBRE'],
            'MCI_classifier':['fas','animales','fas__animales','grandmean'],
            'Proyecto_Ivo':['Animales','P','Animales__P','cog','brain','AAL','conn'],
            'GeroApathy':['Fugu']}

scaler_name = 'StandardScaler'

metrics_names = {'MCI_classifier':['roc_auc','accuracy'],
                 'tell_classifier':['roc_auc','accuracy'],
                    'Proyecto_Ivo':['roc_auc','accuracy'],
                    'GeroApathy':['r2','mean_squared_error','mean_absolute_error']}

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for task in tasks[project_name]:
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
    for dimension in dimensions:
        print(task,dimension)
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,'target','hyp_opt','feature_selection')
        random_seeds_test = [folder.name for folder in Path(path).iterdir() if folder.is_dir()]
    
        for random_seed_test in random_seeds_test:
            
            path_to_data = Path(path,random_seed_test)
            files = [file for file in path_to_data.iterdir() if file.is_file() and f'all_models_{scoring}' in file.stem]
            for file in files:
                best_classifiers = pd.read_csv(file)
                best_classifiers = best_classifiers.drop(columns=['Unnamed: 0'])
                path_to_figures = Path(path_to_data,'figures_models')
                path_to_figures.mkdir(parents=True,exist_ok=True)
                for metric in metrics_names[project_name]:
                    model_name = file.stem.split('_')[-2]
                    plt.figure(figsize=(12,8))

                    sns.scatterplot(data=best_classifiers,x=f'mean_{metric}_dev',y=f'mean_{metric}_test')
                    plt.xlabel(f'mean_{metric}_dev')
                    plt.ylabel(f'mean_{metric}_holdout')
                    plt.title(f'{model_name} dev vs test')
                    #Add y=x line to plot for reference
                    plt.plot([0, np.max((np.max(best_classifiers[f'mean_{metric}_dev']),np.max(best_classifiers[f'mean_{metric}_test'])))], 
                             [0, np.max((np.max(best_classifiers[f'mean_{metric}_dev']),np.max(best_classifiers[f'mean_{metric}_test'])))],
                            transform=plt.gca().transAxes, ls="--", c="red")
                    plt.savefig(Path(path_to_figures,f'mean_{metric}_{model_name}.png'))
                
                