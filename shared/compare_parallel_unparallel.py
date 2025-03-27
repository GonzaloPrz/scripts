import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import json, os
import itertools,pickle

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
calibrate = config["calibrate"]
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
calibrate = bool(config["calibrate"])
parallel = bool(config["parallel"])

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
models = main_config["models"][project_name]

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

problem_type = main_config['problem_type'][project_name]

metrics_names = main_config['metrics_names'][problem_type]

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

for task,model,y_label,scoring in itertools.product(tasks,models,y_labels,[scoring_metrics]):    
    
    dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

    for dimension in dimensions:
        path = Path(results_dir,task,dimension,scaler_name, kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else '','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type == 'reg' else '','shuffle' if shuffle_labels else '')
        
        if not path.exists():  
            continue

        random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
        if len(random_seeds) == 0:
            random_seeds = [folder.name for folder in path.parent.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
       
        if len(random_seeds) == 0:
            random_seeds = ['']
        
        for random_seed in random_seeds:
            outputs_parallel = pickle.load(open(Path(path,"parallel",random_seed,f'cal_outputs_{model}.pkl' if calibrate else f'outputs_{model}.pkl'),'rb')) 
            
            y_dev_parallel = pickle.load(open(Path(path,"parallel",random_seed,'y_dev.pkl'),'rb')).astype(int)
            
            IDs_dev_parallel = pickle.load(open(Path(path,"parallel",random_seed,'IDs_dev.pkl'),'rb'))

            outputs = pickle.load(open(Path(path,random_seed,f'cal_outputs_{model}.pkl' if calibrate else f'outputs_{model}.pkl'),'rb')) 
            
            y_dev = pickle.load(open(Path(path,random_seed,'y_dev.pkl'),'rb')).astype(int)
            
            IDs_dev = pickle.load(open(Path(path,random_seed,'IDs_dev.pkl'),'rb'))

            all_models_parallel = pd.read_csv(Path(path,"parallel",random_seed,f'all_models_{model}_dev_bca_calibrated.csv')) if calibrate else pd.read_csv(Path(path,"parallel",random_seed,f'all_models_{model}_dev_bca.csv'))
            all_models = pd.read_csv(Path(path,random_seed,f'all_models_{model}_dev_bca_calibrated.csv')) if calibrate else pd.read_csv(Path(path,random_seed,f'all_models_{model}_dev_bca.csv'))

