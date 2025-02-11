import pandas as pd
import json,itertools,pickle
from pathlib import Path
import numpy as np

early_fusion = True
project_name = "arequipa"
avoid_stats = ['median','min','max','kurtosis','skewness']
stat_folder = '_'.join(sorted(list(set(['mean','std','median','min','max','kurtosis','skewness']) - set(avoid_stats))))
scaler_name = 'StandardScaler'
hyp_opt = True
feature_selection = True
n_folds = 5
early_fusion = False

if n_folds == 0:
    kfold_folder = 'l2ocv'
elif n_folds == -1:
    kfold_folder = 'loocv'
else:
    kfold_folder = f'{n_folds}_folds'

main_config = json.load(open('main_config.json'))

data_file_test = main_config["data_file_test"][project_name]
single_dimensions = main_config["single_dimensions"][project_name]
y_labels = main_config["y_labels"][project_name]
tasks = main_config["tasks"][project_name]
problem_type = main_config["problem_type"][project_name]
id_col = main_config["id_col"][project_name]

data_dir = Path(Path.home(),"data",project_name) if "/Users/gp" in str(Path.home()) else Path("D:/CNC_Audio/gonza/data",project_name)
results_dir = Path(str(data_dir).replace("data","results"))

for y_label, task in itertools.product(y_labels, tasks):
    print(y_label)
    print(task)
    # Determine feature dimensions. For projects with a dictionary, pick based on the task.
    dimensions = []
    single_dims = single_dimensions
    if isinstance(single_dims, list) and early_fusion:
        for ndim in range(1, len(single_dims)+1):
            for dimension in itertools.combinations(single_dims, ndim):
                dimensions.append("__".join(dimension))
    else:
        dimensions = single_dims
    
    for dimension in dimensions:
        print(dimension)
        # Load dataset. Use CSV or Excel based on file extension.
        data_path = data_dir / (data_file_test if data_file_test.endswith(".csv") else data_file_test)
        data = pd.read_csv(data_path if data_file_test.endswith(".csv") else data_path.with_suffix(".csv"))
        
        # Identify feature columns (avoid stats and other unwanted columns)
        features = [col for col in data.columns if any(f"{x}__{y}__" in col 
                    for x,y in itertools.product(task.split("__"), dimension.split("__"))) 
                    and not isinstance(data.iloc[0][col], str)]
        # Select only the desired features along with the target and id
        data = data[features + [y_label, id_col]]
        data = data.dropna(subset=[y_label])
        
        # Separate features, target and ID.
        ID = data.pop(id_col)
        y = data.pop(y_label)

        path_to_save = Path(results_dir, task, dimension, scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '')

        with open(Path(path_to_save, 'X_test.pkl'), 'wb') as f:
            pickle.dump(data,f)
        with open(Path(path_to_save, 'y_test.pkl'), 'wb') as f:
            pickle.dump(y.values,f)
        with open(Path(path_to_save, 'IDs_test.pkl'), 'wb') as f:
            pickle.dump(ID.values,f)
