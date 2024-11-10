import pandas as pd
from pathlib import Path

project_name = 'GeroApathy'

tasks = ['DiaTipico']
dimensions = ['opensmile','pitch','talking-intervals']

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

for task in tasks:
    all_features = pd.DataFrame()

    for dim in dimensions:
        data = pd.read_csv(Path(data_dir,f'{dim}_features_{task}.csv'))
        
        if all_features.empty:
            all_features = data
        else:
            all_features = pd.merge(all_features,data,on='id')

    all_features.to_csv(Path(data_dir,f'all_features_{task}.csv'),index=False)