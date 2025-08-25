import pandas as pd
from pathlib import Path

conditions = ['has_depression','has_anxiety']

data_dir = Path(Path.home(),'data','53_ceac') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','53_ceac')

all_data = pd.read_csv(Path(data_dir,'features_CEAC.csv'))

for condition in conditions:
    labels = pd.read_csv(Path(data_dir,'labels.csv'))[['id'] + [condition,'sex','age']]
    labels['id'] = labels['id'].astype(str).tolist()
    
    data = pd.merge(labels,all_data,on='id',how='left')
    data = data[[ft for ft in data.columns if 'Unnamed' not in ft]]
    data.to_csv(Path(data_dir,f'all_data_{condition}.csv'))