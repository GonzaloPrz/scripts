import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','53_ceac') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','53_ceac')

all_data = pd.read_csv(Path(data_dir,'all_data.csv'))

labels = pd.read_csv(Path(data_dir,'labels.csv'))[['id','burnout','sex','age']]
labels['id'] = labels['id'].map(lambda x: f'CEAC_{x}')
all_data = pd.merge(labels,all_data,on='id',how='left')
all_data = all_data[[ft for ft in all_data.columns if 'Unnamed' not in
    ft]]

all_data.to_csv(Path(data_dir,'all_data.csv'),index=None)