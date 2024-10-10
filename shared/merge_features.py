import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','GeroApathy') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','GeroApathy')

opensmile_data = pd.read_csv(Path(data_dir,'opensmile_features.csv'))
pitch_data = pd.read_csv(Path(data_dir,'pitch_features.csv'))
talking_intervals_data = pd.read_csv(Path(data_dir,'talking_intervals_features.csv'))

all_features = pd.merge(opensmile_data,talking_intervals_data,on='id').sort_values('id')

all_features = pd.merge(all_features,pitch_data,on='id').sort_values('id')

all_features.to_csv(Path(data_dir,'features_data.csv'),index=False)
