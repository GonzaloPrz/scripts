import pandas as pd
from pathlib import Path

project_name = 'MCI_classifier'

data_dir = Path(Path.home(), 'data', project_name) if 'Users/gp' in str(Path.home()) else Path(Path.home(), 'D:', 'CNC_Audio','gonza', 'data', project_name)

data = pd.read_csv(Path(data_dir,'data_matched_group.csv'))
features = pd.read_csv(Path(data_dir,'features.csv'))

merged_data = pd.merge(data, features, on='id')

merged_data.to_csv(Path(data_dir,'data_matched_group.csv'), index=False)