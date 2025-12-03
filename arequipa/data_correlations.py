import pandas as pd
from pathlib import Path

base_dir = Path(Path.home(),'data','arequipa') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','arequipa')

match_data_imagenes = pd.read_csv(Path(base_dir,'data_imagenes_matched_group.csv'))['id']

data = pd.read_csv(Path(base_dir,'data_matched_group.csv'))

filtered_data = data[data['id'].isin(match_data_imagenes)]

features = pd.read_csv(Path(base_dir,'feature_importance_lr.csv'))

filtered_data = filtered_data[features['feature'].tolist()[:10] + ['id']]

filtered_data.to_csv(Path(base_dir,'data_top10_features_matched_group.csv'), index=False)

