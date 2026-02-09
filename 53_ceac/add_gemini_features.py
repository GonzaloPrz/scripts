import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','53_ceac') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','53_ceac')

all_data = pd.read_csv(data_dir / 'all_data.csv')

gemini_features_anxiety = pd.read_csv(data_dir / 'anxiety_features.csv')
gemini_features_anxiety['id'] = gemini_features_anxiety['id'].apply(lambda x: f'CEAC_{x}')
gemini_features_burnout = pd.read_csv(data_dir / 'burnout_features.csv')
gemini_features_burnout['id'] = gemini_features_burnout['id'].apply(lambda x: f'CEAC_{x}')

all_data = all_data.merge(gemini_features_anxiety, on='id', how='left')
all_data = all_data.merge(gemini_features_burnout, on='id', how='left')

labels = pd.read_csv(data_dir / 'labels.csv')[['id','anxiety','burnout']]
labels['id'] = labels['id'].apply(lambda x: f'CEAC_{x}')

all_data = all_data.merge(labels, on='id', how='left')

all_data.to_csv(data_dir / 'all_data.csv', index=False)