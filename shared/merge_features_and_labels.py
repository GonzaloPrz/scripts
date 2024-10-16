import pandas as pd
from pathlib import Path

project_name = 'MCI_classifier'
feature_filename = 'features_fas_animales.xlsx'
labels_filename = 'matched_data.csv'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

features = pd.read_excel(Path(data_dir,feature_filename)) if 'xlsx' in feature_filename else pd.read_csv(Path(data_dir,feature_filename))
labels = pd.read_excel(Path(data_dir,labels_filename)) if 'xlsx' in labels_filename else pd.read_csv(Path(data_dir,labels_filename))[['id','target']]

features = features.merge(labels,on='id')
features.to_csv(Path(data_dir,'features_data.csv'),index=False)