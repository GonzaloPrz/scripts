import pandas as pd
from pathlib import Path

project_name = 'GERO_females'
feature_filename = 'predictions_dev.csv'
labels_filename = 'all_data.csv'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

features = pd.read_excel(Path(data_dir,feature_filename)) if 'xlsx' in feature_filename else pd.read_csv(Path(data_dir,feature_filename))

labels = pd.read_excel(Path(data_dir,labels_filename)) if 'xlsx' in labels_filename else pd.read_csv(Path(data_dir,labels_filename))

all_data = pd.merge(features,labels[['id','group','age','education']],on='id',how='inner')
all_data.drop_duplicates(subset=['id'],inplace=True)
all_data.to_csv(Path(data_dir,'predictions_dev_with_demographics.csv'),index=False)