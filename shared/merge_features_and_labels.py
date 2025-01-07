import pandas as pd
from pathlib import Path

project_name = 'MPLS'
feature_filename = 'transformed_features_data_MPLS_cat.csv'
labels_filename = 'nps_data.csv'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

features = pd.read_excel(Path(data_dir,feature_filename)) if 'xlsx' in feature_filename else pd.read_csv(Path(data_dir,feature_filename))

features['language'] = 'cat'

all_data_esp = pd.read_csv(Path(data_dir,'all_data_esp.csv'))
all_data_esp['language'] = 'esp'

all_data = pd.concat([features,all_data_esp])

labels = pd.read_excel(Path(data_dir,labels_filename)) if 'xlsx' in labels_filename else pd.read_csv(Path(data_dir,labels_filename))

all_data = pd.merge(all_data,labels,on='id',how='outer')

all_data.to_csv(Path(data_dir,'all_data.csv'),index=False)