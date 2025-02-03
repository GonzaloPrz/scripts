import pandas as pd
from pathlib import Path

project_name = 'ad_mci_hc'
feature_filename = 'features_data_gero.csv'
demographic = 'demographic_data_gero.csv'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

features = pd.read_excel(Path(data_dir,feature_filename)) if 'xlsx' in feature_filename else pd.read_csv(Path(data_dir,feature_filename))

demo = pd.read_excel(Path(data_dir,demographic)) if 'xlsx' in demographic else pd.read_csv(Path(data_dir,demographic))

all_data = pd.merge(features,demo[['id','group','sex','age','education']],on='id',how='outer')
all_data.drop_duplicates(subset=['id'],inplace=True)
all_data.to_csv(Path(data_dir,'all_data_gero.csv'),index=False)