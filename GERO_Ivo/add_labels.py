import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','GERO_Ivo') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','GERO_Ivo')

all_labels = pd.read_csv(Path(data_dir,'covariates.csv'))[['id','sex','age','education']]

all_data_ = pd.read_csv(Path(data_dir,'all_data.csv')) 

all_data = pd.merge(all_labels,all_data_, on='id')

'''
all_data.dropna(subset=['id'], inplace=True)
#If 'AD_' in all_data['id'], then group is ADD:
all_data['group'] = all_data.apply(lambda x: 'ADD' if 'AD_' in x['id'] else x['group'], axis=1)
all_data['group'] = all_data.apply(lambda x: 'HC' if 'CTR_' in x['id'] else x['group'], axis=1)
'''

all_data.to_csv(Path(data_dir,'all_data.csv'), index=False)