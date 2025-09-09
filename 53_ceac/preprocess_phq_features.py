import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','53_ceac') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','53_ceac')

features = pd.read_csv(Path(data_dir,'phq9_data.csv'))

features['id'] = features['file_path'].map(lambda x: f'CEAC_{x.split("\\")[-2].split("_")[0]}')
features['task'] = features['file_name'].map(lambda x: x.replace('.txt',''))

features = features.drop(columns=[col for col in features.columns if any (x in col for x in ['file_path','file_name','text','segments','timestamp','msg'])])

tasks = features['task'].unique()

all_data = pd.DataFrame()

for task in tasks:
    data = features[features['task']==task].drop(columns=['task'])
    data = data.add_prefix(f'{task}__phq9__')
    data = data.rename(columns={f'{task}__phq9__id':'id'})
    if all_data.empty:
        all_data = data
    else:
        all_data = pd.merge(all_data,data,on='id',how='left')

all_data.to_csv(Path(data_dir,'phq_features.csv'),index=None)
    