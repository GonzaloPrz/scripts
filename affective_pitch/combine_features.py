import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','affective_pitch') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')

feature_files = [file.name for file in data_dir.iterdir() if file.name.endswith('_features.csv')]

all_data = pd.DataFrame()

for file in feature_files:

    df = pd.read_csv(Path(data_dir,file))
    if all_data.empty:
        all_data = df
    else:
        all_data = pd.merge(all_data,df,on='id',how='inner')

labels = pd.read_csv(Path(data_dir,'matched_ids.csv'))[['id','group','site','age','sex']]

all_data = pd.merge(labels,all_data,on='id',how='inner')

all_data.to_csv(Path(data_dir,'all_data.csv'))