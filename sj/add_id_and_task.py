import pandas as pd
from pathlib import Path

data = pd.read_csv(Path(Path.home(),'data','sj','features_all.csv'))
data['filename'] = data['id']
data['id'] = data['filename'].apply(lambda x: '_'.join(x.split('_')[:2]))
data['task'] = data['filename'].apply(lambda x: x.split('_')[2])
data['task'] = data['task'].apply(lambda x: x.replace('Preg','Pre').replace('pre','Pre').replace('post','Post').replace('2Pos1','2Post1').replace('2Pos2','2Post2'))

data.to_csv(Path(Path.home(),'data','sj','all_features.csv'), index=False)