import pandas as pd
from pathlib import Path
import numpy as np
import itertools

data_dir = Path(Path.home(),'data','affective_pitch') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')

labels = pd.read_csv(data_dir / 'transcripts_fugu_matched_group.csv')[['id','sex','age','site','group']]

labels['group'] = labels['group']

all_data = labels

for sentiment in ['POS','NEG','NEU']:
    pitch_data = pd.read_csv(data_dir / f'{sentiment}_energy_pitch_features.csv')

    features = np.unique([col.split('__')[3] for col in pitch_data.columns if col.startswith('Fugu')])

    mean_data = pd.DataFrame(columns=['id'] + [f'Fugu__{sentiment}__{feature}' for feature in features])

    for feature in features:
        col_name = f'Fugu__{sentiment}__{feature}'
        
        columns = [col for col in pitch_data.columns if feature in col]
        filtered_data = pitch_data[columns]
        mean_data[col_name] = filtered_data.mean(axis=1)
    mean_data['id'] = pitch_data['id']
    
    mean_data.to_csv(Path(data_dir,f'mean_{sentiment}_pitch_features.csv'),index=False)

    all_data = pd.merge(all_data,mean_data,on='id',how='left')

all_data.to_csv(data_dir / 'mean_pitch_features.csv',index=False)
