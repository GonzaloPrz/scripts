import pandas as pd
from pathlib import Path
import numpy as np
import itertools

data_dir = Path(Path.home(),'data','affective_pitch') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')

labels = pd.read_csv(data_dir / 'transcripts_fugu_matched_group_sentiment.csv')

all_data = labels

for sentiment in ['ALL','POS','NEG','NEU']:
    pitch_data = pd.read_csv(data_dir / f'{sentiment}_energy_pitch_features.csv')

    features = np.unique([col.split('__')[3] for col in pitch_data.columns if col.startswith('Fugu')])

    mean_data = pd.DataFrame(columns=['id'] + [f'Fugu__{sentiment}__{feature}' for feature in features]
                             + [f'Fugu__{sentiment}__{feature}_{s}' for feature,s in itertools.product(features,['POS','NEG','NEU'])] if sentiment == 'ALL' else [])

    for feature in features:
        col_name = f'Fugu__{sentiment}__{feature}'
        
        columns = [col for col in pitch_data.columns if feature in col]
        filtered_data = pitch_data[columns].values
        if sentiment == 'ALL':
            for r in range(filtered_data.shape[0]):
                filtered_data_ = pitch_data.loc[r,columns].dropna().values
                if len(filtered_data_) != len(np.atleast_1d(np.array(labels['pos_proba_norm'][r].replace('[','').replace(']','').split(),dtype=float))):
                    print(f"Skipping row {r} due to length mismatch")
                    continue
                mean_data.at[r, f'{col_name}_POS'] = np.average(filtered_data_,weights=np.atleast_1d(np.array(labels['pos_proba_norm'][r].replace('[','').replace(']','').split(),dtype=float)))
                mean_data.at[r, f'{col_name}_NEG'] = np.average(filtered_data_,weights=np.atleast_1d(np.array(labels['neg_proba_norm'][r].replace('[','').replace(']','').split(),dtype=float)))
                mean_data.at[r, f'{col_name}_NEU'] = np.average(filtered_data_,weights=np.atleast_1d(np.array(labels['neu_proba_norm'][r].replace('[','').replace(']','').split(),dtype=float)))
        else:
            mean_data[col_name] = np.average(filtered_data,axis=1,weights=None)

    mean_data['id'] = pitch_data['id']
    
    mean_data.to_csv(Path(data_dir,f'mean_{sentiment}_pitch_features.csv'),index=False)

    all_data = pd.merge(all_data,mean_data,on='id',how='left')

all_data.to_csv(data_dir / 'mean_pitch_features.csv',index=False)
