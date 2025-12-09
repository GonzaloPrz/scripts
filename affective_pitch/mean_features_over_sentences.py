import pandas as pd
from pathlib import Path
import numpy as np
import itertools

data_dir = Path(Path.home(),'data','affective_pitch') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')

labels = pd.read_csv(data_dir / 'transcripts_fugu_matched_group_sentiment.csv')

all_data = labels

for sentiment in ['ALL','POS','NEG','NEU']:
    pitch_data = pd.read_csv(data_dir / 'divided_audios' / sentiment / f'{sentiment}_energy_pitch_features.csv')
    pitch_data = labels.merge(pitch_data,on='id',how='inner')

    pitch_data.columns = [col.replace('Fugu_','Fugu__') for col in pitch_data.columns]

    features = np.unique([col.split('__')[3] for col in pitch_data.columns if col.startswith('Fugu__sentence')])

    mean_data = pd.DataFrame(columns=['id'] + [f'Fugu__{sentiment}__{feature}' for feature in features]
                             + [f'Fugu__{sentiment}__{feature}_{s}' for feature,s in itertools.product(features,['POS','NEG','NEU'])] if sentiment == 'ALL' else [])

    for feature in features:
        col_name = f'Fugu__{sentiment}__{feature}'
        
        columns = sorted([col for col in pitch_data.columns if feature in col])
        filtered_data = pitch_data[columns]
        if sentiment == 'ALL':
            r = 0
            for _, row in pitch_data.iterrows():
                filtered_data_ = row[columns].dropna().values
                if len(filtered_data_) != len(np.atleast_1d(np.array(row['pos_proba_norm'].replace('[','').replace(']','').split(),dtype=float))):
                    print(f"Skipping participant {row['id']} due to length mismatch")
                    continue
                mean_data.at[r,'id'] = row['id']
                mean_data.at[r, f'{col_name}_POS'] = np.average(filtered_data_,weights=np.atleast_1d(np.array(row['pos_proba_norm'].replace('[','').replace(']','').split(),dtype=float)))
                mean_data.at[r, f'{col_name}_NEG'] = np.average(filtered_data_,weights=np.atleast_1d(np.array(row['neg_proba_norm'].replace('[','').replace(']','').split(),dtype=float)))
                mean_data.at[r, f'{col_name}_NEU'] = np.average(filtered_data_,weights=np.atleast_1d(np.array(row['neu_proba_norm'].replace('[','').replace(']','').split(),dtype=float)))
                
                r += 1
        else:
            mean_data[col_name] = np.nanmean(filtered_data,axis=1)

            mean_data['id'] = pitch_data['id']
    mean_data = mean_data.merge(pitch_data['id'],on='id',how='inner').drop_duplicates(subset=['id'])
    
    mean_data.to_csv(Path(data_dir,f'mean_{sentiment}_pitch_features.csv'),index=False)

    all_data = pd.merge(all_data,mean_data,on='id',how='left')

all_data.to_csv(data_dir / 'mean_pitch_features.csv',index=False)