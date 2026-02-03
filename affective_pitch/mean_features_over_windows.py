import pandas as pd
from pathlib import Path
import numpy as np
import itertools

data_dir = Path(Path.home(),'data','affective_pitch') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')

segmentations = ['windows']

feature_sets = ["embeddings",'energy_pitch','timing',]

for segmentation in segmentations:
    all_data = pd.DataFrame()

    for feature_set in feature_sets:
        print(segmentation,feature_set)
        for sentiment in ['ALL','POS','NEG','NEU']:
            print(sentiment)
            labels = pd.read_csv(data_dir / f'transcripts_fugu_matched_group_sentiment_{segmentation}.csv')
            
            labels['id'] = labels['id'].map(lambda x: x.replace('REDLAT_','').replace('_Fugu',''))
            data = pd.read_csv(data_dir / 'divided_audios' / segmentation / sentiment / f'{sentiment}_{feature_set}_features_{segmentation}.csv')
            data['id'] = data['id'].map(lambda x: x.split('_')[0] if '.wav' in x else x)
            data = labels.merge(data,on='id',how='inner')

            data.columns = [col.replace('Fugu_','Fugu__').replace('_pysent','').replace('___','__') for col in data.columns]

            features = np.unique([col.split('__')[-1] for col in data.columns if col.startswith(f'Fugu__{segmentation[:-1]}') and all(x not in col for x in ['msg','query'])])

            mean_data = pd.DataFrame(columns=['id'] + [f'Fugu__{sentiment}__{feature}' for feature in features] if sentiment != 'ALL' else []
                                    + [f'Fugu__{sentiment}__{feature}_{s}' for feature,s in itertools.product(features,['POS','NEG','NEU'])] if sentiment == 'ALL' else [])

            for feature in features:
                col_name = f'Fugu__{sentiment}__{feature}'
                
                columns = sorted([col for col in data.columns if feature in col])
                filtered_data = data[columns]
                if sentiment == 'ALL':
                    r = 0
                    for _, row in data.iterrows():
                        filtered_data_ = row[columns].dropna().values
                        
                        if any(isinstance(x,str) for x in filtered_data_):
                            continue

                        if len(filtered_data_) != len(np.atleast_1d(np.fromstring(row['pos_proba_norm'].replace(',','').strip('[]'),sep = ' ',dtype=float))):
                            #print(f"Skipping participant {row['id']} due to length mismatch")
                            continue
                        mean_data.at[r,'id'] = row['id']
                        mean_data.at[r, f'{col_name}_POS'] = np.average(filtered_data_,weights=np.atleast_1d(np.fromstring(row['pos_proba_norm'].replace(',','').strip('[]'),sep = ' ',dtype=float)))
                        mean_data.at[r, f'{col_name}_NEG'] = np.average(filtered_data_,weights=np.atleast_1d(np.fromstring(row['neg_proba_norm'].replace(',','').strip('[]'),sep = ' ',dtype=float)))
                        mean_data.at[r, f'{col_name}_NEU'] = np.average(filtered_data_,weights=np.atleast_1d(np.fromstring(row['neu_proba_norm'].replace(',','').strip('[]'),sep = ' ',dtype=float)))
                        
                        r += 1
                else:
                    mean_data['id'] = data['id']
                    try:
                        mean_data[col_name] = np.nanmean(filtered_data,axis=1)
                    except:
                        continue
                    
            mean_data = mean_data.merge(data['id'],on='id',how='inner').drop_duplicates(subset=['id'])
            
            mean_data.to_csv(Path(data_dir,f'mean_{sentiment}_{feature_set}_{segmentation}.csv'),index=False)
            
            if all_data.empty:
                all_data = mean_data
            else:
                all_data = pd.merge(all_data,mean_data,on='id',how='left')

    all_data = all_data.merge(labels,on='id',how='left')
    all_data.to_csv(data_dir / f'mean_{segmentation}.csv',index=False)