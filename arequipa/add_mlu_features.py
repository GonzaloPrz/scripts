import pandas as pd
from pathlib import Path

path_to_data = Path(Path.home(),'data','arequipa')
all_data = pd.read_csv(Path(path_to_data, 'all_data.csv'))
df = pd.read_csv(Path(path_to_data,'arequipa__semantic_distance.csv'))

df['id'] = df['filename'].apply(lambda x: x.split('_')[0].split('/')[-1].lower().replace(' ',''))
df['task'] = df['filename'].apply(lambda x: x.split('__')[-1].replace('.wav','').replace('.txt',''))
df.drop(columns=['filename'], inplace=True)
tasks  = df['task'].unique()
for task in tasks:
    df_ = df[df['task'] == task]
    feature_names = [col for col in df_.columns if not isinstance(df[col][0], str)]
    for feature_name in feature_names:
        df_.rename(columns={feature_name: f'{task}__semantic_distance__{feature_name}'}, inplace=True)
    
    df_ = df_[['id'] + [f'{task}__semantic_distance__{feature_name}' for feature_name in feature_names]]
    df_ = df_.drop_duplicates(subset='id')
    all_data = pd.merge(all_data, df_, on='id', how='outer')

all_data.to_csv(Path(path_to_data, 'all_data.csv'), index=False)
