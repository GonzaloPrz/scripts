import pandas as pd
from pathlib import Path

path_to_data = Path(Path.home(),'data','arequipa')

feature_files = list(path_to_data.glob('*_features.csv'))

merged_data = pd.read_csv(path_to_data / 'demographic_data.csv')[['id','group','sex','age','education']]

for feature_file in feature_files:
    dimension = feature_file.stem.split('_')[1]
    df = pd.read_csv(feature_file)
    df['id'] = df['filename'].apply(lambda x: x.split('_')[1])
    df.drop(columns=['filename'], inplace=True)
    feature_names = [col for col in df.columns if not isinstance(df[col][0], str)]
    for feature_name in feature_names:
        df.rename(columns={feature_name: f'fugu__{dimension}__{feature_name}'}, inplace=True)
    
    df = df[['id'] + [f'fugu__{dimension}__{feature_name}' for feature_name in feature_names]]

    merged_data = pd.merge(merged_data, df, on='id', how='outer')

merged_data.to_csv(path_to_data / 'all_data.csv', index=False)