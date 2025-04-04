import pandas as pd 
from pathlib import Path

features_text = pd.read_csv(Path(Path.home(),'data','sj','nlp_features.csv'))
talking_intervals_features = pd.read_csv(Path(Path.home(),'data','sj','talking_intervals_features.csv'))
pitch_analysis_features = pd.read_csv(Path(Path.home(),'data','sj','pitch_analysis_features.csv'))
features_audio = pd.merge(talking_intervals_features, pitch_analysis_features, on='id', how='outer')

features = pd.merge(features_text, features_audio, on='id', how='inner')
data1 = pd.read_csv(Path(Path.home(),'data','sj','data_pre_post1.csv'))
data2 = pd.read_csv(Path(Path.home(),'data','sj','data_pre_post2.csv'))

features_Pre1 = features[['id'] + [col for col in features.columns if 'Pre1' in col]]
features_Pre1['task'] = 'Pre1'
features_Pre1 = features_Pre1.rename(columns=lambda x: x.replace('Pre1__', ''))
features_Pre1['time'] = 0

features_Pre2 = features[['id'] + [col for col in features.columns if 'Pre2__' in col]]
features_Pre2['task'] = 'Pre2'
features_Pre2 = features_Pre2.rename(columns=lambda x: x.replace('Pre2__', ''))
features_Pre2['time'] = 0

features_Post1 = features[['id'] + [col for col in features.columns if 'Post1__' in col]]
features_Post1['task'] = 'Post1'
features_Post1 = features_Post1.rename(columns=lambda x: x.replace('Post1__', ''))
features_Post1['time'] = 1

features_Post2 = features[['id'] + [col for col in features.columns if 'Post2__' in col]]
features_Post2['task'] = 'Post2'
features_Post2 = features_Post2.rename(columns=lambda x: x.replace('Post2__', ''))
features_Post2['time'] = 1

features_1 = pd.concat([features_Pre1,features_Post1], axis=0)

features_2 = pd.concat([features_Pre2,features_Post2], axis=0)

features_1.to_csv(Path(Path.home(),'data','sj','all_data_1.csv'), index=False)
features_2.to_csv(Path(Path.home(),'data','sj','all_data_2.csv'), index=False)