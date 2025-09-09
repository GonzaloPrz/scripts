import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','53_ceac') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','53_ceac')

all_data = pd.DataFrame()

feature_sets = ['timing_segmented','pitch_segmented','sentiment_sentence','psycholinguistic','osv','granularity','verbosity','phq','gemini']

for feature_set in feature_sets:
    features = pd.read_csv(Path(data_dir,f'{feature_set}_features.csv'))
    
    if all(['CEAC' not in str(x) for x in features['id']]):
        features['id'] = features['id'].map(lambda x: f'CEAC_{x}')

    features = features.drop(columns=[col for col in features.columns if any (x in col for x in ['text','segments','timestamp','msg'])])
    if all_data.empty:
        all_data = features
    else:
        all_data = pd.merge(all_data,features,on='id',how='left')

labels = pd.read_csv(Path(data_dir,'labels.csv'))[['id','depression','anxiety']]
labels['id'] = labels['id'].map(lambda x: f'CEAC_{x}')
all_data = pd.merge(labels,all_data,on='id',how='left')
all_data = all_data[[ft for ft in all_data.columns if 'Unnamed' not in
    ft]]

all_data.to_csv(Path(data_dir,'all_data.csv'),index=None)