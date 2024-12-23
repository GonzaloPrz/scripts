from pathlib import Path
import sys,tqdm,itertools
import pandas as pd
from warnings import filterwarnings

filterwarnings('ignore')

dimensions = ['pitch','talking-intervals']
tasks = ['agradable']
project_name = 'GeroApathy'
subsets = ['pos','neg','neu','notneu']

all_data = pd.DataFrame()

for dimension,task,subset in itertools.product(dimensions,tasks,subsets):
    sys.path.append(str(Path(Path.home(),'local_feature_extraction','audio_features',f'{dimension}_analysis')))

    from app import *

    data_dir = Path(Path.home(),'data',project_name,f'{task}_audios','denoised')

    extracted_features = pd.DataFrame(columns=['id'])
    for file in tqdm.tqdm(Path(data_dir,subset) .glob(f'*_{subset}.wav')):
        features_dict = main(str(file))
        
        try:
            features = features_dict['data'].copy()
        except:
            print(f'Error extracting features from {file.stem}')
            continue
        features.update(features_dict['scores'])

        features['id'] = file.stem.split('__')[0].replace('T1_','')

        if extracted_features.empty:
            extracted_features = pd.DataFrame(features, index=[0])
        else:
            extracted_features.loc[len(extracted_features)] = features

    extracted_features.columns = [f'{task}_{dimension}__{col}_{subset}' for col in extracted_features.columns if col != 'id' ] + ['id']

    if all_data.empty:
        all_data = extracted_features
    else:
        all_data = pd.merge(all_data,extracted_features,on='id',how='outer')

    extracted_features.to_csv(Path(Path.home(),'data',project_name,f'{dimension}_features_{subset}.csv'),index=False)

all_data.to_csv(Path(Path.home(),'data',project_name,f'all_features_pos_neg_neu_notneu.csv'),index=False)

