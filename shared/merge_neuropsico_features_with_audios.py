import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data')

Audios_T1 = pd.read_csv(data_dir / 'Audios_GERO_T1.csv')[['id','target','agradable','desagradable','sex','age','education','handedness']]

neuropsico_features = pd.read_csv(data_dir / 'nps_data_GERO.csv')

relevant_features = ['id','target','agradable','desagradable','sex','age','education','handedness'] + [f for f in neuropsico_features.columns if any(x in f.lower() for x in ['moca','mmse','dass','aes_','tmt','depression'])]

all_data = pd.merge(Audios_T1, neuropsico_features, on='id', how='inner')
all_data = all_data.loc[all_data.target == 1,relevant_features]
all_data = all_data[all_data.agradable == 1]
all_data = all_data[all_data.desagradable == 1]

all_data.reset_index(drop=True, inplace=True)

all_data['sex'] = all_data['sex'].map({2:'F',1:'M'})
all_data['handedness'] = all_data['handedness'].map({1:'R',2:'L'})
all_data['target'] = all_data['target'].map({1:'MCI',0:'HC'})

for column in all_data.columns:
    if isinstance(all_data.loc[0,column],str):
        continue
    all_data[column] = all_data[column].astype(int)
    #Convert negative values to NaN
    all_data.loc[all_data[column] < 0,column] = None

all_data.to_csv(data_dir / 'Audios_GERO_T1_MCI_agradable_desagradable_with_nps.csv', index=False)


