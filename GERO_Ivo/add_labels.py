import pandas as pd
from pathlib import Path

labels_GERO = pd.read_csv(Path(Path.home(),'data','GERO_Ivo','labels_GERO.csv'))[['id','group','sex','age','education']]
labels_GERO['group'] = labels_GERO['group'].map({0:'SCD',1:'MCI'})

labels_redlat_fondecyt_cetram = pd.read_csv(Path(Path.home(),'data','GERO_Ivo','labels_redlat_fondecyt_cetram.csv'))[['id','group','sex','age','education']]
labels_redlat_fondecyt_cetram['group'] = labels_redlat_fondecyt_cetram['group'].map({'Alzheimer':'ADD','Control':'HC'})

all_labels = pd.concat([labels_GERO, labels_redlat_fondecyt_cetram])
nps = pd.read_csv(Path(Path.home(),'data','GERO_Ivo','all_data.csv'))[['id','TIV','MoCA_Total_Boni_3']]
all_labels = all_labels.merge(nps, on='id', how='left')

all_labels.to_csv(Path(Path.home(),'data','GERO_Ivo','all_labels.csv'), index=False)

all_data_ = pd.read_csv(Path(Path.home(),'data','GERO_Ivo','all_data.csv')) 

all_data = pd.merge(all_labels,all_data_, on='id', how='right')

all_data.dropna(subset=['id'], inplace=True)
#If 'AD_' in all_data['id'], then group is ADD:
all_data['group'] = all_data.apply(lambda x: 'ADD' if 'AD_' in x['id'] else x['group'], axis=1)
all_data['group'] = all_data.apply(lambda x: 'HC' if 'CTR_' in x['id'] else x['group'], axis=1)

all_data.drop(columns=['group_x','group_y'],inplace=True)
all_data.to_csv(Path(Path.home(),'data','GERO_Ivo','all_data.csv'), index=False)