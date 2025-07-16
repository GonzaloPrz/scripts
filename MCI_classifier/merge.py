import pandas as pd
from pathlib import Path

nps_data = pd.read_csv(Path(Path.home(),'data','MCI_classifier','nps_GERO_completa.csv'))
nps_data = nps_data[['id'] + [col for col in nps_data.columns if col.startswith('nps__')]]

biomarkers = pd.read_excel(Path(Path.home(),'data','MCI_classifier','biomarkers_data.xlsx'))

biomarkers.columns = ['id'] + [f"bio__bio__{col}" for col in biomarkers.columns[1:]]

all_data = pd.merge(nps_data,biomarkers,on='id',how='left')

matched_data = pd.read_csv(Path(Path.home(),'data','MCI_classifier','data_matched_group.csv'))
matched_unbalanced_data = pd.read_csv(Path(Path.home(),'data','MCI_classifier','data_matched_unbalanced_group.csv'))[['id','group']]

data_matched_group = pd.merge(matched_data,all_data,on='id',how='inner')
data_matched_unbalanced_group = pd.merge(matched_unbalanced_data,all_data,on='id',how='inner')

data_matched_group.to_csv(Path(Path.home(),'data','MCI_classifier','data_matched_group.csv'), index=False)
data_matched_unbalanced_group.to_csv(Path(Path.home(),'data','MCI_classifier','data_matched_unbalanced_group.csv'), index=False)
