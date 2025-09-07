import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','MCI_classifier_unbalanced') if 'Users/gp' in str(Path.home()) else Path(Path.home(),'D:','CNC_Audio','gonza','data','MCI_classifier_unbalanced')

all_data = pd.read_csv(Path(data_dir,'nps_GERO_completa.csv'))[['id','moca_total']]
all_data.columns = ['id'] + [f"nps__nps__{col}" for col in all_data.columns[1:]]
#nps_data = nps_data[['id'] + [col for col in nps_data.columns if col.startswith('nps__')]]

#biomarkers = pd.read_excel(Path(data_dir,'biomarkers_data.xlsx'))

#biomarkers.columns = ['id'] + [f"bio__bio__{col}" for col in biomarkers.columns[1:]]

#all_data = pd.merge(nps_data,biomarkers,on='id',how='left')

matched_data = pd.read_csv(Path(data_dir,'data_matched_group.csv'))
matched_data.drop(columns=['group'],inplace=True)
demographic_data = pd.read_csv(Path(data_dir,'nps_relevant_data.csv'))

matched_unbalanced_data = pd.read_csv(Path(data_dir,'data_matched_unbalanced_group.csv'))
matched_unbalanced_data.drop(columns=['group'],inplace=True)

data_matched_group = pd.merge(matched_data,demographic_data,on='id',how='inner')
data_matched_unbalanced_group = pd.merge(matched_unbalanced_data,demographic_data,on='id',how='inner')

data_matched_group.to_csv(Path(data_dir,'data_matched_group.csv'), index=False)
data_matched_unbalanced_group.to_csv(Path(data_dir,'data_matched_unbalanced_group.csv'), index=False)
