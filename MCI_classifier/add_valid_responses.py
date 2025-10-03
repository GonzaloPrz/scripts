import pandas as pd
from pathlib import Path

#data = pd.read_csv(Path(Path.home(),'data','MCI_classifier','data_matched_group.csv'))
data_unbalanced = pd.read_csv(Path(Path.home(),'data','MCI_classifier_unbalanced','data_matched_unbalanced_group.csv'))
dem_data = pd.read_csv(Path(Path.home(),'data','MCI_classifier_unbalanced','nps_relevant_data.csv'))[['id','dem__dem__sex','dem__dem__age','dem__dem__education']]

#data = pd.merge(data, valid_responses, on='id', how='left')
data_unbalanced = pd.merge(data_unbalanced, dem_data, on='id', how='left')

#data.to_csv(Path(Path.home(),'data','MCI_classifier','data_matched_group.csv'), index=False)
data_unbalanced.to_csv(Path(Path.home(),'data','MCI_classifier_unbalanced','data_matched_unbalanced_group.csv'), index=False)