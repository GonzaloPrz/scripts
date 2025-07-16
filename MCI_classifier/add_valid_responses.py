import pandas as pd
from pathlib import Path

#data = pd.read_csv(Path(Path.home(),'data','MCI_classifier','data_matched_group.csv'))
data_unbalanced = pd.read_csv(Path(Path.home(),'data','MCI_classifier','data_matched_unbalanced_group.csv'))
valid_responses = pd.read_csv(Path(Path.home(),'data','MCI_classifier','combine_correctas_sep.csv'))

#data = pd.merge(data, valid_responses, on='id', how='left')
data_unbalanced = pd.merge(data_unbalanced, valid_responses, on='id', how='left')

#data.to_csv(Path(Path.home(),'data','MCI_classifier','data_matched_group.csv'), index=False)
data_unbalanced.to_csv(Path(Path.home(),'data','MCI_classifier','data_matched_unbalanced_group.csv'), index=False)