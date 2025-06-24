import pandas as pd
from pathlib import Path

project_name = 'MCI_classifier_unbalanced'

data_dir = Path(Path.home(), 'data', project_name) if 'Users/gp' in str(Path.home()) else Path(Path.home(), 'D:', 'CNC_Audio','gonza', 'data', project_name)

data = pd.read_csv(Path(data_dir,'data_matched_unbalanced_group.csv'))
valid_responses_data = pd.read_csv(Path(data_dir,'combine_correctas_sep.csv'))

merged_data = pd.merge(data, valid_responses_data, on='id')

merged_data.to_csv(Path(data_dir,'data_matched_unbalanced_group.csv'), index=False)