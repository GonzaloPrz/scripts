import pandas as pd
from pathlib import Path
import numpy as np

project = 'MCI_classifier'

data_dir = Path(Path.home(),'data',project) if 'Users/gp' in str(Path.home()) else Path(Path.home(),'D:','data',project)

data = pd.read_excel(Path(data_dir,'features_fas_animales.xlsx'))
matched_ids = pd.read_csv(Path(data_dir,'matched_data.csv'))[['ID','DCL']]

data_matched = pd.merge(data,matched_ids,left_on='id',right_on='ID',how='inner')
data_matched['target'] = data_matched['DCL']
data_matched.drop(columns=['ID','DCL'],inplace=True)

data_matched.to_excel(Path(data_dir,'features_fas_animales_matched.xlsx'),index=False)

IDs_not_included = [id for id in matched_ids if id not in data['id'].values]