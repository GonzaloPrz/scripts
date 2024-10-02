import pandas as pd
from pathlib import Path
import numpy as np
import re

project = 'MCI_classifier'

data_dir = Path(Path.home(),'data',project) if 'Users/gp' in str(Path.home()) else Path(Path.home(),'D:','gonza','data',project)

data = pd.read_excel(Path(data_dir,'features_mod.xlsx'))

features = [re.sub('letra_f_|letra_a_|letra_s_|animales_','',col) for col in data.columns if col != 'id']

for feature in features:
    try:
        data[f'fas_{feature}'] = data[[f'letra_f_{feature}',f'letra_a_{feature}',f'letra_s_{feature}']].mean(axis=1)
        data[f'grandmean_{feature}'] = data[[f'letra_f_{feature}',f'letra_a_{feature}',f'letra_s_{feature}',f'animales_{feature}']].mean(axis=1)
    except:
        print(f'Error with {feature}')
data.to_excel(Path(data_dir,'features_fas_animales.xlsx'),index=False)