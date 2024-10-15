import pandas as pd
from pathlib import Path
import numpy as np
import re

project_name = 'GERO_Ivo'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','data','gonza',project_name)

data = pd.read_excel(Path(data_dir,'speech_timing_features.xlsx'))

features = [re.sub('letra_f_|letra_a_|letra_s_|animales_','',col) for col in data.columns if col != 'Codigo']

for feature in features:
    if isinstance(data[f'letra_f_{feature}'][0],str):
        continue
    data[f'fas_{feature}'] = data[[f'letra_f_{feature}',f'letra_a_{feature}',f'letra_s_{feature}']].mean(axis=1)
    data[f'grandmean_{feature}'] = data[[f'letra_f_{feature}',f'letra_a_{feature}',f'letra_s_{feature}',f'animales_{feature}']].mean(axis=1)

data.to_excel(Path(data_dir,'speech_timing_fas_animales.xlsx'),index=False)