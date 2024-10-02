import pandas as pd
from pathlib import Path
import numpy as np
import re

data = pd.read_excel(Path(Path(__file__).parent,'data','valid_responses_mod.xlsx'))

features = [re.sub('letra_f_|letra_a_|letra_s_|animales_','',col) for col in data.columns if col != 'Codigo']

for feature in features:
    data[f'fas_{feature}'] = data[[f'letra_f_{feature}',f'letra_a_{feature}',f'letra_s_{feature}']].mean(axis=1)
    data[f'grandmean_{feature}'] = data[[f'letra_f_{feature}',f'letra_a_{feature}',f'letra_s_{feature}',f'animales_{feature}']].mean(axis=1)

data.to_excel(Path(Path(__file__).parent,'data','valid_responses_fas_animales.xlsx'),index=False)