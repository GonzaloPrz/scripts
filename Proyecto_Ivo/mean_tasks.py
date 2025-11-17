import pandas as pd
from pathlib import Path
import numpy as np

data_dir = Path(Path.home(),'data','Proyecto_Ivo') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','Proyecto_Ivo')

all_data = pd.read_csv(Path(data_dir,'all_data.csv'))

speech_features = np.unique([col.replace('Animales__','') for col in all_data.columns if 'Animales' in col])

mean_data = pd.DataFrame()

for feature in speech_features:
    mean_data[f'grandmean__{feature}'] = all_data[[f'Animales__{feature}',f'P__{feature}']].mean(axis=1)
mean_data['id'] = all_data['id']

all_data = pd.merge(all_data, mean_data, on='id')
all_data.to_csv(Path(data_dir,'all_data.csv'), index=False)
