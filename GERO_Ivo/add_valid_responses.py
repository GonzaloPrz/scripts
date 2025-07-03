import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','GERO_Ivo') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','GERO_Ivo')

all_data = pd.read_csv(Path(data_dir,'all_data.csv'))

valid_responses = pd.read_excel(Path(data_dir,'valid_responses_fas_animales.xlsx'))

all_data = pd.merge(all_data, valid_responses, on='id', how='left')

all_data.to_csv(Path(data_dir,'all_data.csv'), index=False)