from pathlib import Path
import pandas as pd

data_dir = Path(Path.home(), 'data', 'Proyecto_Ivo') if '/Users/gp/' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','Proyecto_Ivo')

all_data = pd.read_csv(Path(data_dir, 'data_matched_target.csv'))
nps_data = pd.read_csv(Path(data_dir, 'nps_data.csv'))
nps_data.drop(columns=['group'], inplace=True)
all_data = pd.merge(all_data,nps_data,on='id',how='left')
all_data.to_csv(Path(data_dir,'all_data.csv'))