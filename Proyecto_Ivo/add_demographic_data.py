from pathlib import Path
import pandas as pd

data_dir = Path(Path.home(),'data','Proyecto_Ivo') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','Proyecto_Ivo')

all_data = pd.read_csv(Path(data_dir,'all_data.csv'))
dem_data = pd.read_csv(Path(data_dir,'nps_data.csv'))[['id','sex','age','education','handedness']]
