import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(), 'data', 'Proyecto_Ivo') if '/Users/gp/' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','Proyecto_Ivo')
data = pd.read_csv(data_dir / 'all_data.csv')
data = data[[col for col in data.columns if 'conn' not in col]]

data_networks = pd.read_csv(data_dir / 'connectivity_networks.csv')
data_networks.drop(columns=['target'], inplace=True)

data_selected_areas = pd.read_csv(data_dir / 'connectivity_selected_areas.csv')
data_selected_areas.drop(columns=['target'], inplace=True)

all_data = pd.merge(data, data_networks, on='id',how='outer')
all_data = pd.merge(all_data, data_selected_areas, on='id',how='outer')

all_data.dropna(axis=0,subset=['target'],inplace=True)

all_data.to_csv(data_dir / 'all_data.csv', index=False)

