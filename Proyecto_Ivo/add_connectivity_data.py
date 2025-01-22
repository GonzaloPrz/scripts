import pandas as pd
from pathlib import Path

data = pd.read_csv(Path(Path.home(),'data','Proyecto_Ivo','data_total.csv'))
data = data[[col for col in data.columns if 'conn' not in col]]

data_networks = pd.read_csv(Path(Path.home(),'data','Proyecto_Ivo','connectivity_networks.csv'))
data_networks.drop(columns=['target'], inplace=True)

data_selected_areas = pd.read_csv(Path(Path.home(),'data','Proyecto_Ivo','connectivity_selected_areas.csv'))
data_selected_areas.drop(columns=['target'], inplace=True)

all_data = pd.merge(data, data_networks, on='id',how='outer')
all_data = pd.merge(all_data, data_selected_areas, on='id',how='outer')

all_data.dropna(axis=0,subset=['target'],inplace=True)

all_data.to_csv(Path(Path.home(),'data','Proyecto_Ivo','all_data.csv'), index=False)

