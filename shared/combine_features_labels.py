import pandas as pd
from pathlib import Path

project_name = 'GeroApathy'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

features_data = pd.read_csv(Path(data_dir,'features_data.csv'))

y_labels = ['DASS_21_Depression','DASS_21_Anxiety','DASS_21_Stress','AES_Total_Score','MiniSea_MiniSea_Total_FauxPas','Depression_Total_Score','MiniSea_emf_total','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total']

labels_data =pd.read_excel(Path(data_dir,f'nps_data_filtered_no_missing.xlsx'))[['id']+y_labels]

all_data = features_data.merge(labels_data, on='id')

all_data.to_csv(Path(data_dir,'all_data.csv'),index=False)