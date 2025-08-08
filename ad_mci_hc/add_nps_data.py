import pandas as pd
from pathlib import Path
import numpy as np

data_dir = Path(Path.home(),'data','ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','ad_mci_hc')

if not data_dir.exists():
    data_dir = Path(r'D:\gperez\data\ad_mci_hc')

all_data = pd.read_csv(Path(data_dir,'all_data_no_hallucinations.csv'))

nps_data = pd.read_csv(Path(data_dir,'Audios_GERO_T1.csv'))[['id','moca_puntaje','FCRST_Total_Inmediate_Free_Recall','FCRST_Total_Inmediate_Recall']]

all_data_gero = pd.merge(all_data,nps_data,on='id')

all_data_redlat = all_data[~all_data['id'].isin(all_data_gero['id'])]


condition = (
    (np.abs(all_data_gero['moca_puntaje'] - 20) > 2) |
    (np.abs(all_data_gero['FCRST_Total_Inmediate_Free_Recall'] - 26) > 2) |
    (np.abs(all_data_gero['FCRST_Total_Inmediate_Recall'] - 46) > 2)
)

filtered_data_gero = all_data_gero[condition]
filtered_data = pd.concat((all_data_redlat,filtered_data_gero),axis=0)

filtered_data.to_csv(Path(data_dir,'filtered_data_no_hallucinations.csv'))
