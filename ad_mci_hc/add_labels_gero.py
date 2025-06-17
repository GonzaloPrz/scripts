from pathlib import Path
import pandas as pd

labels = pd.read_csv(Path(Path.home(),'data','ad_mci_hc','Audios_GERO_T1.csv'))[['id','group']]
labels['group'] = labels['group'].map({0:'HC',1:'MCI'})

data_gero = pd.read_csv(Path(Path.home(),'data','ad_mci_hc','all_data_gero.csv'))

all_data_gero = pd.merge(labels,data_gero,on='id')

all_data_gero.to_csv(Path(Path.home(),'data','ad_mci_hc','all_data_gero.csv'))