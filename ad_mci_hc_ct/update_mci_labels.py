import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','ad_mci_hc_ct')
labels_COH = pd.read_csv(Path(data_dir,'Audios_GERO_T1.csv'))[['id','group']]

labels_COH['group'] = labels_COH['group'].map({0:"HC",1:"MCI"})

filenames = [file for file in data_dir.iterdir() if 'combinada' in file.name and 'lock' not in file.name]

for filename in filenames:
    df = pd.read_csv(Path(data_dir,filename))

    df_COH = df[['COH' in x for x in df.id]]
    df_COH.pop('group')

    df_COH = pd.merge(labels_COH,df_COH,on='id',how='right')

    df_others = df[['COH' not in x for x in df.id]]

    df = pd.concat((df_others,df_COH),axis=0,ignore_index=True)

    df.to_csv(Path(data_dir,filename))




