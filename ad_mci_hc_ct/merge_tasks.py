import pandas as pd
from pathlib import Path
import itertools
data_dir = Path(Path.home(),'data','ad_mci_hc_ct') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','ad_mci_hc_ct')

tasks = ['image','pleasant_memory','routine']

for comb in itertools.combinations(tasks,2):
    task1 = comb[0]
    task2 = comb[1]

    filename1 = f'{task1}__features.csv'
    filename2 = f'{task2}__features.csv'

    filename_save = f'{task1}_{task2}__features.csv'

    data1 = pd.read_csv(Path(data_dir,filename1))
    data2 = pd.read_csv(Path(data_dir,filename2))
    data2.drop(['group','sex','age','education'],axis=1,inplace=True)

    data = pd.merge(data1,data2,on='id')

    data.to_csv(Path(data_dir,filename_save))

for comb in itertools.combinations(tasks,3):
    task1 = comb[0]
    task2 = comb[1]
    task3 = comb[2]

    filename1 = f'{task1}__features.csv'
    filename2 = f'{task2}__features.csv'
    filename3 = f'{task3}__features.csv'

    filename_save = f'{task1}_{task2}_{task3}__features.csv'

    data1 = pd.read_csv(Path(data_dir,filename1))
    data2 = pd.read_csv(Path(data_dir,filename2))
    data2.drop(['group','sex','age','education'],axis=1,inplace=True)

    data3 = pd.read_csv(Path(data_dir,filename3))
    data3.drop(['group','sex','age','education'],axis=1,inplace=True)


    data = pd.merge(data1,data2,on='id')
    data = pd.merge(data,data3,on='id')

    data.to_csv(Path(data_dir,filename_save))