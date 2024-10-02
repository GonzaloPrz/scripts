import pandas as pd
from pathlib import Path

project = 'MCI_classifier'

data_dir = Path(Path.home(),'data',project) if 'Users/gp' in str(Path.home()) else Path(Path.home(),'D:','CNC_Audio','gonza','data',project)

df = pd.read_excel(Path(data_dir,'features_mod.xlsx'))

df_modified = pd.DataFrame()

tasks = ['letra_f','letra_a','letra_s','animales']

for task in tasks:
    df_task = df[df['task'] == task]
    for col in df_task.columns:
        if col == 'id':
            continue
        df_task[f'{task}_{col}'] = df_task[col]
        df_task.drop(columns=[col], inplace=True)

    if df_modified.empty:
        df_modified = df_task.reset_index(drop=True)
    else:
        df_modified = pd.merge(df_modified, df_task, on='id')

df_modified.to_excel(Path(data_dir,'features_mod.xlsx'), index=False)