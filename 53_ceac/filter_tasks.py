import pandas as pd
from pathlib import Path
import numpy as np
import itertools

base_dir = Path(Path.home(),'data','53_ceac') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','53_ceac')

data = pd.read_csv(Path(base_dir,'all_data.csv'))

tasks = np.unique([col.split('__')[0] for col in data.columns if '__' in col])
dimensions = np.unique([col.split('__')[1] for col in data.columns if '__' in col])

na_values = pd.DataFrame(columns=['task','dimension','feature','na_values'])

for task,dimension in itertools.product(tasks,dimensions):
    data_task_dimension = data[[col for col in data.columns if all(x in col for x in [task,dimension])]]

    for col in data_task_dimension.columns:
        na_count = data_task_dimension[col].isna().sum()
        na_values.loc[na_values.shape[0],:] = [task,dimension,col.replace(f'{task}__','').replace(f'{dimension}__',''),na_count]

na_values = na_values.groupby(['task','dimension']).agg({'na_values':'mean'}).reset_index()
na_values.to_csv(Path(base_dir,'na_values_per_task_dimension.csv'),index=None)