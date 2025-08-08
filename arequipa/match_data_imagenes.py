import pandas as pd
from pathlib import Path
from tableone import TableOne

all_data = pd.read_csv(r"/Users/gp/data/arequipa/data_imagenes.csv")
all_data['id'] = all_data['id'].apply(lambda x: x.lower().replace('_',''))

matched_ids = pd.read_csv(r"/Users/gp/data/arequipa/data_matched_group.csv")['id']

all_data = pd.merge(all_data, matched_ids, on='id', how='inner')

vars = ['sex','age','education', 'group', 'id']
    
fact_vars = ['sex']
cont_vars = ['education','age']

for fact_var in fact_vars:
    all_data[fact_var] = all_data[fact_var].astype('category').cat.codes
 
table_before = TableOne(all_data,list(set(vars) - set(['group','id'])),fact_vars,groupby='group', pval=True, nonnormal=[])

print(table_before)

all_data.to_csv(r"/Users/gp/data/arequipa/data_imagenes_matched_group.csv", index=False)

