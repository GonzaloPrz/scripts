import pandas as pd
import numpy as np
from tableone import TableOne
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import sys

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from matching_module import *

project_name = 'ad_mci_hc_ct'

data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

target_vars = ['group']

filenames = ['combinada__ad_mci_hc__image_pleasant_memory__features.csv',
             'combinada__ad_mci_hc__image_routine__features.csv',
             'combinada__ad_mci_hc__pleasant_memory_routine__features.csv',
             'combinada__ad_mci_hc__image_pleasant_memory_routine__features.csv'
             ]
for filename in filenames:
    task = filename.split('__')[2]

    for target_var in target_vars:
        print(target_var)
        # Define variables
        vars = ['sex','age','education', target_var, 'id']
        output_var = target_var
        
        matching_vars = ['age','sex','education']

        fact_vars = ['sex']
        cont_vars = ['age','education']

        data = pd.read_csv(Path(data_dir,filename))

        data.dropna(subset=[target_var] + matching_vars,inplace=True)
        for fact_var in fact_vars:
            data[fact_var] = data   [fact_var].astype('category').cat.codes
    
        caliper =  0.43
        #data['group'] = data['group'].map({'AD': 1, 'HC':0})

        matched_data = perform_three_way_matching(data, output_var,matching_vars,fact_vars,treatment_values=('AD','MCI','HC'),caliper=caliper)
        matched_data = matched_data.drop_duplicates(subset='id')

        # Save tables and matched data
        table_before = TableOne(data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])

        #print(table_before)

        table = TableOne(matched_data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])
        print(table_before)
        print(table)

        matched_data.to_csv(Path(data_dir,f'data_matched_{task}_{target_var}.csv'), index=False)
        table_before.to_csv(Path(data_dir,f'table_before_{task}_{target_var}.csv'))
        table.to_csv(Path(data_dir,f'table_matched_{task}_{target_var}.csv'))