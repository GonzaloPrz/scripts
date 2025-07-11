import pandas as pd
import numpy as np
from tableone import TableOne
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import sys

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from matching_module import perform_matching

project_name = 'mci_hc_fugu'
data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data',project_name)

target_vars = ['group']

filename = 'filtered_data_no_hallucinations_matched_group.csv'

for target_var in target_vars:
    print(target_var)
    # Define variables
    vars = ['sex','age','education', target_var, 'id']

    dem_data = pd.read_csv(Path(data_dir,'filtered_data_no_hallucinations.csv'))[vars]
    dem_data.drop(target_var, axis=1, inplace=True)
    output_var = target_var
    
    matching_vars = ['sex','age','education']

    fact_vars = ['sex']
    cont_vars = ['education','age']

    data = pd.read_csv(Path(data_dir,filename))

    data = pd.merge(data, dem_data, on='id', how='left')

    data.to_csv(Path(data_dir,filename), index=False)
    data.dropna(subset=[target_var],inplace=True)
    for fact_var in fact_vars:
        data[fact_var] = data[fact_var].astype('category').cat.codes
 
    caliper = 0.4

    table_before = TableOne(data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])
    print(table_before)

    matched_data = perform_matching(data, output_var,matching_vars,fact_vars,caliper=caliper)

    matched_data = matched_data.drop_duplicates(subset='id')

    # Save tables and matched data
    #print(table_before)

    table = TableOne(matched_data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])
    print(table)

    matched_data.to_csv(Path(data_dir,f'data_matched_{target_var}.csv'), index=False)
    table_before.to_csv(Path(data_dir,f'table_before_{target_var}.csv'))
    table.to_csv(Path(data_dir,f'table_matched_{target_var}.csv'))