import pandas as pd
import numpy as np
from tableone import TableOne
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import sys

sys.path.append(str(Path(Path.home(),'scripts_generales')))

from matching_module import perform_matching

project_name = 'MCI_classifier'
target_vars = ['group']

for target_var in target_vars:
    print(target_var)
    # Define variables
    vars = ['sex','age','education','handedness', target_var, 'id']
    output_var = target_var
    
    matching_vars = ['sex','age','education','handedness']

    fact_vars = ['sex','handedness']
    cont_vars = ['education','age']

    data = pd.read_csv(Path(Path.home(),'data',project_name,'data_matched_group.csv'))

    data.dropna(subset=[target_var],inplace=True)
    for fact_var in fact_vars:
        data[fact_var] = data[fact_var].astype('category').cat.codes
 
    caliper = 0.4

    matched_data = perform_matching(data, output_var,matching_vars,fact_vars,caliper=caliper)

    matched_data = matched_data.drop_duplicates(subset='id')

    # Save tables and matched data
    table_before = TableOne(data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])

    #print(table_before)

    table = TableOne(matched_data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])
    print(table_before)
    print(table)

    matched_data.to_csv(Path(Path.home(),'data',project_name,f'data_matched_{target_var}.csv'), index=False)
    table_before.to_csv(Path(Path.home(),'data',project_name,f'table_before_{target_var}.csv'))
    table.to_csv(Path(Path.home(),'data',project_name,f'table_matched_{target_var}.csv'))