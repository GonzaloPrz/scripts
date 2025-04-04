import pandas as pd
import numpy as np
from tableone import TableOne
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import sys

sys.path.append(str(Path(Path.home(),'scripts_generales')))

from matching_module import *

project_name = 'ad_mci_hc'
target_vars = ['group']

for target_var in target_vars:
    print(target_var)
    # Define variables
    vars = ['sex','age','education', target_var, 'id','country']
    output_var = target_var
    
    matching_vars = ['age','sex','education']

    fact_vars = ['sex']
    cont_vars = ['age','education']

    data = pd.read_csv(Path(Path.home(),'data',project_name,'all_data.csv'))

    data.dropna(subset=[target_var] + matching_vars,inplace=True)
    for fact_var in fact_vars:
        data[fact_var] = data   [fact_var].astype('category').cat.codes
 
    caliper =  0.05
    #data['group'] = data['group'].map({'AD': 1, 'HC':0})

    matched_data = perform_three_way_matching(data, output_var,matching_vars,fact_vars,treatment_values=('AD','MCI','HC'),caliper=caliper)
    matched_data = matched_data.drop_duplicates(subset='id')

    # Save tables and matched data
    table_before = TableOne(data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])

    #print(table_before)

    table = TableOne(matched_data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])
    print(table_before)
    print(table)

    matched_data.to_csv(Path(Path.home(),'data',project_name,f'data_matched_{target_var}_country.csv'), index=False)
    table_before.to_csv(Path(Path.home(),'data',project_name,f'table_before_{target_var}_country.csv'))
    table.to_csv(Path(Path.home(),'data',project_name,f'table_matched_{target_var}.csv'))