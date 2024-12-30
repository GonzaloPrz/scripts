import pandas as pd
import numpy as np
from tableone import TableOne
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import sys

sys.path.append(str(Path(Path.home(),'scripts_generales')))

from matching_module import perform_matching

project_name = 'GeroApathy'
target_vars = ['Depression_Total_Score_label','DASS_21_Depression_V_label','AES_Total_Score_label',
                'MiniSea_MiniSea_Total_EkmanFaces_label','MiniSea_minisea_total_label']

for target_var in target_vars:
    # Define variables
    vars = ['sex','age','education', target_var, 'id','target']
    output_var = target_var
    
    matching_vars = ['education','target']

    fact_vars = ['target','sex']
    cont_vars = ['education','age']

    data = pd.read_csv(Path(Path.home(),'data',project_name,'all_data_agradable.csv'))
    data.drop(['target'],axis=1,inplace=True)
    demographic_data = pd.read_csv(Path(Path.home(),'data','Audios_GERO_T1.csv'))
    data = pd.merge(data,demographic_data,left_on='id',right_on='id',how='inner')

    data.dropna(subset=[target_var],inplace=True)
    for fact_var in fact_vars:
        data[fact_var] = data[fact_var].astype('category').cat.codes
 
    matched_data = perform_matching(data, output_var,matching_vars,fact_vars)

    matched_data = matched_data.drop_duplicates(subset='id')

    # Save tables and matched data
    table_before = TableOne(data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])

    print(table_before)

    table = TableOne(matched_data,list(set(vars) - set([output_var,'id'])),fact_vars,groupby=output_var, pval=True, nonnormal=[])

    print(table)

    matched_data.to_csv(Path(Path.home(),'data',project_name,f'data_matched_agradable_{target_var}.csv'), index=False)
    table_before.to_csv(Path(Path.home(),'data',project_name,f'table_before_agradable_{target_var}.csv'))
    table.to_csv(Path(Path.home(),'data',project_name,f'table_matched_agradable_{target_var}.csv'))