import pandas as pd
from pathlib import Path

tasks = ['Animales','cog','brain','AAL','conn'] 

dimensions = {'cog':['neuropsico','neuropsico_mmse'],
              'brain':['norm_brain_lit'],
              'AAL':['norm_AAL'],
              'conn':['connectivity'],
              'Animales':['properties_timing_vr']
}

merged_data = pd.DataFrame()

for task in tasks:
    for dimension in dimensions[task]:
        data = pd.read_excel(Path(Path.home(),'data','Proyecto_Ivo','data_total.xlsx'),sheet_name=f'{dimension}')
        if merged_data.empty:
            merged_data = data
        else:
            data.drop('target',axis=1,inplace=True)
            merged_data = pd.merge(merged_data,data,on='id',how='inner')
            merged_data.drop([col for col in merged_data.columns if any(x in col for x in ['_x','_y'])],axis=1,inplace=True)

merged_data.to_csv(Path(Path.home(),'data','Proyecto_Ivo','data_total.csv'),index=False)

