import pandas as pd
from pathlib import Path

ids = pd.read_excel(Path(Path.home(),'data','Proyecto_Ivo','data_total.xlsx'),sheet_name='brain_lit')[['id','target']]
conn_data = pd.read_excel(Path(Path.home(),'data','Proyecto_Ivo','data_total.xlsx'),sheet_name='connectivity')
conn_data.drop('target',axis=1,inplace=True)
conn_data_add = pd.merge(conn_data,ids,on='id',how='right')

conn_data_add.to_excel(Path(Path.home(),'data','Proyecto_Ivo','conn_data.xlsx'),index=False)