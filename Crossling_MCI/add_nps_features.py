import pandas as pd
from pathlib import Path

variables = ['id','NPI-Q_Agitation/Aggression_Severity_Creada','NPI-Q_Anxiety_Severity_Creada','NPI-Q_Apathy/Indifference_Severity_creada','NPI-Q_Depression/Dysphoria_Severity_creada']
data = pd.read_csv(Path(Path.home(),'data','Crossling_MCI','audios_MCI_pleasant_unpleasant_with_nps.csv'))

additional_nps = pd.read_csv(Path(Path.home(),'data','Crossling_MCI','nps_GERO_completa.csv'))[variables]

all_data = pd.merge(data,additional_nps,on='id',how='left')
all_data.to_csv(Path(Path.home(),'data','Crossling_MCI','MCI_pleasant_unpleasant_with_nps.csv'),index=False)