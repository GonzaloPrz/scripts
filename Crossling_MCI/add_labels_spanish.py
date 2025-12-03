import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','crossling_mci') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','crossling_mci')

data = pd.read_csv(Path(data_dir,'data_spanish.csv'))
labels = pd.read_csv(Path(data_dir,'nps_GERO_filtrada.csv'))[['id',
                                                          'NPI_Agitation_Severity',
                                                          'NPI_Depression_Severity',
                                                          'NPI_Anxiety_Severity',
                                                          'NPI_Apathy_Severity']]

data = data.merge(labels, on='id', how='inner')
data.to_csv(Path(data_dir,'data_spanish.csv'), index=False)