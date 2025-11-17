import pandas as pd
from pathlib import Path

base_dir = Path(Path.home(),'data','impact_dementia') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','impact_dementia')

features = pd.read_csv(Path(base_dir,'features.csv'))
labels = pd.read_csv(Path(base_dir,'labels_cognitive_imp.csv'))

all_data = pd.merge(labels,features,on='id')

all_data.to_csv(Path(base_dir,'all_data_ci_rudas.csv'),index=False)