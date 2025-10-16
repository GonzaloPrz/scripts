from pathlib import Path
import pandas as pd


dara_dir = Path(Path.home(),'data','arequipa_reg') if '/Users/gp' in (str(Path.home())) else Path('D:','CNC_Audio','gonza','data','arequipa_reg')

features = pd.read_csv(dara_dir/'data_unmatched_group.csv')
labels = pd.read_csv(dara_dir/'nps_data.csv')[['id','IFS','MOCA_TOTAL']]
labels['id'] = labels['id'].map(lambda x: x.lower())

all_data = features.merge(labels,on='id',how='inner')
all_data.to_csv(dara_dir/'data_unmatched_group.csv')
