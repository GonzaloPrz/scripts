import pandas as pd
from pathlib import Path
import numpy as np

conditions = ['depression','anxiety']

data_dir = Path(Path.home(),'data','53_ceac') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','53_ceac')

merged_dataset = pd.DataFrame()
for condition in conditions:
    df = pd.read_csv(Path(data_dir,f'all_data_{condition}.csv'))
    for c in set(conditions) - set([condition]):
        df[c] = np.nan
    features = [ft for ft in df.columns if '__' in ft]
    if merged_dataset.empty:
        merged_dataset = df[['id','anxiety','depression'] + features]
    else:
        merged_dataset = pd.concat((merged_dataset, df[['id','anxiety','depression'] + features]),axis=0)

merged_dataset.to_csv(Path(data_dir,'all_data.csv'),index=None)