import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','GERO_Ivo')

data = pd.DataFrame()

for file in data_dir.glob('*.csv'):
    if file.stem == 'all_data':
        continue
    
    df = pd.read_csv(file)
    if data.empty:
        data = df
    else:
        data = pd.merge(data, df, on='id', how='outer')

data.to_csv(data_dir / 'all_data.csv', index=False)