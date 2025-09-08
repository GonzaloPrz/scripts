import pandas as pd
from pathlib import Path

filenames = ['granularity','osv','sentiment','psycho','verbosity']
data_dir = Path(Path.home(),'data','53_ceac') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','53_ceac')
all_data = pd.DataFrame()

for filename in filenames:
    file_path = Path(f"{data_dir}/{filename}.csv")
    if file_path.exists():
        df = pd.read_csv(file_path)
    if all_data.empty:
        all_data = df
    else:
        all_data = pd.merge(all_data,df,on='id',how='outer')

labels = pd.read_csv(Path(data_dir,'labels.csv'))[['id','sex','age','depression','anxiety']]

labels['id'] = labels['id'].map(lambda x: f'CEAC_{x}')
all_data = pd.merge(labels, all_data, on='id', how='outer')

all_data.to_csv(Path(data_dir,'all_data.csv'), index=False)