import pandas as pd
from pathlib import Path

base_dir = Path(Path.home(),'data','affective_pitch') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')

data = pd.read_csv(Path(base_dir,'updated_data_24-09-2025.csv'))

filtered_data = data.loc[[group in ['CN','FTD','AD'] for group in data['group']]]
filtered_data = filtered_data[['id','group','site','age','sex']]

filtered_data.to_csv(Path(base_dir,'filtered_data.csv'), index=False)