import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(), 'data', 'Include') if 'Users/gp' in str(Path.home()) else Path('D', 'CNC_Audio', 'gonza', 'data', 'Include')

language = 'greek'

df = pd.DataFrame()

for file in Path(data_dir,language).glob('*.xlsx'):
    if df.empty:
        df = pd.read_excel(file)
    else:
        df = pd.merge(df, pd.read_excel(file), on='word', how='outer')

df.to_csv(Path(data_dir,f'all_features_{language}.csv'), index=False)
