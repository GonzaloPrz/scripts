import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data',Path(__file__).parent.name)

data = pd.read_csv(Path(data_dir,'all_data.csv'))

drop_features = [col for col in data.columns if 'freeling' in col and any(x in col for x in ['pronoun','person','content','quantity','token'])]
data = data.drop(columns=drop_features)

data.to_csv(Path(data_dir,'all_data_filtered.csv'), index=False)
