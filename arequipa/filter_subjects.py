import pandas as pd
from pathlib import Path

all_data = pd.read_csv(r"/Users/gp/data/arequipa/all_data.csv")
matched_data = pd.read_csv(r"/Users/gp/data/arequipa/data_matched_group.csv")[['id']]

all_data = pd.merge(all_data, matched_data, on='id', how='inner')
all_data.drop([col for col in all_data.columns if any(x in col for x in  ['text','pos_tag'])], axis=1, inplace=True)
all_data.to_csv(r"/Users/gp/data/arequipa/data_matched_group.csv", index=False)