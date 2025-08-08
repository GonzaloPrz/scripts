import pandas as pd
from pathlib import Path

matched_data = pd.read_csv(r"/Users/gp/data/arequipa_reg_mci/data_matched_group.csv")
all_data = pd.read_csv(r"/Users/gp/data/arequipa_reg_mci/all_data.csv")

unmatched_data = all_data[~all_data['id'].isin(matched_data['id'])]

unmatched_data.to_csv(r"/Users/gp/data/arequipa/data_unmatched_group.csv", index=False)