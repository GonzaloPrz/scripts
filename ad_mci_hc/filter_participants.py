import pandas as pd
from pathlib import Path

matched_data = pd.read_csv(Path(Path.home(),'data','ad_mci_hc','data_matched_group.csv'))[['id','group']]
speech_timing = pd.read_excel(Path(Path.home(),'data','ad_mci_hc','speech_timing.xlsx'))

data = pd.merge(matched_data,speech_timing,how='inner',on='id')
data.to_csv(Path(Path.home(),'data','ad_mci_hc','data_mfa_matched_group.csv'),index=False)

