import pandas as pd
from pathlib import Path

#matched_ids = pd.read_csv(Path(Path.home(),'data','ad_mci_hc','data_matched_group.csv'))[['id','group']]
all_data = pd.read_csv(Path(r'D:\gperez\data\ad_mci_hc','all_data.csv'))
all_transcripts_divided = pd.read_csv(Path(r'D:\gperez\data\ad_mci_hc','all_transcripts_divided.csv'))['id']

all_data = pd.merge(all_transcripts_divided,all_data,on='id')

all_data.to_csv(Path(r'D:\gperez\data\ad_mci_hc','all_data.csv'),index=False)
