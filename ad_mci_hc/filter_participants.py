import pandas as pd
from pathlib import Path

#matched_ids = pd.read_csv(Path(Path.home(),'data','ad_mci_hc','data_matched_group.csv'))[['id','group']]
data_dir = Path(Path.home(),'data','ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','ad_mci_hc')

if not data_dir.exists():
    data_dir = Path(r'D:\gperez\data\ad_mci_hc')

all_data = pd.read_csv(Path(data_dir,'filtered_data_no_hallucinations_matched_group.csv'))
all_transcripts_divided = pd.read_csv(Path(data_dir,'all_transcripts_divided.csv'))

filtered_transcripts_divided = pd.merge(all_data['id'],all_transcripts_divided,how='inner',on='id')

filtered_transcripts_divided.to_excel(Path(data_dir,'filtered_transcripts_no_hallucinations_matched_group.xlsx'),index=False)
