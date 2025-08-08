import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','ad_mci_hc')

if not data_dir.exists():
    data_dir = Path(r'D:\gperez\data\ad_mci_hc')

original_features = pd.read_csv(Path(data_dir,"speech_timing__text.csv"))

word_properties = pd.read_csv(Path(data_dir,"word_properties.csv"),encoding='utf-8')

features = pd.merge(original_features,word_properties,how='inner',on='id')

features.to_csv(Path(data_dir,"data_matched_group.csv"),index=False)