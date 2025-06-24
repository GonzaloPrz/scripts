import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','ad_mci_hc')

if not data_dir.exists():
    data_dir = Path(r'D:\gperez\data\ad_mci_hc')

original_features = pd.read_csv(Path(data_dir,"filtered_data_no_hallucinations_matched_group.csv"))

original_features = original_features[['id','group'] + [f for f in original_features.columns if 'talking' in f]]

text_features = pd.read_csv(Path(data_dir,"distances_all-MiniLM-L6-v2.csv"),encoding='utf-16')

features = pd.merge(original_features,text_features,how='inner',on='id')

features.to_csv(Path(data_dir,"filtered_data_no_hallucinations_matched_group.csv",index=False))