import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','ad_mci_hc') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','ad_mci_hc')

if not data_dir.exists():
    data_dir = Path(r'D:\gperez\data\ad_mci_hc')

original_features = pd.read_csv(Path(data_dir,"data_matched_group.csv"))

original_features = original_features[['id','group'] + [f for f in original_features.columns if 'talking' in f]]

hiiamsid_features = pd.read_csv(Path(data_dir,"distances_hiiamsid.csv"))

features = pd.merge(original_features,hiiamsid_features,how='inner',on='id')

cross_enc_features = pd.read_csv(Path(data_dir,"distances_cross_enc.csv"))

features = pd.merge(features,cross_enc_features,how='inner',on='id')

features.to_csv(Path(data_dir,"all_data_matched_group.csv",index=False))