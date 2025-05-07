import pandas as pd

original_features = pd.read_csv("/Users/gp/data/ad_mci_hc/data_matched_group.csv")
original_features = original_features[['id','group'] + [f for f in original_features.columns if 'talking' in f]]

mfa_features = pd.read_csv("/Users/gp/data/ad_mci_hc/speech_timing_mfa.csv")

features = pd.merge(original_features,mfa_features,how='inner',on='id')
features.to_csv("/Users/gp/data/ad_mci_hc/data_mfa_matched_group.csv",index=False)