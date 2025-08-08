import pandas as pd
from pathlib import Path

mean_all_data = pd.read_csv(Path(Path.home(),'data','MCI_classifier','features_mean_all.csv'))
sep_data = pd.read_csv(Path(Path.home(),'data','MCI_classifier','original_features.csv'))

data = pd.merge(mean_all_data, sep_data, on='id', how='inner')

word_properties = ['granularity','log_frq','num_phon','phon_neigh']
speech_timing = ['nsyll','npause','pause_duration','ASD','articulation_rate']

ids = data.pop('id')
data = data[[col for col in data.columns if any(x in col for x in word_properties + speech_timing)]]

data.columns = [f'{x.split("__")[1]}__word_properties__{x.split("__")[0]}' if any(word_property in x for word_property in word_properties) else f'{x.split("__")[1]}__speech_timing__{x.split("__")[0]}' for x in data.columns]

data['id'] = ids

#data_matched_group = pd.read_csv(Path(Path.home(),'data','MCI_classifier','data_matched_group.csv'))
data_matched_unbalanced_group = pd.read_csv(Path(Path.home(),'data','MCI_classifier','data_matched_unbalanced_group.csv'))

#data_matched_group = pd.merge(data_matched_group,data,on='id')
data_matched_unbalanced_group = pd.merge(data_matched_unbalanced_group,data,on='id')

#data_matched_group.to_csv(Path(Path.home(),'data','MCI_classifier','data_matched_group.csv'))
data_matched_unbalanced_group.to_csv(Path(Path.home(),'data','MCI_classifier','data_matched_unbalanced_group.csv'))
