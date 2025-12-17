import pandas as pd
from pathlib import Path

base_dir = Path(Path.home(),'data','affective_pitch') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')

data = pd.read_csv(Path(base_dir,'mean_pitch_features.csv'))
data = data.drop(['group','sex','education','site','age'],axis=1)

labels = pd.read_csv(Path(base_dir,'matched_ids.csv'))

data = data.merge(labels,on='id',how='inner')

for groups in[['CN','AD'],['CN','FTD'],['AD','FTD']]:
    group_label = '_'.join(groups)
    subset = data[data['group'].isin(groups)].copy()
    subset['group'] = subset['group'].map({groups[0]:0,groups[1]:1})
    Path(base_dir.parent,f'affective_pitch_{group_label}').mkdir(exist_ok=True,parents=True)
    subset.to_csv(Path(base_dir.parent,f'affective_pitch_{group_label}',f'mean_pitch_features_{group_label}.csv'),index=False)

data['group'] = data['group'].map({'CN':0,'AD':1,'FTD':2})
data = data.drop([col for col in data.columns if 'Unnamed' in col],axis=1)

data.to_csv(Path(base_dir,'mean_pitch_features.csv'),index=False)