import pandas as pd
from pathlib import Path

base_dir = Path(Path.home(),'data','affective_pitch') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')

filename = 'mean_windows'

data = pd.read_csv(Path(base_dir,f'{filename}.csv'))

try:
    data = data.drop(['sex','group','site','age'],axis=1)
except:
    pass

#baseline_data = pd.read_csv(Path(base_dir,'all_audios_energy_pitch_features_baseline.csv'))

labels = pd.read_csv(Path(base_dir,'matched_ids.csv'))[['id','sex','age','site','group']]

data = data.merge(labels,on='id',how='inner')

for groups in[['CN','AD'],['CN','FTD'],['AD','FTD']]:
    group_label = '_'.join(groups)
    subset = data[data['group'].isin(groups)].copy()
    subset['group'] = subset['group'].map({groups[0]:0,groups[1]:1})
    Path(base_dir.parent,f'affective_pitch_{group_label}').mkdir(exist_ok=True,parents=True)
    subset.drop([col for col in subset.columns if 'Unnamed' in col],axis=1,inplace=True)

    #subset = pd.merge(subset,baseline_data,on='id',how='left')

    subset.to_csv(Path(base_dir.parent,f'affective_pitch_{group_label}',f'{filename}_{group_label}.csv'),index=False)

data['group'] = data['group'].map({'CN':0,'AD':1,'FTD':2})
data = data.drop([col for col in data.columns if 'Unnamed' in col],axis=1)

data.to_csv(Path(base_dir,f'{filename}.csv'),index=False)