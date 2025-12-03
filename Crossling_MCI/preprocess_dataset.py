import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','crossling_mci') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','crossling_mci')

languages = ['french','spanish']
tasks = ['positive','negative']

for language in languages:
    data = pd.DataFrame()
    labels = pd.DataFrame()

    for task in tasks:
        data_task = pd.read_csv(Path(data_dir,f'data_{language}_{task}.csv'))
        
        feature_cols = list(set(data_task.columns) - set(['id','sex','age','education','group',
                                                          'NPI_Agitation_Severity',''
                                                          'NPI_Depression_Severity',
                                                          'NPI_Anxiety_Severity',
                                                          'NPI_Apathy_Severity']))
        
        labels_task = data_task[list(set(data_task.columns) - set(feature_cols))]

        data_task = data_task[['id'] + feature_cols]
        data_task.columns = ['id'] + [f'{task}__all__{col}' for col in feature_cols]
        if data.empty:
            data = data_task
            labels = labels_task
        else:
            data = data.merge(data_task, on='id', how='outer')
            labels = pd.concat((labels,labels_task),axis=0).drop_duplicates('id').reset_index(drop=True)
        
    data = data.merge(labels, on='id', how='outer')
    data.to_csv(Path(data_dir,f'data_{language}.csv'), index=False)

        