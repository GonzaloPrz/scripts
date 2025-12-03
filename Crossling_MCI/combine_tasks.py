from pathlib import Path
import pandas as pd

data_dir = Path(Path.home(),'data','crossling_mci') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','crossling_mci')

languages = ['french','spanish']
tasks = ['negative','positive']

data_french = pd.DataFrame()

for language in languages:
    data = pd.DataFrame()
    labels = pd.DataFrame()
    for task in tasks:
        file_path = Path(data_dir,f'data_{language}_{task}.csv')

        data_task = pd.read_csv(file_path)
        feature_names = list(set(data_task.columns) - set(['id','age','sex','education','group','NPI_Agitation_Frequency','NPI_Agitation_Severity',
                                'NPI_Agitation_F.S','NPI.Depression_Frequency','NPI_Depression_Severity','NPI.Depression_F.S',
                                'NPI.Anxiety_Frequency','NPI_Anxiety_Severity','NPI.Anxiety_F.S','NPI.Euphoria','NPI_Apathy_Frequency',
                                'NPI_Apathy_Severity','NPI_Apathy_F.S','language','story.type']))

        labels_task = data_task[list(set(data_task.columns) - set(feature_names))].sort_values(by='id')

        data_task = data_task[['id'] + feature_names]
        data_task.columns = ['id'] + [f'{task}__all__{col}' for col in feature_names]
        if data.empty:
            data = data_task
            labels = labels_task
        else:
            data = data.merge(data_task,on=['id'],how='outer')
            labels = pd.concat([labels, labels_task],axis=0).drop_duplicates().reset_index(drop=True)

    data = labels.merge(data, on=['id'],how='outer')

    data.to_csv(Path(data_dir,f'features_{language}.csv'),index=False)
