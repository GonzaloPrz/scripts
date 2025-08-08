import pandas as pd
from pathlib import Path

path_to_data = Path(Path.home(),'data','arequipa')

feature_files = list(path_to_data.glob('arequipa__*.csv'))

merged_data = pd.read_csv(path_to_data / 'nps_data.csv')[['id','group','age','sex','education']]
merged_data['id'] = merged_data['id'].apply(lambda x: x.lower())

merged_data = merged_data.drop(columns=[col for col in merged_data.columns if 'text' in col])

relevant_dimensions = {#'vocal_sostenida':['pitch'],
                        'narración_de_video':['pitch','voice_quality','talking_intervals','word_properties','granularity','concreteness','universal_dependencies','verbosity','mlu'],
                        'descripción_de_dibujo_1':['pitch','voice_quality','talking_intervals','word_properties','granularity','concreteness','universal_dependencies','verbosity','mlu'],
                        'descripción_de_dibujo_2':['pitch','voice_quality','talking_intervals','word_properties','granularity','concreteness','universal_dependencies','verbosity','mlu'],
                        'recuerdo_agradable':['pitch','voice_quality','talking_intervals','word_properties','granularity','concreteness','universal_dependencies','freeling','verbosity','mlu'],
                        'narración_de_historia':['pitch','voice_quality','talking_intervals','word_properties','granularity','concreteness','universal_dependencies','freeling','granularity','verbosity','mlu','semantic_distance'],
                        'día_típico':['pitch','voice_quality','talking_intervals','word_properties','granularity','concreteness','universal_dependencies','freeling','granularity','verbosity','mlu'],
                        #'testimonio':['pitch','voice_quality','talking_intervals','psycholinguistics','granularity','concreteness','universal_dependencies','freeling','granularity','osv','sentiment_analysis','graph_content','verbosity','text'],
                        #'auto_sin_ruedas':['pitch','voice_quality','talking_intervals','psycholinguistics','granularity','concreteness','universal_dependencies','freeling','granularity','osv','sentiment_analysis','graph_content','verbosity','text'],
                        #'sílabas':['pitch','voice_quality'],
                        #'lectura_de_párrafo':['pitch','voice_quality','talking_intervals'],
                        #'silencio':[]
                       }

for feature_file in feature_files:
    dimension = feature_file.stem.split('__')[1]
    df = pd.read_csv(feature_file)
    df['id'] = df['filename'].apply(lambda x: x.split('_')[0].split('/')[-1].lower())
    df['task'] = df['filename'].apply(lambda x: x.split('__')[-1].replace('.wav','').replace('.txt',''))
    df.drop(columns=['filename'], inplace=True)
    tasks  = list(set(df['task'].unique()) - set(['error_log_granularity_features']))
    
    if 'text' in feature_file.name:
        df.dropna(subset=['text'], inplace=True)
        df['text'] = df['text'].apply(lambda x:1)

    for task in tasks:
        if task not in relevant_dimensions.keys():
            continue
        if dimension not in relevant_dimensions[task]:
            continue
        print(task,dimension)

        df_ = df[df['task'] == task]
        feature_names = [col for col in df_.columns if col not in ['id','task','group','list_data','query_duration','query_timestamp_start','query_timestamp_end']]

        for feature_name in feature_names:
            df_.rename(columns={feature_name: f'{task}__{dimension}__{feature_name}'}, inplace=True)
        
        df_ = df_[['id'] + [f'{task}__{dimension}__{feature_name}' for feature_name in feature_names]]
        df_.drop(columns = [f'{task}_{dimension}_text'], inplace=True, errors='ignore')
        merged_data = pd.merge(merged_data, df_, on='id', how='outer')

merged_data = merged_data[merged_data['group'].isin(['CONTROL','DCL'])]
merged_data = merged_data.drop_duplicates(subset='id')
merged_data['group'] = merged_data['group'].map({'CONTROL':0,'DCL':1})
#nps_data = pd.read_csv(path_to_data / 'nps_data.csv')
#nps_data['id'] = nps_data['id'].apply(lambda x: x.lower())
#merged_data = pd.merge(merged_data, nps_data, on='id', how='inner')
merged_data = merged_data.drop(columns=[col for col in merged_data.columns if any(x in col for x in ['text','pos_tag'])])
merged_data.to_csv(path_to_data / 'all_data.csv', index=False)