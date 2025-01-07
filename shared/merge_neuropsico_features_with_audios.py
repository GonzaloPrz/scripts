import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','MPLS')

#Audios_T1 = pd.read_csv(data_dir / 'Audios_GERO_T1.csv')[['id','target','agradable','desagradable','sex','age','education','handedness']]

features_data = pd.read_csv(Path(data_dir,'data_MPLS_cat.csv'))
features_data.rename(columns={'participant_code':'id'}, inplace=True)
neuropsico_features = pd.read_csv(data_dir / 'nps_data.csv')

all_data = pd.merge(features_data, neuropsico_features, on='id', how='inner')

all_data.reset_index(drop=True, inplace=True)

all_data.drop(columns=['protocol_item_trigger','raw_data.transcript.segments','raw_data.transcript.eliminated_segments',
                       'raw_data.transcript.query_timestamp_end','raw_data.transcript.query_timestamp_start',
                       'raw_data.talking-intervals.__speech_segments','raw_data.psycholinguistic_features.list_data','raw_data.freeling-features.pos_tag',
                       'raw_data.sentiment-analysis-gonza.text','raw_data.granularity.dependency_args.transcript.data.text','raw_data.granularity.dependency_args.transcript.data.segments',
                       'raw_data.osv-extraction-multilanguage.list_data','raw_data.words-count.words_count','raw_data.granularity','raw_data.pitch-analysis.msg','raw_data.freeling-features.message','raw_data.talking-intervals.msg','raw_data.osv-extraction-multilanguage.dependency_args.transcript.data.segments'], inplace=True)

all_data = all_data[all_data['Minimental'] != 'sin puntuacion']

all_data = all_data.sort_values(by=['id','protocol_item_order'])
all_data.to_csv(data_dir / 'nps_features_data_MPLS_cat.csv', index=False)



