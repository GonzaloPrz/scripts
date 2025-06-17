import pandas as pd
from pathlib import Path

matched_ids = pd.read_csv(Path(Path.home(),'data','ad_mci_hc','data_matched_group.csv'))[['id','group']]

transcripts_redlat = pd.read_excel(Path(Path.home(),'data','redlat_fugu','redlat_fugu_transcripts.xlsx'))
transcripts_redlat['id'] = transcripts_redlat['filename'].str.split('_').str[1]

transcripts_redlat.drop(columns=['filename'], inplace=True)

transcripts_gero = pd.read_excel(Path(Path.home(),'data','ad_mci_hc','Transcripts_Fugu_diarized.xlsx'))[['id','text']]

all_transcripts = pd.concat([transcripts_redlat, transcripts_gero], ignore_index=True)

matched_transcripts = pd.merge(matched_ids, all_transcripts, how='inner', on='id')

matched_transcripts.to_csv(Path(Path.home(),'data','ad_mci_hc','data_matched_transcripts.csv'), index=False)