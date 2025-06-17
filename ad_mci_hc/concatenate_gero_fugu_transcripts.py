import pandas as pd
from pathlib import Path

#matched_ids = pd.read_csv(Path(Path.home(),'data','ad_mci_hc','data_matched_group.csv'))[['id','group']]
transcripts_gero = pd.read_excel(Path(r'D:\gperez\data\ad_mci_hc','fugu_transcripts_gero.xlsx'))[['id','text']]
transcripts_redlat = pd.read_excel(Path(r'D:\gperez\data\ad_mci_hc','fugu_transcripts_redlat.xlsx'))
transcripts_redlat['id'] = transcripts_redlat['filename'].apply(lambda x: x.split('_')[1])
transcropts_redlat = transcripts_redlat[['id','text']]
transcropts_gero = transcripts_gero[['id','text']]

all_transcripts = pd.concat((transcripts_gero,transcripts_redlat))
#matched_transcripts = pd.merge(matched_ids,all_transcripts,how='inner',on='id')

#matched_transcripts.to_csv(Path(Path.home(),'data','ad_mci_hc','transcripts_matched_group.csv'),index=False)
all_transcripts.to_excel(Path(r'D:\gperez\data\ad_mci_hc','all_transcripts.xlsx'),index=False)


