import pandas as pd
from pathlib import Path

data_dir = Path(Path.home(),'data','redlat_fugu') if '/Users/gp' in str(Path.home()) else Path('D:\\CNC_Audio\\gonza\\data\\redlat_fugu')
sentiment_analysis_features = pd.read_csv(Path(data_dir,'sentiment_features.csv'))

transcripts = pd.read_excel(Path(data_dir,'fugu_transcripts_redlat.xlsx'))
transcripts['id'] = transcripts['filename'].map(lambda x: x.split('_')[1])

data = sentiment_analysis_features.merge(transcripts,on='id',how='left')

data.to_csv(Path(data_dir,'fugu_sentiment_analysis_with_transcripts.csv'),index=False,encoding='utf-8')