import pandas as pd
from pathlib import Path
import re 

transcripts = pd.read_csv(Path(Path.home(),'data','sj','kept_transcripts.csv'))

for r, row in transcripts.iterrows():
    #Filter all sentences between []
    transcripts.loc[r,'transcript_participant'] = re.sub(r'\[.*?\]', '', row['revised_text'])

transcripts.to_csv(Path(Path.home(),'data','sj','kept_transcripts_participants.csv'), index=False)
