import pandas as pd
from pathlib import Path

transcripts = pd.read_csv(Path(Path.home(),'data','sj','transcripts_participants.csv'))[['filename','text']]

for r, row in transcripts.iterrows():
    filename = row['filename'].replace({'Preg'})
    with open(Path(Path.home(),'data','sj','AUDIOS PROCESADOS',row['filename'].replace('.wav','.txt')),'w') as f:
        f.write(row['text'])

    