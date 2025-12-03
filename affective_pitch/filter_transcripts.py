import pandas as pd
from pathlib import Path

base_dir = Path(Path.home(),'data','affective_pitch') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')

all_transcripts = pd.read_csv(Path(base_dir,'participant_timestamps.csv'))
matched_subset = pd.read_csv(Path(base_dir,'matched_ids.csv'))[['id','sex','age','site','education']]

matched_transcripts = pd.merge(all_transcripts,matched_subset,on='id',how='inner')
matched_transcripts = matched_transcripts[matched_transcripts['task'] == 'Fugu']

matched_transcripts.to_csv(Path(base_dir,'transcripts_fugu_matched_group.csv'), index=False)