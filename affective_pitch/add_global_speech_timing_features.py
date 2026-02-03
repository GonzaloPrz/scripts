import pandas as pd
from pathlib import Path
import itertools

base_dir = Path.home() / 'data' / 'affective_pitch' if Path.home().name == 'gp' else Path('D:/CNC_Audio/gonza/data/affective_pitch')

speech_timing_features = pd.read_csv(base_dir / 'all_audios_timing_features.csv')
segmentations = ['windows', 'phrases']
groups = ['CN_FTD']

for segmentation,groups in itertools.product(segmentations,groups):
    all_data = pd.read_csv(base_dir.parent / f'affective_pitch_{groups}' / f'mean_{segmentation}_{groups}.csv')

    data = pd.read_csv(Path(base_dir.parent,f'affective_pitch_{groups}',f'baseline_data_{groups}.csv'))

    #merged_data = data.merge(speech_timing_features, on='id', how='left')
    merged_data = all_data.merge(data, on='id', how='left')
    #merged_data = merged_data[merged_data['id'].isin(ids)]
    merged_data.to_csv(base_dir.parent / f'affective_pitch_{groups}'/ f'mean_{segmentation}_{groups}.csv', index=False)