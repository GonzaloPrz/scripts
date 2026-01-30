import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import skew, kurtosis

sentiments = ['pos','neg','neu']
stats = ['mean','std','min','max','median','skewness','kutrosis']
segmentation = 'phrases'
base_dir = Path(Path.home(),'data','affective_pitch') if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','data','affective_pitch')
transcripts = pd.read_csv(Path(base_dir,f'transcripts_fugu_matched_group_sentiment_{segmentation}.csv'))

output_rows = []
for r, row in transcripts.iterrows():
    output_row = {'id': row['id']}
    for i, sentiment in enumerate(sentiments):
        if isinstance(row[f'{sentiment}_proba'], float) and np.isnan(row[f'{sentiment}_proba']):
            for stat in stats:
                output_row[f'{sentiment}_{stat}'] = np.nan
            continue
    
        sentiment_values = np.fromstring(row[f'{sentiment}_proba'].strip('[]'), sep=' ')

        output_row[f'{sentiment}_mean'] = np.nanmean(sentiment_values)
        output_row[f'{sentiment}_std'] = np.nanstd(sentiment_values)
        output_row[f'{sentiment}_min'] = np.nanmin(sentiment_values)
        output_row[f'{sentiment}_max'] = np.nanmax(sentiment_values)
        output_row[f'{sentiment}_median'] = np.nanmedian(sentiment_values)
        output_row[f'{sentiment}_skewness'] = skew(sentiment_values)
        output_row[f'{sentiment}_kutrosis'] = kurtosis(sentiment_values)

    output_rows.append(output_row)

output_df = pd.merge(transcripts,pd.DataFrame(output_rows))

output_df.to_csv(Path(base_dir,f'transcripts_fugu_matched_group_sentiment_{segmentation}.csv'), index=False)
