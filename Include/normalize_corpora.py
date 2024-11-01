import pandas as pd
from pathlib import Path
import numpy as np

data_dir = Path(Path.home(),'data','Include') if 'Users/gp' in str(Path.home()) else Path('D','CNC_Audio','gonza','data','Include')

for file in data_dir.glob('*.csv'):
    if 'norm' in file.stem:
        continue
    df = pd.read_csv(file)
    features = [col for col in df.columns if col != 'word' and 'norm' not in col]
    for feature in features:
        if feature not in ['familiarity','imageability','concreteness','valence','dominance','aruousal']:
            df[f'{feature}_standard_norm'] = (df[[feature]] - np.nanmean(df[[feature]]))/np.nanstd(df[[feature]])
            
            df[f'{feature}_log_standard_norm'] = [np.log(x) if not np.isnan(x) else x for x in df[feature]]
            df[f'{feature}_log_standard_norm'] = (df[f'{feature}_log_standard_norm'] - np.nanmean(df[f'{feature}_log_standard_norm']))/np.nanstd(df[f'{feature}_log_standard_norm'])
            df[f'{feature}_robust_norm'] = (df[[feature]] - np.nanmedian(df[[feature]]))/np.nanpercentile(df[[feature]],75)-np.nanpercentile(df[[feature]],25)
            df[f'{feature}_log_robust_norm'] = [np.log(x) if not np.isnan(x) else x for x in df[feature]]
            df[f'{feature}_log_robust_norm'] = (df[f'{feature}_log_robust_norm']-np.nanmedian(df[f'{feature}_log_robust_norm']))/np.nanpercentile(df[f'{feature}_log_robust_norm'],75)-np.nanpercentile(df[f'{feature}_log_robust_norm'],25)
        else:
            df[f'{feature}_minmax_norm'] = (df[[feature]] - np.nanmin(df[[feature]]))/(np.nanmax(df[[feature]])-np.nanmin(df[[feature]]))
        
    df.to_csv(Path(file.parent,f'{file.stem}_norm.csv'), index=False)
    print(f'Normalized {file.name}')