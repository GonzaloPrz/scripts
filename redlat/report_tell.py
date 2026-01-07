import pandas as pd
from pathlib import Path
import numpy as np

datasets = ['24_javeriana_speech']


for dataset in datasets:
    summary = pd.DataFrame(columns=['id','group','tasks'])

    data_dir = Path('D:','CNC_Audio','data',dataset)

    for subfolder in [folder.name for folder in data_dir.iterdir() if folder.is_dir()]:
        groups = [folder.name.split('_')[0] for folder in (data_dir / subfolder).iterdir() if folder.is_dir() and 'checked' in folder.name]

        for group in groups:

            uids = np.unique([folder.name for folder in (data_dir / subfolder / f'{group}_checked').iterdir() if folder.is_dir()])

            for uid in uids:
                
                summary = pd.concat([summary, pd.DataFrame({'id':[uid],'group':group,'tasks':['N/A']})
                ], ignore_index=True)

    summary.to_csv(Path(data_dir,f'report_tell_{dataset}.csv'), index=False)

