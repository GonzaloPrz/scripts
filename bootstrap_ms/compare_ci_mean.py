import numpy as np
from pathlib import Path

data_dir = Path(Path.home(),'expected_cost','notebooks','data') if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','expected_cost','notebooks','data')

datasets = [folder.name for folder in data_dir.iterdir() if folder.is_dir() and 'cifar' in folder.name]

n_boot = 10000
bootstrap_method = 'bca'

for dataset in datasets:

    scores = np.load(Path(data_dir,dataset,'scores.npy'))
    targets = np.load(Path(data_dir,dataset,'targets.npy'))


