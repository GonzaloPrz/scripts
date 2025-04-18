from pathlib import Path
import pandas as pd

data_dir = Path(Path.home(),'data',Path(__file__).parent.name)

dependency_data = pd.read_csv(Path(data_dir,'arequipa__universal_dependencies.csv'))

vars = [col for col in dependency_data.columns if col != 'filename']

for var in vars:
    dependency_data[f'{var}_norm'] = dependency_data[var]/dependency_data[vars].sum(axis=1) 
    #dependency_data = dependency_data.drop(columns=[var])

dependency_data = dependency_data.drop(columns=vars)
dependency_data.to_csv(Path(data_dir,'arequipa__universal_dependencies.csv'), index=False)