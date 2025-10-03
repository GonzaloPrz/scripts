import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

projects = ['53_ceac']

scoring = 'r2_score'

for project in projects:
    results_dir = Path(Path.home(),'results',project) if '/Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project)
    all_models = pd.DataFrame()
    for file in results_dir.glob('best_models_*.csv'):
        models = pd.read_csv(file)
        
        models[f'{scoring}_mean'] = models[scoring].map(lambda x: float(x.split(', ')[0]))
        models[f'{scoring}_sup'] = models[scoring].map(lambda x: float(x.split('(')[1].split(', ')[1].replace(')','')))

        all_models = pd.concat((all_models,models))
    
    fig = plt.figure()
    sns.scatterplot(data=models,x=f'{scoring}_mean',y=f'{scoring}_sup')
    
    plt.show()
