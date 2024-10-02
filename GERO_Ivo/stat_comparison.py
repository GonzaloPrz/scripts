import pandas as pd
import numpy as np
import pickle
from pingouin import rm_anova
from pathlib import Path

results_dir = Path(Path.home(),'results','GERO_Ivo') if 'Users/gp' in str(Path.home()) else Path('D:','results','GERO_Ivo')

metrics = pickle.load(open(Path(results_dir,'all_metrics.pkl'),'rb'))

results = rm_anova(data=metrics,dv=metrics.columns[-1],within=['task','dimension'],subject='bootstrap',detailed=True)

print(results)