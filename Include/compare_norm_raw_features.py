import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

data_dir = Path(Path.home(),'data','Include') if 'Users/gp' in str(Path.home()) else Path('D','CNC_Audio','gonza','data','Include')

Path(data_dir,'figures').mkdir(exist_ok=True)

for file in data_dir.glob('*_norm.csv'):
    language = file.stem.split('_')[-2]
    Path(data_dir,'figures',language).mkdir(exist_ok=True)
    print(file)
    df = pd.read_csv(file)
    features = [col for col in df.columns if 'norm' not in col and col != 'word']

    for feature in features:
        fig = plt.figure()
        #Subplots
        raw_feature = df[feature].dropna()
        try:
            norm_feature = df[f'{feature}_standard_norm'].dropna()
        except:
            norm_feature = df[f'{feature}_minmax_norm'].dropna()
        
        plt.subplot(2,1,1)
        #sns.histplot(raw_feature, bins=50, alpha=0.5, label='Raw')
        sns.kdeplot(raw_feature, label='Raw')
        plt.legend()
        plt.subplot(2,1,2)
        #sns.histplot(norm_feature, bins=50, alpha=0.5, label='Normalized')
        sns.kdeplot(norm_feature, label='Normalized')
        try:
            robust_norm_feature = df[f'{feature}_robust_norm'].dropna()
            robust_log_norm_feature = df[f'{feature}_log_robust_norm'].dropna()
            standard_log_norm_feature = df[f'{feature}_log_standard_norm'].dropna()
            #sns.histplot(robust_norm_feature, bins=50, alpha=0.5, label='Robust Normalized')
            sns.kdeplot(robust_norm_feature, label='Robust Normalized')
            #sns.histplot(robust_log_norm_feature, bins=50, alpha=0.5, label='Log Robust Normalized')
            sns.kdeplot(robust_log_norm_feature, label='Log Robust Normalized')
            #sns.histplot(standard_log_norm_feature, bins=50, alpha=0.5, label='Log Standard Normalized')
            sns.kdeplot(standard_log_norm_feature, label='Log Standard Normalized')
        except:
            pass
        plt.legend()
        plt.savefig(Path(data_dir,'figures',language,f'{feature}.png'))
        plt.close(fig)
