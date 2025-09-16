import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, shapiro
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests
import itertools
import statsmodels.api as sm
data_dir = Path(Path.home(),'data','MCI_classifier_unbalanced') if 'Users/gp' in str(Path.home()) else Path(Path.home(),'D:','CNC_Audio','gonza','data','MCI_classifier_unbalanced')

all_data = pd.read_csv(Path(data_dir,'data_matched_unbalanced_group.csv'))

only_mci = False
all_data = all_data[all_data['group'] == 1] if only_mci else all_data

biomarkers = ['bio__bio__ptau217']

features = ['word_properties__log_frq_mean','word_properties__num_syll_mean','word_properties__granularity_mean','word_properties__phon_neigh_mean','speech_timing__npause']
tasks = ['grandmean','fas','animales']
results = []

for task,feature,biomarker in itertools.product(tasks,features,biomarkers):    
    valid_data = all_data[[f'{task}__{feature}',biomarker]].dropna()
    _, pval_feature = shapiro(valid_data[f'{task}__{feature}'])
    _, pval_biomarker = shapiro(valid_data[biomarker])
    
    method = 'pearson' if pval_feature > 0.1 and pval_biomarker > 0.1 else 'spearman'
    
    corr, pval = pearsonr(valid_data[f'{task}__{feature}'], valid_data[biomarker]) if method == 'pearson' else spearmanr(valid_data[f'{task}__{feature}'], valid_data[biomarker])
    
    results.append({'feature': f'{task}__{feature}', 'biomarker': biomarker, 'method': method, 'correlation': corr, 'p_value': pval, 'n': len(valid_data)})
    
results_df = pd.DataFrame(results)

pvals = results_df['p_value'].values
reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
results_df['p_value_corrected'] = pvals_corrected
results_df['reject_null'] = reject

results_df.to_csv(Path(str(data_dir).replace('data','results'),'correlations_with_biomarkers_mci.csv' if only_mci else 'correlations_with_biomarkers.csv'), index=False)

for task,biomarker in itertools.product(tasks,biomarkers):
    valid_data = all_data[[biomarker] + [f'{task}__{feature}' for feature in features]].dropna()
    model = sm.OLS(valid_data[biomarker], valid_data[[f'{task}__{feature}' for feature in features]])
    results = model.fit()
    print(f'Biomarker: {biomarker}')
    print(results.summary())
