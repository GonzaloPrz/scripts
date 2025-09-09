import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, shapiro
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests
import itertools

data_dir = Path(Path.home(),'data','MCI_classifier_unbalanced') if 'Users/gp' in str(Path.home()) else Path(Path.home(),'D:','CNC_Audio','gonza','data','MCI_classifier_unbalanced')

all_data = pd.read_csv(Path(data_dir,'data_matched_unbalanced_group.csv'))

biomarkers = [col for col in all_data.columns if col.startswith('bio__bio__')]

features = ['grandmean__word_properties__log_frq_mean','grandmean__word_properties__num_syll_mean','grandmean__word_properties__granularity_mean',
            'grandmean__word_properties__phon_neigh_mean','grandmean__speech_timing__npause']

results = []

for feature,biomarker in itertools.product(features,biomarkers):    
    valid_data = all_data[[feature,biomarker]].dropna()
    _, pval_feature = shapiro(valid_data[feature])
    _, pval_biomarker = shapiro(valid_data[biomarker])
    
    method = 'pearson' if pval_feature > 0.1 and pval_biomarker > 0.1 else 'spearman'
        
    corr, pval = pearsonr(valid_data[feature], valid_data[biomarker]) if method == 'pearson' else spearmanr(valid_data[feature], valid_data[biomarker])
    
    results.append({'feature': feature, 'biomarker': biomarker, 'method': method, 'correlation': corr, 'p_value': pval, 'n': len(valid_data)})
    
results_df = pd.DataFrame(results)

pvals = results_df['p_value'].values
reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
results_df['p_value_corrected'] = pvals_corrected
results_df['reject_null'] = reject

results_df.to_csv(Path(str(data_dir).replace('data','results'),'correlations_with_biomarkers.csv'), index=False)

results_regression = []

for biomarker in biomarkers:
    slope, intercept, r_value, p_value, std_err = linregress(valid_data[feature], valid_data[biomarker])
    results_regression.append({'feature': feature, 'biomarker': biomarker, 'slope': slope, 'intercept': intercept, 'r_squared': r_value**2, 'p_value': p_value, 'std_err': std_err, 'n': len(valid_data)})