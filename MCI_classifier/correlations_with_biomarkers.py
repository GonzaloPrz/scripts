import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, shapiro
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests
import itertools
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def filter_outliers(data,parametric,n_sd=3):
    for f,feature in enumerate(data.columns):
        if parametric[f]:
            data = data[np.abs(data[feature]-np.nanmean(data[feature]))/np.nanstd(data[feature]) < n_sd]
        else:
            iqr = np.abs(np.nanpercentile(data[feature],q=75) - np.nanpercentile(data[feature],q=25))
            data = data[np.abs(data[feature] - np.nanmedian(data[feature]))/iqr < 1.5]
    
    return data

data_dir = Path(Path.home(),'data','MCI_classifier_unbalanced') if 'Users/gp' in str(Path.home()) else Path(Path.home(),'D:','CNC_Audio','gonza','data','MCI_classifier_unbalanced')
Path(str(data_dir).replace('data','results'),'plots').mkdir(exist_ok=True,parents=True)

no_outliers = False

all_data = pd.read_csv(Path(data_dir,'data_matched_unbalanced_group.csv'))

only_mci = False
all_data = all_data[all_data['group'] == 1] if only_mci else all_data

biomarkers = [col for col in all_data.columns if 'ptau' in col]

features = ['word_properties__log_frq_mean','word_properties__num_syll_mean','word_properties__granularity_mean','word_properties__phon_neigh_mean','speech_timing__npause']
tasks = ['fas','animales']
results = []

for task,feature,biomarker in itertools.product(tasks,features,biomarkers):    
    valid_data = all_data[[f'{task}__{feature}',biomarker]].dropna()

    _, pval_feature = shapiro(valid_data[f'{task}__{feature}'])
    _, pval_biomarker = shapiro(valid_data[biomarker])
    parametric = [p_val > .1 for p_val in [pval_feature,pval_biomarker]]
    
    if no_outliers:
        valid_data = filter_outliers(valid_data,parametric=parametric,n_sd=2)

        _, pval_feature = shapiro(valid_data[f'{task}__{feature}'])
        _, pval_biomarker = shapiro(valid_data[biomarker])
        
    method = 'pearson' if pval_feature > 0.1 and pval_biomarker > 0.1 else 'spearman'
    #method = 'pearson'

    corr, pval = pearsonr(valid_data[f'{task}__{feature}'], valid_data[biomarker]) if method == 'pearson' else spearmanr(valid_data[f'{task}__{feature}'], valid_data[biomarker])
    
    results.append({'feature': f'{task}__{feature}', 'biomarker': biomarker, 'method': method, 'correlation': corr, 'p_value': pval, 'n': len(valid_data)})
    
results_df = pd.DataFrame(results)

pvals = results_df['p_value'].values
reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
results_df['p_value_corrected'] = pvals_corrected
results_df['reject_null'] = reject

filename_to_save = 'correlations_with_biomarkers_mci_no_outliers.csv'
if not no_outliers:
    filename_to_save = filename_to_save.replace('_no_outliers','')

if not only_mci:
    filename_to_save = filename_to_save.replace('_mci','')

results_df.to_csv(Path(str(data_dir).replace('data','results'),filename_to_save))

results_df = results_df.sort_values(by='p_value',ascending=True).iloc[:1]

biomarkers_plot = {'bio__bio__ptau217':'pTau217'}
features_plot = {'animales__word_properties__granularity_mean':'Mean granularity'}

for _, row in results_df.iterrows():
    feature = row['feature']
    biomarker = row['biomarker']
    method = row['method']
    corr = row['correlation']
    pval = row['p_value']
    n = row['n']

    # Extraer datos originales
    task, feat = feature.split('__', 1)
    plot_data = all_data[[f'{feature}', biomarker]].dropna()

    _, pval_feature = shapiro(plot_data[f'{feature}'])
    _, pval_biomarker = shapiro(plot_data[biomarker])
    parametric = [p_val > .1 for p_val in [pval_feature,pval_biomarker]]
    
    if filter_outliers:
        plot_data = filter_outliers(plot_data,parametric=parametric,n_sd=2)

        _, pval_feature = shapiro(plot_data[f'{feature}'])
        _, pval_biomarker = shapiro(plot_data[biomarker])
        
    method = 'pearson' if pval_feature > .1 and pval_biomarker > .1 else 'spearman'
    #method = 'pearson'

    plt.figure(figsize=(7, 6))
    plt.rcParams.update({'font.family': 'Arial'})  # Cambia todo a Arial
    ax = sns.regplot(
        x=f'{feature}',
        y=biomarker,
        data=plot_data,
        scatter_kws={'s': 70, 'alpha': 0.85, 'color': '#005a8d', 'edgecolor': 'white'},
        line_kws={'color': '#d62728', 'lw': 2.5},
        ci=95 if method == 'pearson' else None
    )
    ax.set_xlabel(features_plot.get(feature, feature.replace('_', ' ')), fontsize=16)
    ax.set_ylabel(biomarkers_plot.get(biomarker, biomarker.replace('_', ' ')), fontsize=16)
    ax.set_facecolor('#f7f7f7')
    #plt.title(f'{features_plot.get(feature, feature)} vs {biomarkers_plot.get(biomarker, biomarker)}', fontsize=18, pad=20)

    # Agrandar los label ticks
    ax.tick_params(axis='both', labelsize=15)

    # Cuadro de texto con r y p-value (más grande)
    textstr = f'$r$ = {corr:.2f}\n$p$ = .049'
    props = dict(boxstyle='round,pad=1.0', facecolor='white', edgecolor='black', linewidth=1.5)
    ax.text(0.80, 0.80, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    filename = f'plot_{feature}_vs_{biomarker}_mci_no_outliers'

    if not no_outliers:
        filename = filename.replace('_no_outliers','')
    if not only_mci:
        filename = filename.replace('_mci','')

    plt.savefig(Path(str(data_dir).replace('data','results'),'plots',f'{filename}.png'), dpi=300)
    plt.savefig(Path(str(data_dir).replace('data','results'),'plots',f'{filename}.jpg'), dpi=300)
    plt.close()