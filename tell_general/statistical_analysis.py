import pandas as pd

import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from warnings import filterwarnings
import numpy as np
import itertools

filterwarnings('ignore')

correction_method = 'bonferroni'

import sys

sys.path.append(str(Path(Path.home(),'scripts_generales')))

from stat_analysis import *

data_dir = Path(Path.home(),'data','tell_general') if 'Users/gp' in str(Path.home()) else Path('D','CNC_Audio','data','tell_general')
saving_dir = Path(Path.home(),'data','tell_general','stats') if 'Users/gp' in str(Path.home()) else Path('D','CNC_Audio','data','tell_general','stats')

saving_dir.mkdir(parents=True,exist_ok=True)

# Load the dataset into a pandas DataFrame
data = pd.read_csv(Path(data_dir,'features_filtered.csv'))

# Create an empty DataFrame to store the results
results_df = pd.DataFrame()

# Perform statistical analysis on each feature
descriptive_stats = pd.DataFrame()

categories = data['category'].unique()
for category in categories:
    print(category)
    #if Path(saving_dir,category,'results_stats.csv').exists():
    #    continue
    Path(saving_dir,category,'plots').mkdir(parents=True,exist_ok=True)

    data_category = data[data['category'] == category]    
    for feature in data_category.columns:
        groups = data_category['group'].unique()
        for group in groups:
            if len(data_category[data_category['group'] == group]) < 10:
                data_category = data_category[data_category['group'] != group]
            
        data_category_feature = data_category[[feature,'group']].dropna()

        if feature not in ['group','sex','age','task','id','base','category']:
            if isinstance(data_category[feature].iloc[0],str):
                continue
            print(f"Statistical analysis for {feature}:")
            summary_stats = stat_describe(data_category_feature, feature, group='group')
            #Add feature as double index to concatenate the DataFrames
            if summary_stats is None:
                continue
            summary_stats = summary_stats.stack().unstack(0)
            summary_stats['feature'] = feature
            summary_stats['category'] = category
            if descriptive_stats.empty:
                descriptive_stats = pd.DataFrame(summary_stats)
            else:
                descriptive_stats = pd.concat([descriptive_stats, pd.DataFrame(summary_stats)], ignore_index=False)
            
            results_summary = analyze_data(data_category_feature, feature, saving_dir=Path(saving_dir,category,'qq_plots'),group='group',alpha=.05)
            # Add the results to the DataFrame
            results_summary['Category'] = category
            results_summary['Feature'] = feature

            if results_df.empty:
                results_df = pd.DataFrame(results_summary,index=[feature])
            else:
                results_df = pd.concat([results_df, pd.DataFrame(results_summary,index=[feature])], ignore_index=False)

            #plt.savefig(Path(saving_dir,base,category,'qq_plots',f"qq_plot_{feature}_{group}.png"))

            #descriptive_plots(data_category_feature,feature,path_to_save=Path(saving_dir,category,'plots'), group='group')

pvals = results_df['P-value'].values
reject, pvals_corrected, _, _ = multipletests(pvals, alpha=.05, method=correction_method)
results_df['Corrected P-value'] = np.round(pvals_corrected,3)
results_df['Correction method'] = correction_method
results_df['Conclusion'] = results_df.apply(lambda x: "Reject Ho" if x['Corrected P-value'] < .05 else "Fail to Reject" + f" Ho at alpha=.05.",axis=1)

#Reorder columns
results_df = results_df[['Category','Feature','Test','Statistic','P-value','Corrected P-value','Correction method','Conclusion']]

results_df.to_csv(Path(saving_dir,'results_stats.csv'))
descriptive_stats.to_csv(Path(saving_dir,'descriptive_stats.csv'))

results_HC_AD = pd.DataFrame()
results_HC_PD = pd.DataFrame()
results_HC_MCI = pd.DataFrame()
results_HC_bvFTD = pd.DataFrame()

groups = data['group'].unique()

for group,category,feature in itertools.product(groups,results_df['Category'].unique(),results_df['Feature'].unique()):
    print(f"Category: {category}, Feature: {feature}")
    if group == 'HC':
        continue
    df = data.loc[data['category'] == category]
    df.dropna(subset=[feature],inplace=True)
    
    if results_df[(results_df['Category'] == category) & (results_df['Feature'] == feature)]['Corrected P-value'].values < .05:
        if not Path(saving_dir,category,'plots',f'hist_plot_{feature}_group.png').exists():
            descriptive_plots(df,feature,Path(saving_dir,category,'plots'),group='group')
        
        results = analyze_data(df[df['group'].isin([group,'HC'])], feature, saving_dir=None,group='group',alpha=.05)
        if results is None:
            continue
        results['Category'] = category
        results['Comparison'] = f'{group} vs HC'
    else:
        continue
    
    if group == 'AD':
        if results_HC_AD.empty:
            results_HC_AD = pd.DataFrame(results,index=[feature])
        else:
            results_HC_AD = pd.concat([results_HC_AD, pd.DataFrame(results,index=[feature])], ignore_index=False)
    elif group == 'PD':
        if results_HC_PD.empty:
            results_HC_PD = pd.DataFrame(results,index=[feature])
        else:
            results_HC_PD = pd.concat([results_HC_PD, pd.DataFrame(results,index=[feature])], ignore_index=False)
    elif group == 'MCI':
        if results_HC_MCI.empty:
            results_HC_MCI = pd.DataFrame(results,index=[feature])
        else:
            results_HC_MCI = pd.concat([results_HC_MCI, pd.DataFrame(results,index=[feature])], ignore_index=False)
    elif group == 'bvFTD':
        if results_HC_bvFTD.empty:
            results_HC_bvFTD = pd.DataFrame(results,index=[feature])
        else:
            results_HC_bvFTD = pd.concat([results_HC_bvFTD, pd.DataFrame(results,index=[feature])], ignore_index=False)

try:
    pvals = results_HC_AD['P-value'].values
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=.05, method=correction_method)
    results_HC_AD['Corrected P-value'] = np.round(pvals_corrected,3)
    results_HC_AD['Correction method'] = correction_method
    results_HC_AD['Conclusion'] = results_HC_AD.apply(lambda x: "Reject Ho" if x['Corrected P-value'] < .05 else "Fail to Reject" + f" Ho at alpha=.05.",axis=1)
except:
    pass

try:
    pvals = results_HC_PD['P-value'].values
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=.05, method=correction_method)
    results_HC_PD['Corrected P-value'] = np.round(pvals_corrected,3)
    results_HC_PD['Correction method'] = correction_method
    results_HC_PD['Conclusion'] = results_HC_PD.apply(lambda x: "Reject Ho" if x['Corrected P-value'] < .05 else "Fail to Reject" + f" Ho at alpha=.05.",axis=1)
except:
    pass

try:
    pvals = results_HC_MCI['P-value'].values
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=.05, method=correction_method)
    results_HC_MCI['Corrected P-value'] = np.round(pvals_corrected,3)
    results_HC_MCI['Correction method'] = correction_method
except:
    pass

try:
    pvals = results_HC_bvFTD['P-value'].values
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=.05, method=correction_method)
    results_HC_bvFTD['Corrected P-value'] = np.round(pvals_corrected,3)
    results_HC_bvFTD['Correction method'] = correction_method
    results_HC_bvFTD['Conclusion'] = results_HC_bvFTD.apply(lambda x: "Reject Ho" if x['Corrected P-value'] < .05 else "Fail to Reject" + f" Ho at alpha=.05.",axis=1)
except:
    pass

results_HC_AD.to_csv(Path(saving_dir,'posthocs_HC_AD.csv'))
results_HC_PD.to_csv(Path(saving_dir,'posthocs_HC_PD.csv'))
results_HC_MCI.to_csv(Path(saving_dir,'posthocs_HC_MCI.csv'))
results_HC_bvFTD.to_csv(Path(saving_dir,'posthocs_HC_bvFTD.csv'))
