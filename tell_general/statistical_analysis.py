import pandas as pd

import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from warnings import filterwarnings
import numpy as np
import itertools
from pingouin import compute_effsize
from sklearn.metrics import roc_auc_score
import json

filterwarnings('ignore')

correction_method = 'fdr_bh'
min_subjects = 10
alpha = .05

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
            if len(data_category[data_category['group'] == group]) < min_subjects:
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
            
            results_summary = analyze_data(data_category_feature, feature, saving_dir=Path(saving_dir,category,'qq_plots'),group='group',alpha=alpha)
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
reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method=correction_method)
results_df['Corrected P-value'] = np.round(pvals_corrected,3)
results_df['Correction method'] = correction_method
results_df['Significant'] = results_df.apply(lambda x: True if x['Corrected P-value'] < alpha else False,axis=1)

#Reorder columns
results_df = results_df[['Category','Feature','Test','Statistic','P-value','Corrected P-value','Correction method','Significant']]

#results_df.to_csv(Path(saving_dir,'results_stats.csv'))
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
    df = data[data['category'] == category]
    #Keep subjects within group or HC
    df = df[df['group'].isin([group,'HC'])]
    df.dropna(subset=[feature],inplace=True)
    
    if results_df[(results_df['Category'] == category) & (results_df['Feature'] == feature)]['Corrected P-value'].values < alpha:
        if not Path(saving_dir,category,'plots',f'hist_plot_{feature}_group.png').exists():
            descriptive_plots(df,feature,Path(saving_dir,category,'plots'),group='group')
        
        results = analyze_data(df[df['group'].isin([group,'HC'])], feature, saving_dir=None,group='group',alpha=alpha)
        if results['Test'] == 'Mann-Whitney U Test':
            results['effect size type'] = 'AUC ROC'
            df['group'] = df['group'].apply(lambda x: 1 if x == group else 0)
            results['effect size value'] = roc_auc_score(df['group'],df[feature]).round(3)
            results['effect size category'] = None if np.abs(results['effect size value'] - .5) < .056 else 'small' if np.abs(results['effect size value'] - .5) < .14 else 'medium' if np.abs(results['effect size value'] - .5) < .214 else 'large' #Based on https://sci-hub.se/10.1007/s10979-005-6832-7
        else:
            results['effect size type'] = "Cohen's d"
            results['effect size value'] = np.round(compute_effsize(df[df['group'] == group][feature],df[df['group'] == 'HC'][feature],paired=False),3)
            results['effect size category'] = 'small' if np.abs(results['effect size value']) < 0.2 else 'medium' if np.abs(results['effect size value']) < 0.5 else 'large'
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

all_pairwise_results = pd.concat((results_HC_AD,results_HC_PD,results_HC_MCI,results_HC_bvFTD),axis=0)
all_pairwise_results = all_pairwise_results.dropna(subset=['P-value'])
all_pairwise_results['Feature'] = all_pairwise_results.index
pvals = all_pairwise_results['P-value'].values
reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method=correction_method)
all_pairwise_results['P-value'] = np.round(pvals,3)
all_pairwise_results['Corrected P-value'] = np.round(pvals_corrected,3)
all_pairwise_results['Correction method'] = correction_method
all_pairwise_results['Significant'] = all_pairwise_results.apply(lambda x: True if x['Corrected P-value'] < alpha else False,axis=1)
all_pairwise_results.drop('conclusion',axis=1,inplace=True)   

all_pairwise_results = all_pairwise_results[['Comparison','Feature','Category','Test','Statistic','P-value','Corrected P-value','Correction method','Significant','effect size type','effect size value','effect size category']]
all_pairwise_results = all_pairwise_results.sort_values(by=['Comparison','Feature','Category'])
all_pairwise_results.to_csv(Path(saving_dir,'posthoc_stats.csv'),index=False)
significant_results = all_pairwise_results[all_pairwise_results['Significant'] == True]
significant_results.to_csv(Path(saving_dir,'significant_posthoc_stats.csv'),index=False)

#Filter by significant and effect size
significant_results = significant_results[significant_results['effect size category'] != 'small'].groupby(['Category']).apply(lambda x: x[x['effect size category'] != 'small'])
significant_results.to_csv(Path(saving_dir,'significant_posthoc_stats_filtered_medium_large.csv'),index=False)

categories = significant_results['Category'].unique()

show_features = dict((category,list(significant_results[significant_results['Category'] == category]['Feature'].unique())) for category in categories)

number_of_features_to_show = sum([len(show_features[category]) for category in categories])
total_number_of_features = results_df.shape[0]

for r, row in results_df.iterrows():
    if row['Feature'] in show_features[row['Category']]:
        results_df.at[r,'Show'] = True
    else:
        results_df.at[r,'Show'] = False

results_df.to_csv(Path(saving_dir,'results_stats.csv'))


