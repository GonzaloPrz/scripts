import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import ttest_ind,ttest_rel,mannwhitneyu, wilcoxon
import numpy as np
import itertools

# Load the dataset
project_name = 'MCI_classifier_unbalanced'
datafile = 'data_matched_unbalanced_group.csv'
output = 'group'
tasks = ['animales']

#covars = ['nps__TMT__B_Time','nps__TMT__B_Total_Errors','nps__memory__FCRST_Total_Immediate_Free_Recall','nps__memory__FCRST_Total_Recall']
covars = []

single_subsets = ['valid_responses','word_properties']
all_subsets = single_subsets.copy()
if len(all_subsets) > 1:
    for subsets in itertools.combinations(single_subsets,2):
        all_subsets.append(f"{subsets[0]}__{subsets[1]}")

    all_subsets.append('__'.join(single_subsets))

stats_to_remove = ['min','max','median','kurtosis','skewness','std']
#stats_to_remove = []

filter_outliers = False

data_dir = Path(Path.home() if "Users/gp" in str(Path.home())else r'D:\CNC_Audio\gonza','data',project_name)
data = pd.read_csv(Path(data_dir,datafile))

if data.id.nunique() != data.shape[0]:
    print('There are duplicates in the dataset')
    rel = True
else:
    rel = False
#subsets = ['voice_quality','pitch','speech_timing']
ids_to_remove = []

for task, subset in itertools.product(tasks,all_subsets):
    predictors = [col for col in data.columns if (any([f'{x}__{y}__' in col for x,y in itertools.product(task.split('__'),subset.split('__'))])) and (all(f'_{x}' not in col for x in stats_to_remove))] + covars
    data = data.dropna(subset=predictors)

    # Define the dependent and independent variables
    # Assuming 'target' is the dependent variable and the rest are independent variables
    
    #Check normality and perform the group test comparison for each predictor
    for predictor in predictors:
        
        #filter outliers
        if filter_outliers:
            iqr = np.nanpercentile(data[predictor],75) - np.nanpercentile(data[predictor],25)
            X = data.loc[np.abs(data[predictor] - np.nanmedian(data[predictor])) < 1.5*iqr,predictor]
            y = data.loc[np.abs(data[predictor] - np.nanmedian(data[predictor])) < 1.5*iqr,output]
            ids_to_remove += list(data.loc[np.abs(data[predictor] - np.nanmedian(data[predictor])) > 1.5*iqr,'id'])
        else:
            X = data[predictor]
            y = data[output]
        shapiro_results = shapiro(X)
        if shapiro_results[1] > 0.05:    
            if rel:
                result = ttest_rel(X[y==0],X[y==1])
                test = 'Dependent samples t-test'
            else:
                result = ttest_ind(X[y==0],X[y==1])
                test = 'Independent samples t-test'
        else:
            if rel:
                result = wilcoxon(X[y==0],X[y==1])
                test = 'Wilcoxon test'
            else:
                result = mannwhitneyu(X[y==0],X[y==1])
                test = 'Mann-Whitney U test'
        print(f"{test} for {predictor}: stat = {result[0].round(2)}, p-value = {result[1].round(3)}")

    data_without_outliers = data.loc[~data.id.isin(ids_to_remove)]
    X = data_without_outliers[predictors]
    y = data_without_outliers[output]    
    # Add a constant to the independent variables matrix (for the intercept)
    X = sm.add_constant(X)

    model = sm.GLM(y, X, family=sm.families.Binomial(),groups=data_without_outliers['id'] if rel else None)

    # Train the model
    result = model.fit()

    print(result.summary())