import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import ttest_ind,ttest_rel,mannwhitneyu, wilcoxon
import numpy as np

# Load the dataset
project_name = 'MCI_classifier'
datafile = 'data_matched_group.csv'
output = 'group'
subsets = ['speech_timing','word_properties','speech_timing__word_properties']
stats_to_remove = ['median','min','max','skewness','kurtosis']
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

for subset in subsets:
    predictors = [col for col in data.columns if (any([x in col for x in subset.split('__')])) and (all(f'_{x}' not in col for x in stats_to_remove))]
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

    model = sm.GLM(y, X, family=sm.families.Gaussian(),groups=data_without_outliers['id'] if rel else None)

    # Train the model
    result = model.fit()

    print(result.summary())