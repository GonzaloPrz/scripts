import pandas as pd
from pathlib import Path
import statsmodels.api as sm
#Import shapiro test
from scipy.stats import shapiro
from scipy.stats import ttest_ind,ttest_rel,mannwhitneyu, wilcoxon
import numpy as np

# Load the dataset
datafile = 'all_data_2.csv'
project_name = 'sj'
output = 'time'

data_dir = Path(Path.home() if "Users/gp" in str(Path.home())else r'D:\CNC_Audio\gonza','data',project_name)
results_dir = Path(str(data_dir).replace('data','results'))
Path(results_dir,datafile.split('.')[0]).mkdir(parents=True, exist_ok=True)

subsets = ['psycholinguistic_objective','talking_intervals','sentiment_analysis','pitch_analysis','verbosity','universal_dependencies']

univariate_comparisons = pd.DataFrame(columns=['predictor','N pre', 'N post','test', 'stat','p-value'])

for subset in subsets:
    data = pd.read_csv(Path(data_dir,datafile))

    if data.id.nunique() != data.shape[0]:
        rel = True
    else:
        rel = False

    predictors = [col for col in data.columns if subset in col and all(x not in col for x in ['_min','_max','_kurtosis','_skewness','_query','_error'])]
    data = data.dropna(subset=predictors)
    kept_ids = set(data.loc[data[output] ==0,'id']).intersection(set(data.loc[data[output] ==1,'id']))
    data = data.loc[data['id'].isin(kept_ids),:]
    # Check if the predictors are numeric
    for predictor in predictors:
        if any(isinstance(x,str) for x in data[predictor]):
            #Drop the predictor if it is not numeric
            predictors = set(predictors) - set([predictor])
            predictors = list(predictors)
    #Check normality and perform the group test comparison for each predictor
    for predictor in predictors:
        print(predictor)
        #filter outliers
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

        univariate_comparisons.loc[univariate_comparisons.shape[0],:] = {'predictor':predictor,'N pre':np.sum(y==0),'N post': np.sum(y==1),'test':test,'stat':result[0].round(2),'p-value':result[1].round(3)}

    
    y = data[output].values
    model = sm.GLM(y, data[predictors], family=sm.families.Gaussian(),groups=data['id'] if rel else None)

    # Train the model
    result = model.fit()

    summary = result.summary()
    summary = pd.DataFrame(summary.tables[1])
    summary['N pre'] = np.sum(y==0)
    summary['N post'] = np.sum(y==1)

    summary.to_csv(Path(results_dir,datafile.split('.')[0],f'{project_name}_{subset}_summary.csv'),index=False)
    
univariate_comparisons.to_csv(Path(results_dir,datafile.split('.')[0],f'{project_name}_univariate_comparisons.csv'),index=False)
    
