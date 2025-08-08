import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# --- Configuration ---
project_name = 'MCI_classifier'
data_dir = Path(Path.home(),'data',project_name) if 'Users/gp' in str(Path.home()) else Path(Path.home(),'D:','CNC_Audio','gonza','data',project_name)

matched_data = pd.read_csv(Path(data_dir,'matched_data.csv'))[['id','target']]
biomarkers = pd.read_excel(Path(data_dir,'biomarkers_data.xlsx'))
df = pd.merge(matched_data,biomarkers,on='id',how='inner')
apoe_data = pd.read_excel(Path(data_dir,'apoe_data.xlsx'))[['id','APOE']]
# Convert APOE to categorical labels

apoe_data['APOE'] = LabelEncoder().fit_transform(apoe_data['APOE'])
df = pd.merge(df,apoe_data,on='id',how='inner')

# Define dependent variables and grouping variable
dependent_vars = set(df.columns) - set(['id','target'])  # Replace with your variables
group_var = 'target'  # The column with group labels

results = []
discrete_vars = []
continuous_vars = []
for var in dependent_vars:
    print(f"\nAnalyzing: {var}")

    # Identify if variable is continuous or discrete
    unique_vals = df[var].dropna().unique()
    var_type = 'discrete' if len(unique_vals) <= 10 and all(float(x).is_integer() for x in unique_vals) else 'continuous'
    if var_type == 'discrete':
        discrete_vars.append(var)
    else:
        continuous_vars.append(var)
    # Check for normality within each group
    normality = df.groupby(group_var)[var].apply(lambda x: stats.shapiro(x)[1] > 0.05).all()

    # Determine appropriate test
    group_counts = df[group_var].nunique()

    if group_counts == 2:
        groups = [group[var].dropna() for name, group in df.groupby(group_var)]
        if var_type == 'continuous':
            if normality:
                test_stat, p_val = stats.ttest_ind(*groups)
                test_used = 't-test'
            else:
                test_stat, p_val = stats.mannwhitneyu(*groups)
                test_used = 'Mann-Whitney'
        else:
            test_stat, p_val = stats.chi2_contingency(pd.crosstab(df[group_var], df[var]))[0:2]
            test_used = 'Chi-squared'
    else:
        if var_type == 'continuos':
            if normality:
                test_stat, p_val = stats.f_oneway(*[g[var].dropna() for _, g in df.groupby(group_var)])
                test_used = 'ANOVA'
            else:
                test_stat, p_val = stats.kruskal(*[g[var].dropna() for _, g in df.groupby(group_var)])
                test_used = 'Kruskal-Wallis'
        else:
            contingency_table = pd.crosstab(df[group_var], df[var])
            chi2, p_val, _, _ = stats.chi2_contingency(contingency_table)
            test_stat = chi2
            test_used = 'Chi-squared'
            # Post-hoc test if ANOVA is significant
            if p_val < 0.05:
                posthoc = pairwise_tukeyhsd(df[var], df[group_var])
                print(posthoc)
                posthoc_summary = posthoc.summary()
                print(posthoc_summary)
                posthoc_summary_df = pd.DataFrame(posthoc_summary.data[1:], columns=posthoc_summary.data[0])
                posthoc_summary_df.to_csv(Path(data_dir,f'posthoc_{var}.csv'), index=False)

    results.append({
        'Variable': var,
        'Type': var_type,
        'Normality': normality,
        'Group_Count': group_counts,
        'Test': test_used,
        'Test_Statistic': test_stat,
        'Test_p_value': p_val,
    })

# GLM (OLS as a special case)
if len(discrete_vars) > 0:
    formula = f"C({group_var}) ~ C({discrete_vars[0]})"
    for var in discrete_vars[1:]:
        formula += f" + C({var})"
for var in continuous_vars:
    formula += f" + {var}"

model = smf.glm(formula, data=df, family=sm.families.Binomial()).fit()
glm_summary = model.summary2().tables[1]

# --- Result Table ---

result_df = pd.DataFrame(results)
print("\nSummary Table:")
print(result_df.to_string(index=False))

# Optional: Save to CSV
result_df.to_csv(Path(data_dir,"statistical_results_summary.csv"), index=False)
glm_summary.to_csv(Path(data_dir,"glm_results_summary.csv"))




