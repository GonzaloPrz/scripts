import pandas as pd
from pathlib import Path
from pingouin import partial_corr
import pickle, itertools
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import json,os
from statsmodels.stats.multitest import multipletests

correction = 'fdr_bh'

config = json.load(Path(Path(__file__).parent,'config.json').open())

project_name = config["project_name"]
scaler_name = config['scaler_name']
kfold_folder = config['kfold_folder']
shuffle_labels = config['shuffle_labels']
avoid_stats = config["avoid_stats"]
stat_folder = config['stat_folder']
hyp_opt = True if config['n_iter'] > 0 else False
feature_selection = True if config['n_iter_features'] > 0 else False
filter_outliers = config['filter_outliers']
n_models = int(config["n_models"])
n_boot = int(config["n_boot"])
early_fusion = bool(config["early_fusion"])
problem_type = config["problem_type"]

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path("D:/CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
problem_type = main_config['problem_type'][project_name]
covars = main_config["covars"][project_name]
id_col = main_config["id_col"][project_name]

if isinstance(scoring_metrics,str):
    scoring_metrics = [scoring_metrics]

covar_data = pd.read_csv(Path(str(results_dir).replace('results','data'),data_file))[[id_col]+covars]
pearsons_results = pd.DataFrame(columns=['task','dimension','y_label','model_type','r','p_value','covars'])

for y_label,scoring in itertools.product(y_labels,scoring_metrics):
    best_models_file = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection.csv'.replace('__','_')
    if not hyp_opt:
        best_models_file = best_models_file.replace('_hyp_opt','_no_hyp_opt')
    if not feature_selection:
        best_models_file = best_models_file.replace('_feature_selection','')
    
    best_models = pd.read_csv(Path(results_dir,best_models_file))   
    
    extremo = 'sup' if any(x in scoring for x in ['error','norm']) else 'inf'
    ascending = True if extremo == 'sup' else False

    best_models_y_label = best_models[best_models['y_label'] == y_label].sort_values(by=f'{scoring}_mean_dev',ascending=ascending).reset_index(drop=True)
    tasks = best_models_y_label['task'].unique()
    dimensions = best_models_y_label['dimension'].unique()

    for task,dimension in itertools.product(tasks,dimensions):
        model_name = best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label)]['model_type'].values[0]
        model_index = best_models[(best_models['task'] == task) & (best_models['dimension'] == dimension) & (best_models['y_label'] == y_label)]['model_index'].values[0]
        print(f'{y_label}__{task}__{dimension}__{model_name}')
        
        #Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '','plots').mkdir(exist_ok=True)
        IDs_ = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '','IDs_dev.pkl'),'rb'))
        if IDs_.ndim == 1:
            IDs = np.empty((10,len(IDs_)),dtype=object)
            for r,random_seed in enumerate([3**x for x in np.arange(1,11)]):
                CV = KFold(n_splits=5,shuffle=True,random_state=random_seed)
                for train_index, test_index in CV.split(IDs_):
                    IDs[r,test_index] = IDs_.loc[test_index]
        else:
            IDs = IDs_.copy()

        y_pred = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '',f'outputs_{model_name}.pkl'),'rb'))[0,model_index,]
        y_true = pickle.load(open(Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '','y_dev.pkl'),'rb'))[0]

        data = pd.DataFrame({id_col:IDs.flatten(),'y_pred':y_pred.flatten(),'y_true':y_true.flatten()})
        
        data = pd.merge(data,covar_data,on=id_col)
        data = data.drop_duplicates(id_col)

        for covar in covars:
            if isinstance(data.loc[0,covar],str):
                data[covar] = LabelEncoder().fit_transform(data[covar])
                
        if y_label in covars:
            covars = list(set(covars) - set([y_label]))
        
        stats = partial_corr(data=data,x='y_pred',y='y_true',covar=covars,method='pearson')

        pearsons_results.loc[len(pearsons_results.index),:] = {'task':task,'dimension':dimension,'y_label':y_label,'model_type':model_name,'r':stats['r'].values[0],'p_value':stats['p-val'].values[0],'covars':str(covars)}

p_vals = pearsons_results['p_value'].values
reject, p_vals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method=correction)
pearsons_results['p_value_corrected'] = p_vals_corrected
pearsons_results['correction_method'] = correction
pearsons_results.to_csv(Path(results_dir,f'partial_pearsons_results_{scoring}_{kfold_folder}s.csv'),index=False)

