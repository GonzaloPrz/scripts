import pandas as pd
from pathlib import Path
import numpy as np
import itertools,pickle,json
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

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
bayes = bool(config["bayesian"])
random_seeds_train = config["random_seeds_train"]

hyp_opt = True if config["n_iter"] > 0 else False
feature_selection = True if config["n_iter_features"] > 0 else False

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
single_dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = main_config['scoring_metrics'][project_name]
problem_type = main_config['problem_type'][project_name]
models = main_config["models"][project_name]
metrics_names = main_config["metrics_names"][problem_type] 

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

best_best_models = pd.DataFrame()

for scoring in [scoring_metrics]:
    Path(results_dir,'plots',scoring).mkdir(exist_ok=True,parents=True)

    extremo = 'sup' if any(x in scoring for x in ['error','norm']) else 'inf'
    ascending = True if extremo == 'sup' else False

    filename = f'best_models_{scoring}_{kfold_folder}_StandardScaler_{stat_folder}_hyp_opt_feature_selection.csv'.replace('__','_')

    if not Path(results_dir,filename).exists():
        continue

    best_models = pd.read_csv(Path(results_dir,filename))
    best_models[f'{scoring}_{extremo}_dev'] = best_models[f'{scoring}_ic_dev'].map(lambda x: x.replace('[','').replace(']','').split(',')[0 if extremo == 'inf' else 1]).astype(float)
    tasks = best_models['task'].unique()
    y_labels = best_models['y_label'].unique()
    dimensions = best_models['dimension'].unique()

    pearsons_results = pd.DataFrame(columns=['task','dimension','y_label','model_type','r','p_value'])

    for y_label in y_labels:
        
        best_models_y_label = best_models[(best_models['y_label'] == y_label)].sort_values(by=f'{scoring}_{extremo}_dev',ascending=ascending).reset_index(drop=True)
        best_models_y_label = best_models_y_label[best_models_y_label['model_type'].isin(models)].reset_index(drop=True)
        if best_best_models.empty:
            best_best_models = pd.DataFrame(best_models_y_label.loc[0,:])
        else:
            best_best_models = pd.concat([best_best_models,best_models_y_label.loc[0,:]],axis=1)
        
        task = best_models_y_label['task'].values[0]
        dim = best_models_y_label['dimension'].values[0]
        
        random_seed_test = best_models_y_label['random_seed_test'].values[0]
        if np.isnan(random_seed_test):
            random_seed_test = ''

        model_name = best_models[(best_models['task'] == task)  & (best_models['dimension'] == dim) & (best_models['y_label'] == y_label)]['model_type'].values[0]
        
        model_index = best_models[(best_models['task'] == task) & (best_models['dimension'] == dim) &  (best_models['y_label'] == y_label)]['model_index'].values[0]
        print(f'{y_label}_{task}___{dim}___{model_name}')
        
        #Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '','plots').mkdir(exist_ok=True)
        IDs_ = pickle.load(open(Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'mo_hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '',random_seed_test,'IDs_dev.pkl'),'rb'))[0,:]
        y_pred = pickle.load(open(Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '',random_seed_test,f'outputs_{model_name}.pkl'),'rb'))[0,model_index]
        y_true = pickle.load(open(Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '',random_seed_test,'y_dev.pkl'),'rb'))[0,:]
        
        if IDs_.ndim == 1:
            
            IDs = np.empty((len(random_seeds_train),len(IDs_)),dtype=object)

            for i,seed in enumerate(random_seeds_train):
                kf = KFold(n_splits=5,shuffle=True,random_state=int(seed))
                for j,(train_index,test_index) in enumerate(kf.split(IDs_)):
                    IDs[i,test_index] = IDs_[test_index]
        
        else:
            IDs = IDs_

        data = pd.DataFrame({'ID':IDs.flatten(),'y_pred':y_pred.flatten(),'y_true':y_true.flatten()})
        data = data.drop_duplicates('ID')

        # Calculate Pearson's correlation
        r, p = pearsonr(data['y_true'], data['y_pred'])

        plt.figure()
        sns.scatterplot(x='y_true',y='y_pred',data=data)
        plt.xlabel('True vaue')
        plt.ylabel('Predicted value')
        plt.title(f'{model_name} - {y_label} - {dim} - {task}')
        plt.xlim(np.min((data['y_true'].min(),data['y_pred'].min())),np.max((data['y_true'].max(),data['y_pred'].max())))
        plt.ylim(np.min((data['y_true'].min(),data['y_pred'].min())),np.max((data['y_true'].max(),data['y_pred'].max())))
        plt.grid(True)
        # Add the regression line
        a, b = np.polyfit(data['y_true'], data['y_pred'], 1)
        plt.plot(data['y_true'], a * data['y_true'] + b, color='red')

        pearsons_results.loc[len(pearsons_results)] = [task, dim, y_label, model_name, r, p]

        # Add stats to the plot
        plt.text(data['y_true'].min(), data['y_pred'].max(), f'r = {r:.2f}, p = {p:.2e}', fontsize=12)

        # Save the plot
        plt.savefig(Path(results_dir, 'plots',scoring, f'{y_label}_{kfold_folder}_{model_name}_{scoring}.png'))
        plt.close()
    
    best_best_models.T.to_csv(Path(results_dir, f'best_best_models_{scoring}.csv'), index=False)
    pearsons_results.to_csv(Path(results_dir,f'pearons_results_{scoring}.csv'),index=False)
        