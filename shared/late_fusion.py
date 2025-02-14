import pickle, sys, json, os, itertools
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.impute import KNNImputer

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

import utils

parallel = True 
cmatrix = None

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
id_col = config['id_col']
bayesian = config['bayesian']
random_seeds_train = config['random_seeds_train']
random_seeds_shuffle = config["random_seeds_shuffle"]

stratify = config['stratify']
n_splits = int(config['n_folds'])

home = Path(os.environ.get("HOME", Path.home()))
if "Users/gp" in str(home):
    results_dir = home / 'results' / project_name
else:
    results_dir = Path(r"D:/",r"CNC_Audio/gonza/results", project_name)

main_config = json.load(Path(Path(__file__).parent,'main_config.json').open())

y_labels = main_config['y_labels'][project_name]
tasks = main_config['tasks'][project_name]
test_size = main_config['test_size'][project_name]
dimensions = main_config['single_dimensions'][project_name]
data_file = main_config['data_file'][project_name]
thresholds = main_config['thresholds'][project_name]
scoring_metrics = [main_config['scoring_metrics'][project_name]]
problem_type = main_config['problem_type'][project_name]
random_seeds_test = [""] if test_size == 0 else range(int(config["n_seeds_test"]))

##---------------------------------PARAMETERS---------------------------------##
results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

combinations = []

for ndim in range(2,len(dimensions)+1): 

    for combination in itertools.combinations(dimensions,ndim):
        combinations.append(list(combination))
    
for task, y_label in itertools.product(tasks,y_labels):    
    for random_seed in random_seeds_test:

        if not isinstance(random_seed,str):
            random_seed = f'random_seed_{int(random_seed)}'

        for scoring,combination in itertools.product(scoring_metrics,combinations):
            combination_label = '__'.join([comb for comb in combination])
            path_to_save = Path(results_dir,task,combination_label,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '',random_seed,'shuffle' if shuffle_labels else '','late_fusion')
            path_to_save.mkdir(parents=True,exist_ok=True)

            best_models_file = f'best_models_{scoring}_{kfold_folder}_{scaler_name}_{stat_folder}_hyp_opt_feature_selection_shuffle.csv'.replace('__','_')
            if not feature_selection:
                    best_models_file = best_models_file.replace('feature_selection_','')
            if not shuffle_labels:
                best_models_file = best_models_file.replace('_shuffle','')
            if not hyp_opt:
                best_models_file = best_models_file.replace('hyp_opt','no_hyp_opt')

            best_models = pd.read_csv(Path(results_dir,best_models_file))

            for rss, random_seed_shuffle in enumerate(random_seeds_shuffle):
                for r, random_seed_train in enumerate(random_seeds_train):
                    iterator = StratifiedKFold(n_splits=n_splits,shuffle=True) if problem_type == 'clf' and stratify else KFold(n_splits=n_splits,shuffle=True)

                    late_fusion_dev = pd.DataFrame()
                    late_fusion_test = pd.DataFrame()

                    for dimension in combination:                        
                        #if Path(path_to_save,'all_models.csv').exists():
                        #    print(f"Late fusion already done for {task} - {combination_label} - {y_label}. Skipping...")
                        #    continue

                        best_model = best_models[(best_models['task'] == task) & (best_models['y_label'] == y_label) & (best_models['dimension'] == dimension)]

                        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,stat_folder,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection','shuffle')
                        if not feature_selection:
                            path = str(path).replace('feature_selection_','')
                        if not shuffle_labels:
                            path = str(path).replace('shuffle','')
                                                
                            try:
                                model_name = best_model['model_type'].values[0]
                                model_index = best_model['model_index'].values[0]
                            except:
                                continue
                            with open(Path(path,random_seed,'bayesian' if bayesian else '','y_dev.pkl'),'rb') as f:
                                y_dev = pickle.load(f)
                            with open(Path(path,random_seed,'bayesian' if bayesian else '',f'outputs_{model_name}.pkl'),'rb') as f:
                                outputs_dev = pickle.load(f)
                            with open(Path(path,random_seed,'bayesian' if bayesian else '','IDs_dev.pkl'),'rb') as f:
                                IDs_dev = pickle.load(f)
                            try:
                                with open(Path(path,random_seed,'bayesian' if bayesian else '','y_test.pkl'),'rb') as f:
                                    y_test = pickle.load(f)
                                with open(Path(path,random_seed,'bayesian' if bayesian else '','IDs_test.pkl'),'rb') as f:
                                    IDs_test = pickle.load(f)
                                with open(Path(path,random_seed,'bayesian' if bayesian else '',f'outputs_test_{model_name}.pkl'),'rb') as f:
                                    outputs_test = pickle.load(f)
                            except:
                                pass
                        
                        for ndim in range(outputs_dev.shape[-1]-1):
                            late_fusion_test = pd.concat((late_fusion_test,pd.DataFrame({f'outputs_{dimension}_{ndim}':outputs_test[model_index,...,ndim].squeeze()})),axis=1)
                            late_fusion_dev = pd.concat((late_fusion_dev,pd.DataFrame({f'outputs_{dimension}_{ndim}':outputs_dev[rss,model_index,r,...,ndim].squeeze()})),axis=1)
                    #if Path(path_to_save,'all_models.csv').exists():
                    #    print(f"Late fusion already done for {task} - {combination_label} - {y_label}. Skipping...")
                    #    continue

                    model_params, outputs, y_dev_ , IDs_dev_ = utils.CV(LR,{'C':1,'random_state':42}, StandardScaler, KNNImputer, late_fusion_dev, pd.DataFrame(y_dev[rss,r,:]), late_fusion_dev.columns, None, iterator, [int(random_seed_train)], IDs_dev[rss,r,:], cmatrix=None, priors=None, problem_type='clf',parallel=False)

                    if rss == 0 and r == 0:
                        outputs_late_fusion = np.empty((len(random_seeds_shuffle),1,len(random_seeds_train),y_dev.shape[2],outputs.shape[-1]))
                        y_dev_late_fusion = np.empty((len(random_seeds_shuffle),len(random_seeds_train),y_dev.shape[2]))
                        IDs_dev_late_fusion = np.empty((len(random_seeds_shuffle),len(random_seeds_train),y_dev.shape[2]),dtype='object')
                        all_late_fusion = np.empty((len(random_seeds_shuffle),1,len(random_seeds_train),y_dev.shape[2],(outputs.shape[-1]-1)*len(combination)))

                    outputs_late_fusion[rss,:,r,:,:] = outputs
                    y_dev_late_fusion[rss,r,:] = y_dev_[0,:].ravel()
                    IDs_dev_late_fusion[rss,r,:] = IDs_dev_[0,:]
                    try:
                        all_late_fusion[rss,0,r,...] = late_fusion_dev.values
                    except:
                        print('.')
            #if Path(path_to_save,'all_models_lr.csv').exists():
            #    print(f"Late fusion already done for {task} - {combination_label} - {y_label}. Skipping...")
            #   continue
            
            pd.DataFrame(model_params).to_csv(Path(path_to_save,'all_models_lr.csv'))

            with open(Path(path_to_save,'y_dev.pkl'),'wb') as f:
                pickle.dump(y_dev_late_fusion,f)
            with open(Path(path_to_save,f'outputs_lr.pkl'),'wb') as f:
                pickle.dump(outputs_late_fusion,f)
            with open(Path(path_to_save,'IDs_dev.pkl'),'wb') as f:
                pickle.dump(IDs_dev_late_fusion,f) 
            with open(Path(path_to_save,'X_dev.pkl'),'wb') as f:
                pickle.dump(all_late_fusion,f)
            try:
                with open(Path(path_to_save,'y_test.pkl'),'wb') as f:
                    pickle.dump(y_test,f)
                with open(Path(path_to_save,'IDs_test.pkl'),'wb') as f:
                    pickle.dump(IDs_test,f)
                with open(Path(path_to_save,'X_test.pkl'),'wb') as f:
                    pickle.dump(late_fusion_test,f)
            except:
                pass    