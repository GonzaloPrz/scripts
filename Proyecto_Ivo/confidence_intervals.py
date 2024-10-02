import pandas as pd
import numpy as np
from pathlib import Path
import itertools 

tasks = ['cog','Animales','AAL','P','both']
scaler_name = 'StandardScaler'

l2ocv = False
hyp_opt_list = [False]
feature_selection_list = [False]
bootstrap_list = [True]
n_seeds_test = 10

if l2ocv:
    kfold_folder = 'loocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

n_seeds_train = 10
n_seeds_test = 10

metrics_names = ['roc_auc','accuracy','f1','precision','recall','norm_expected_cost','norm_cross_entropy']

for task in tasks:
    dimensions = [folder for folder in Path(Path(__file__).parent,task).iterdir() if folder.is_dir() if 'vr' not in str(folder)]

    for dimension in dimensions:

        y_labels = [folder.name for folder in Path(dimension,scaler_name,kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test').iterdir() if folder.is_dir()]
        for y_label in y_labels:
            for hyp_opt in hyp_opt_list:
                path_to_data = Path(dimension,scaler_name,kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt')

                files = [file for file in path_to_data.iterdir() if file.is_file() and 'all_results' in file.stem and '_val_' in file.stem] 
                for file in files:
                    df_val = pd.read_excel(file)
                    if Path(file.parent,str(file.stem.replace('_val_','_test_')) + '.xlsx').exists():
                        df_test = pd.read_excel(Path(file.parent,str(file.stem.replace('_val_','_test_')) + '.xlsx'))

                    unique_seeds_train = len(df_val['random_seed_train'].unique())
                    unique_seeds_test = len(df_val['random_seed_test'].unique())
                    unique_seeds = unique_seeds_train*unique_seeds_test
                    unique_bt = len(df_val['bootstrap'].unique())
                    
                    conf_int_results = pd.DataFrame()
                    parameters = set(df_val.columns) - set(metrics_names) - set(['random_seed_train','random_seed_test','bootstrap'])

                    unique_combinations = df_val[list(parameters)].drop_duplicates()

                    for r, row in unique_combinations.iterrows():
                        #Filter the dataframe to get the rows that match the current combination of parameters
                        df_aux_val = df_val[(df_val[list(parameters)] == row).all(axis=1)]
                        if Path(file.parent,str(file.stem.replace('_val_','_test_')) + '.xlsx').exists():
                            df_aux_test = df_test[(df_test[list(parameters)] == row).all(axis=1)]

                        df_append = dict([param,df_aux_val[param].unique()[0]] for param in parameters)

                        for metric in metrics_names:
                            inf_val = np.nanpercentile(df_aux_val[metric],2.5)
                            sup_val = np.nanpercentile(df_aux_val[metric],97.5)
                            mean_val = np.nanmean(df_aux_val[metric])

                            df_append[metric+'_inf_val'] = inf_val.round(3)
                            df_append[metric+'_mean_val'] = mean_val.round(3)
                            df_append[metric+'_sup_val'] = sup_val.round(3)
                            
                            if Path(file.parent,str(file.stem.replace('_val_','_test_')) + '.xlsx').exists():
                                inf_test = np.nanpercentile(df_aux_test[metric],2.5)
                                sup_test = np.nanpercentile(df_aux_test[metric],97.5)
                                mean_test = np.nanmean(df_aux_test[metric])

                                df_append[metric+'_inf_holdout'] = inf_test.round(3)
                                df_append[metric+'_mean_holdout'] = mean_test.round(3)
                                df_append[metric+'_sup_holdout'] = sup_test.round(3)
                        
                        if conf_int_results.empty:
                            conf_int_results = pd.DataFrame(columns = list(df_append.keys()))
                        conf_int_results.loc[len(conf_int_results),:] = df_append

                    filename_to_save = f'{task}_{dimension.name}_' + file.stem.replace('all_results_','conf_int_95_') + '_no_hyp_opt.xlsx'
                    if hyp_opt:
                        filename_to_save = filename_to_save.replace('no_hyp_opt','hyp_opt_test_30').replace('_val','')
                    
                    path_to_save = Path(Path(__file__).parent,'hyp_opt',f'{n_seeds_test}_seeds_test') if hyp_opt else Path(Path(__file__).parent,'no_hyp_opt',f'{n_seeds_test}_seeds_test')

                    conf_int_results = conf_int_results.sort_values(by='roc_auc_inf_val',ascending=False)
                    Path(path_to_save,'by_roc_auc_inf_val',f'{n_folds}_folds').mkdir(parents=True,exist_ok=True)
                    conf_int_results.to_excel(Path(path_to_save,'by_roc_auc_inf_val',f'{n_folds}_folds',filename_to_save),index=False)
                    
