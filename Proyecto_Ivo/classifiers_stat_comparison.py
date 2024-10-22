import pandas as pd
from pathlib import Path
import numpy as np
from mlxtend.evaluate import cochrans_q

tasks = ['']
dimensions = ['NLP_temporal',
              'nps_correctas']

models = {'Animales':'xgb',
          'cog':'lr',
          'AAL':'lr'
          }

kfold_folder = '5_folds'
scaler_name = 'StandardScaler'
y_label = 'target'

df_stat = pd.DataFrame(columns=['roc_auc_dev','roc_auc_holdout','accuracy_dev','accuracy_holdout'])

scores_dev = pd.DataFrame(columns=[f'ID_{task}' for task in tasks] + [f'y_true_{task}' for task in tasks] + [f'y_pred_{task}' for task in tasks])
scores_holdout = pd.DataFrame(columns=[f'ID_{task}' for task in tasks] + [f'y_true_{task}' for task in tasks] + [f'y_pred_{task}' for task in tasks])

for task in tasks:
    path_to_data = Path(Path(__file__).parent,task,dimensions[task],scaler_name,kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'hyp_opt')

    best_model = pd.read_excel(Path(Path(__file__).parent,'hyp_opt',f'{n_seeds_test}_seeds_test','by_roc_auc_inf_val',kfold_folder,f'{task}_{dimensions[task]}_conf_int_95_{models[task]}_hyp_opt_test_30.xlsx')).loc[0]

    params = [col for col in best_model.index if 'inf' not in col and 'sup' not in col and 'mean' not in col and 'std' not in col]

    data_dev = pd.read_csv(Path(path_to_data,f'all_scores_val_{models[task]}.csv'))
    #data_holdout = pd.read_excel(Path(path_to_data,f'all_scores_test_{models[task]}.xlsx'))

    for param in params:
        if param == 'C':
            best_model[param] = np.round(best_model[param],3)
            data_dev[param] = np.round(data_dev[param],3)

        data_dev = data_dev[data_dev[param] == best_model[param]].reset_index(drop=True)
        #data_holdout = data_holdout[data_holdout[param] == best_model[param]]
    data_dev[f'ID_{task}'] = data_dev['ID'].astype(str)
    data_dev[f'y_true_{task}'] = data_dev['y_true'].astype(int)
    data_dev[f'y_pred_{task}'] = data_dev['y_pred'].astype(int)

    #data_holdout = data_holdout.reset_index(drop=True)    

    #models_dev[task] = data_dev[['ID',f'y_pred_{task}',f'y_true_{task}']]
    #models_holdout[task] = data_holdout[['ID','y_pred','y_true']]

    if scores_dev.empty:
        scores_dev = data_dev[[f'ID_{task}',f'y_true_{task}',f'y_pred_{task}']].sort_values(by=f'ID_{task}').reset_index(drop=True).drop_duplicates(f'ID_{task}').copy()
    else:
        scores_dev = pd.merge(scores_dev,data_dev[[f'ID_{task}',f'y_true_{task}',f'y_pred_{task}']],right_on=f'ID_{task}',left_on=f'ID_{last_task}',how='inner')
        scores_dev = scores_dev.drop_duplicates(f'ID_{task}').copy()
    last_task = task
    '''
    if scores_holdout.empty:
        scores_holdout = data_holdout[[f'ID_{task}',f'y_true_{task}',f'y_pred_{task}']]
    else:
        scores_holdout = pd.merge(scores_holdout,data_holdout[[f'ID_{task}',f'y_true_{task}',f'y_pred_{task}']],on=f'ID_{task}')
    '''

scores_dev.to_csv(Path(Path(__file__).parent,'hyp_opt',f'{n_seeds_test}_seeds_test','scores_dev.csv'),index=False)
#Cochran's Q test

y_true = scores_dev[f'y_true_{tasks[0]}'].values
y_pred_model1 = scores_dev[f'y_pred_{tasks[0]}'].values
y_pred_model2 = scores_dev[f'y_pred_{tasks[1]}'].values
y_pred_model3 = scores_dev[f'y_pred_{tasks[2]}'].values

q_stat, p_value = cochrans_q(y_true,y_pred_model1,y_pred_model2,y_pred_model3)

print(f'Cochran\'s Q test: {q_stat} p-value: {p_value}')