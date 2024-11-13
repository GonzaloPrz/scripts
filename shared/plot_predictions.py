import pandas as pd
from pathlib import Path
import numpy as np
import itertools,pickle
import seaborn as sns
import matplotlib.pyplot as plt

project_name = 'GeroApathy'
l2ocv = False

cmatrix = None
shuffle_labels = False
bayes_list = [True]
feature_selection_list = [True]

id_col = 'id'
scaler_name = 'StandardScaler'

models = {'MCI_classifier':['lr','svc','knnc','xgb'],
          'tell_classifier':['lr','svc','knnc','xgb'],
          'Proyecto_Ivo':['lr','svc','knnr','xgb'],
          'GeroApathy':['elastic','lasso','ridge','knnr','svr','xgb'],
            'GERO_Ivo':['elastic','lasso','ridge','knnr','svr']
            }

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'],
         'Proyecto_Ivo':['Animales','P','Animales__P','cog','brain','AAL','conn'],
         'GeroApathy':['DiaTipico'],
         'GERO_Ivo':['animales','grandmean','fas__animales','fas']
         }

single_dimensions = {'tell_classifier':['voice-quality','talking-intervals','pitch'],
                     'MCI_classifier':['talking-intervals','psycholinguistic'],
                     'Proyecto_Ivo':[],
                     'GERO_Ivo':[],
                     'GeroApathy':[]}

metrics_names = {'MCI_classifier':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'tell_classifier':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'Proyecto_Ivo':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'GeroApathy':['r2_score','mean_squared_error','mean_absolute_error'],
                    'GERO_Ivo':['r2_score','mean_squared_error','mean_absolute_error']}

y_labels = {'MCI_classifier':['target'],
            'tell_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GERO_Ivo':['MMSE_Total_Score'],
            'GeroApathy':['DASS_21_Depression','DASS_21_Anxiety','DASS_21_Stress','AES_Total_Score','MiniSea_MiniSea_Total_FauxPas','Depression_Total_Score','MiniSea_emf_total','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total'],
            }

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

best_models = pd.read_csv(Path(results_dir,'metrics_feature_selection_dev.csv'))

dimensions = list()

for task,y_label,bayes,feature_selection in itertools.product(tasks[project_name],y_labels[project_name],bayes_list,feature_selection_list):

    for ndim in range(1,len(single_dimensions[project_name])+1):
        for dimension in itertools.combinations(single_dimensions[project_name],ndim):
            dimensions.append('__'.join(dimension))

    if len(dimensions) == 0:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

    for dim in dimensions:
        model_name = best_models[(best_models['task'] == task) & (best_models['dimension'] == dim)]['model'].values[0]
        print(f'{task}___{dim}___{model_name}')
        
        try:
            Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '','plots').mkdir(exist_ok=True)

            IDs = pickle.load(open(Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '','IDs_val.pkl'),'rb'))
            y_pred = pickle.load(open(Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '',f'y_pred_best_{model_name}.pkl'),'rb'))
            y_true = pickle.load(open(Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '','y_true_dev.pkl'),'rb'))
            
            data = pd.DataFrame({'ID':IDs.flatten(),'y_pred':y_pred.flatten(),'y_true':y_true.flatten()})
            data = data.drop_duplicates('ID')

            plt.figure()
            sns.scatterplot(x='y_true',y='y_pred',data=data)
            plt.xlabel('True vaue')
            plt.ylabel('Predicted value')
            plt.title(f'{model_name} - {y_label}')
            plt.savefig(Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '','plots',f'{model_name}.png'))
        except:
            pass