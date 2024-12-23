import pandas as pd
from pathlib import Path
import numpy as np
import itertools,pickle
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

random_seeds_train = [3**x for x in np.arange(1,11)]

project_name = 'GERO_Ivo'
l2ocv = False

cmatrix = None
shuffle_labels = False
bayes_list = [False]
feature_selection_list = [True]

id_col = 'id'
scaler_name = 'StandardScaler'

models = {'MCI_classifier':['lr','svc','knnc','xgb'],
          'tell_classifier':['lr','svc','knnc','xgb'],
          'Proyecto_Ivo':['lr','svc','knnr','xgb'],
          'GeroApathy':['elastic','lasso','ridge','knnr','svr','xgb'],
            'GERO_Ivo':['elastic','lasso','ridge','knnr','svr']
            }

scoring = {'MCI_classifier':'roc_auc',
           'tell_classifier':'roc_auc',
           'Proyecto_Ivo':'roc_auc',
           'GeroApathy':'r2_score',
           'GERO_Ivo':'r2_score'
           }

ascending = {'MCI_classifier':True,
                'tell_classifier':True,
                'Proyecto_Ivo':True,
                'GeroApathy':False,
                'GERO_Ivo':False
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
            'GERO_Ivo':[],
            'GeroApathy':['DASS_21_Depression','DASS_21_Anxiety','DASS_21_Stress','AES_Total_Score','MiniSea_MiniSea_Total_FauxPas','Depression_Total_Score','MiniSea_emf_total','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total'],
            }

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

Path(results_dir,'plots').mkdir(exist_ok=True)

best_models = pd.read_csv(Path(results_dir,f'best_models_{scoring[project_name]}_{n_folds}_folds_StandardScaler_hyp_opt_feature_selection.csv'))

dimensions = list()

pearsons_results = pd.DataFrame(columns=['task','dimension','y_label','model_type','r','p_value'])

if y_labels[project_name] == []:
    y_labels = {project_name:best_models['y_label'].unique()}

for y_label,bayes,feature_selection in itertools.product(y_labels[project_name],bayes_list,feature_selection_list):
    best_models_y_label = best_models[best_models['y_label'] == y_label].sort_values(by=f'{scoring[project_name]}_mean_dev',ascending=ascending[project_name]).reset_index(drop=True)
    task = best_models_y_label['task'].values[0]

    for ndim in range(1,len(single_dimensions[project_name])+1):
        for dimension in itertools.combinations(single_dimensions[project_name],ndim):
            dimensions.append('__'.join(dimension))

    if len(dimensions) == 0:
        #dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]
        dimensions = [best_models_y_label['dimension'].values[0]]
    for dim in dimensions:
        model_name = best_models[(best_models['task'] == task) & (best_models['dimension'] == dim) & (best_models['y_label'] == y_label)]['model_type'].values[0]
        model_index = best_models[(best_models['task'] == task) & (best_models['dimension'] == dim) & (best_models['y_label'] == y_label)]['model_index'].values[0]
        print(f'{y_label}_{task}___{dim}___{model_name}')
        
        #Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '','plots').mkdir(exist_ok=True)
        IDs_ = pickle.load(open(Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '','IDs_dev.pkl'),'rb'))
        y_pred = pickle.load(open(Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '',f'outputs_{model_name}.pkl'),'rb'))[model_index,]
        y_true = pickle.load(open(Path(results_dir,task,dim,scaler_name,kfold_folder,y_label,'hyp_opt','bayes' if bayes else '','feature_selection' if feature_selection else '','y_true_dev.pkl'),'rb'))
        
        if IDs_.ndim == 1:
            
            IDs = np.empty((10,len(IDs_)),dtype=object)

            for i,seed in enumerate(random_seeds_train):
                kf = KFold(n_splits=5,shuffle=True,random_state=seed)
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
        plt.title(f'{model_name} - {y_label}')

        pearsons_results.loc[len(pearsons_results)] = [task, dim, y_label, model_name, r, p]

        # Add stats to plot
        plt.text(data['y_true'].min(), data['y_pred'].max(), f'r = {r:.2f}, p = {p:.2e}', fontsize=12)
        plt.legend()

        # Save the plot
        plt.savefig(Path(results_dir, 'plots', f'{y_label}_{kfold_folder}_{model_name}.png'))
        plt.close()
