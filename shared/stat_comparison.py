import pandas as pd
from pathlib import Path
import numpy as np
from mlxtend.evaluate import cochrans_q
import itertools

l2ocv = False

project_name = 'Proyecto_Ivo'

models = {'MCI_classifier':['lr','svc','knn','xgb'],
          'tell_classifier':['lr','svc','knn','xgb'],
          'Proyecto_Ivo':['lr','svc','knn','xgb'],
          'GeroApathy':['lasso','ridge','knn']
            }

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean'
                           ],
         'Proyecto_Ivo':['Animales','P','Animales_P','cog','brain','AAL','conn'],
         'GeroApathy':['Fugu']}

single_dimensions = {'tell_classifier':['voice-quality',
                                        'talking-intervals','pitch'
                                        ],
                     'MCI_classifier':['talking-intervals','psycholinguistic'],
                     'Proyecto_Ivo':[],
                     'GeroApathy':[]}

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GeroApathy':'reg'}

metrics_names = {'MCI_classifier':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'tell_classifier':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'Proyecto_Ivo':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                    'GeroApathy':['r2','mean_squared_error','mean_absolute_error']}

y_labels = {'MCI_classifier':['target'],
            'tell_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GeroApathy':['DASS_21_Depression','DASS_21_Anxiety','DASS_21_Stress','AES_Total_Score','MiniSea_MiniSea_Total_FauxPas','Depression_Total_Score','MiniSea_emf_total','MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total']}


if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 5
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

best_classifiers = pd.read_csv(Path(results_dir))
for y_label,task in itertools.product(y_labels[project_name],tasks[project_name]):

