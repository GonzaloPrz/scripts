import pandas as pd
import pickle
from pathlib import Path
from expected_cost.utils import *
import itertools
from joblib import Parallel, delayed
import sys,tqdm
from pingouin import compute_bootci
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import logging, sys

sys.path.append(str(Path(Path.home(),'scripts_generales'))) if 'Users/gp' in str(Path.home()) else sys.path.append(str(Path(Path.home(),'gonza','scripts_generales')))

from utils import *

def compute_metrics(model_index, j, r, outputs, y_dev, metrics_names, n_boot, problem_type, project_name):
    # Calculate the metrics using the bootstrap method
    results = get_metrics_bootstrap(outputs[model_index, j, r], y_dev[j, r], metrics_names[problem_type[project_name]], n_boot=n_boot, problem_type=problem_type[project_name])
    
    metrics_result = {}
    for metric in metrics_names[problem_type[project_name]]:
        metrics_result[metric] = results[1][metric]
    return model_index, j, r, metrics_result

def get_metrics_bootstrap(samples, targets, metrics_names, random_state=42, n_boot=2000, decimals=5,cmatrix=None, priors=None, problem_type='clf'):
    assert samples.shape[0] == targets.shape[0]
    
    metrics_ci = dict((metric,(np.empty(0),np.empty((0,2)))) for metric in metrics_names)
    all_metrics = dict((metric,np.empty(n_boot)) for metric in metrics_names)
    
    for metric in metrics_names:
        def get_metric(indices):
            if problem_type == 'clf':
                metric_value, y_pred = get_metrics_clf(samples[indices], targets[indices], [metric], cmatrix)
            else:
                metric_value = get_metrics_reg(samples[indices], targets[indices], [metric])
            return metric_value[metric]
                    
        results = compute_bootci(x=np.arange(targets.shape[0]),func=get_metric,n_boot=n_boot,method='cper',seed=random_state,return_dist=True,decimals=decimals)
        metrics_ci[metric] = (np.round(np.nanmean(results[1]),2),results[0])
        all_metrics[metric] = results[1]

    return metrics_ci, all_metrics
##---------------------------------PARAMETERS---------------------------------##
project_name = 'Proyecto_Ivo'
hyp_opt = True
filter_outliers = False
shuffle_labels = False
feature_selection = True
n_folds = 3
n_models_ = 0.2

n_boot = 200
scaler_name = 'StandardScaler'
id_col = 'id'

# Check if required arguments are provided
if len(sys.argv) > 1:
    #print("Usage: python bootstrap_models_bca.py <project_name> [hyp_opt] [filter_outliers] [shuffle_labels] [feature_selection] [k]")
    project_name = sys.argv[1]
if len(sys.argv) > 2:
    hyp_opt = bool(int(sys.argv[2]))
if len(sys.argv) > 3:
    filter_outliers = bool(int(sys.argv[3]))
if len(sys.argv) > 4:
    shuffle_labels = bool(int(sys.argv[4]))
if len(sys.argv) > 5:
    feature_selection = bool(int(sys.argv[5]))
if len(sys.argv) > 6:
    n_folds = int(sys.argv[6])

parallel = True
 
cmatrix = None

models = {'MCI_classifier':['lr','svc','knnc'],
          'tell_classifier':['lr','svc','knnc'],
          'Proyecto_Ivo':['lr','svc','knnc','xgb'],
          'GeroApathy':['lr','svc','knnc',],
          'GeroAopathy_reg':['lasso','ridge','elastic','svr'],
          'GERO_Ivo':['lasso','ridge','elastic','svr'],
          'MPLS':['lasso','ridge','elastic','svr'],
          'AKU':['lasso','ridge','elastic','svr','xgb']
        }

tasks = {'tell_classifier':['MOTOR-LIBRE'],
         'MCI_classifier':['fas','animales','fas__animales','grandmean' ],
         'Proyecto_Ivo':['cog',
                         'Animales__P',
                         'brain',
                         'Animales','P'
                         ],
         'GeroApathy':['agradable'],
         'GeroApathy_reg':['agradable'],
         'GERO_Ivo':['animales','fas','grandmean','fas__animales'],
         'MPLS':['Estado General','Estado General 2',
                 #'Consulta sobre soledad 1','Consulta sobre soledad 2',
                #'Recuerdo feliz','Animales','Palabras con F'
                ],
         'AKU':['picture_description','pleasant_memory',
                'routine','video_retelling'
                ]}

single_dimensions = {'tell_classifier':['voice-quality','talking-intervals','pitch'],
                     'MCI_classifier':['talking-intervals','psycholinguistic'],
                     'Proyecto_Ivo':[],
                     'GeroApathy':[],
                     'GeroApathy_reg':[],
                     'GERO_Ivo':[],
                     'MPLS':[],
                     'AKU':[]}

problem_type = {'tell_classifier':'clf',
                'MCI_classifier':'clf',
                'Proyecto_Ivo':'clf',
                'GeroApathy':'clf',
                'GeroApathy_reg':'reg',
                'GERO_Ivo':'reg',
                'MPLS':'reg',
                'AKU':'reg'}	

metrics_names = {'clf':['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy'],
                 'reg':['r2_score','mean_squared_error','mean_absolute_error']}

y_labels = {'MCI_classifier':['target'],
            'tell_classifier':['target'],
            'Proyecto_Ivo':['target'],
            'GeroApathy':['DASS_21_Depression_V_label','Depression_Total_Score_label','AES_Total_Score_label',
                         'MiniSea_MiniSea_Total_EkmanFaces_label','MiniSea_minisea_total_label'
                         ],
            'GeroApath_reg':['DASS_21_Depression_V','Depression_Total_Score','AES_Total_Score',
                         'MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total'
                         ],
            'GERO_Ivo':[#'GM_norm','WM_norm','norm_vol_bilateral_HIP','norm_vol_mask_AD',
                        'MMSE_Total_Score','ACEIII_Total_Score','IFS_Total_Score','MoCA_Total_Boni_3'
                        ],
            'MPLS':['Minimental'],
            'AKU':  ['sdi0001_age',
                    'cerad_learn_total_corr',
                    'cerad_dr_correct',
                    'braveman_dr_total',
                    'stick_dr_total',
                    'bird_total',
                    'fab_total',
                    'setshift_total',
                    'an_correct',
                    'mint_total'
                    ]}

scoring_metrics = {'MCI_classifier':['norm_cross_entropy'],
           'tell_classifier':['norm_cross_entropy'],
           'Proyecto_Ivo':['roc_auc'],
           'GeroApathy':['norm_cross_entropy','roc_auc'],
           'GeroApathy_reg':['r2_score','mean_absolute_error'],
           'GERO_Ivo':['r2_score','mean_absolute_error'],
           'MPLS':['r2_score'],
           'AKU':['r2_score']}
##---------------------------------PARAMETERS---------------------------------##

if n_folds == 0:
    kfold_folder = 'l2ocv'
else:
    kfold_folder = f'{n_folds}_folds'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

log_file = Path(results_dir,Path(__file__).stem + '.log')

logging.basicConfig(
    level=logging.DEBUG,  # Log all messages (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Log to a file
        logging.StreamHandler(sys.stdout)  # Keep output in the terminal as well
    ]
)

# Redirect stdout and stderr to the logger
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():  # Avoid logging blank lines
            self.level(message)

    def flush(self):  # Required for file-like behavior
        pass

sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)

for task,model,y_label,scoring in itertools.product(tasks[project_name],models[project_name],y_labels[project_name],scoring_metrics[project_name]):    
    
    dimensions = list()

    for ndim in range(1,len(single_dimensions[project_name])+1):
        for dimension in itertools.combinations(single_dimensions[project_name],ndim):
            dimensions.append('__'.join(dimension))

    if len(dimensions) == 0:
        dimensions = [folder.name for folder in Path(results_dir,task).iterdir() if folder.is_dir()]

    for dimension in dimensions:
        print(task,model,dimension,y_label)
        path = Path(results_dir,task,dimension,scaler_name,kfold_folder,y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection' if feature_selection else '','filter_outliers' if filter_outliers and problem_type[project_name] == 'reg' else '','shuffle' if shuffle_labels else '')
        
        if not path.exists():  
            continue

        random_seeds = [folder.name for folder in path.iterdir() if folder.is_dir() and 'random_seed' in folder.name]
        if len(random_seeds) == 0:
            random_seeds = ['']
        
        for random_seed in random_seeds:
            '''
            if n_models_ == np.inf:

                if Path(path,random_seed,f'all_models_{model}_dev_bca.csv').exists():
                    continue
            elif Path(path,random_seed,f'best_models_{model}_dev_bca_{scoring}.csv').exists():
                    continue 
                
            if not Path(path,random_seed,f'all_models_{model}.csv').exists():
                continue

            if not Path(path,random_seed,f'all_models_{model}.csv').exists():
                continue
            '''
            all_models = pd.read_csv(Path(path,random_seed,f'all_models_{model}.csv'))
            outputs = pickle.load(open(Path(path,random_seed,f'outputs_{model}.pkl'),'rb'))
            y_dev = pickle.load(open(Path(path,random_seed,'y_true_dev.pkl'),'rb'))
            
            if not shuffle_labels and outputs.shape[1] != 1:
                outputs = np.expand_dims(outputs,axis=1)
                y_dev = np.expand_dims(y_dev,axis=0)

            scorings = np.empty(outputs.shape[0])
            
            if n_models_ == np.inf:
                n_models = outputs.shape[0]
                all_models_bool = True
            else:
                all_models_bool = False
                if n_models_ < 1:
                    n_models = int(outputs.shape[0]*n_models_)

                for i in range(outputs.shape[0]):
                    scorings_i = np.empty((outputs.shape[1],outputs.shape[2]))
                    for j,r in itertools.product(range(outputs.shape[1]),range(outputs.shape[2])):
                        if problem_type[project_name] == 'clf':
                            metrics, y_pred = get_metrics_clf(outputs[i,j,r], y_dev[j,r], [scoring], cmatrix)
                            scorings_i[j,r] = metrics[scoring]
                        else:
                            metrics = get_metrics_reg(outputs[i,j,r], y_dev[j,r],[scoring])
                            scorings_i[j,r] = metrics[scoring]
                    scorings[i] = np.nanmean(scorings_i.flatten())
                
                scorings = scorings if any(x in scoring for x in ['norm','error']) else -scorings

                best_models = np.argsort(scorings)[:n_models]
            
                all_models = all_models.iloc[best_models].reset_index(drop=True)
                all_models['idx'] = best_models
                outputs = outputs[best_models]
            
            outputs_bootstrap = np.empty((n_boot,) + outputs.shape)
            y_dev_bootstrap = np.empty((n_boot,) + y_dev.shape)
            y_pred_bootstrap = np.empty((n_boot,)+outputs.shape) if problem_type[project_name] == 'reg' else np.empty((n_boot,)+outputs.shape[:-1])
            
            metrics = dict((metric,np.empty((len(all_models),outputs.shape[1],outputs.shape[2],n_boot))) for metric in metrics_names[problem_type[project_name]])
            try:
                all_results = Parallel(n_jobs=-1)(delayed(compute_metrics)(model_index, j, r, outputs, y_dev, metrics_names, n_boot, problem_type, project_name) for model_index,j,r in itertools.product(range(outputs.shape[0]),range(outputs.shape[1]),range(outputs.shape[2])))

                # Update the metrics array with the computed results
                for model_index,j,r, metrics_result in tqdm.tqdm(all_results):
                    for metric in metrics_names[problem_type[project_name]]:
                        metrics[metric][model_index,j,r,:] = metrics_result[metric]

                # Update the summary statistics in all_models
                for model_index in tqdm.tqdm(range(outputs.shape[0])):
                    for metric in metrics_names[problem_type[project_name]]:
                        all_models.loc[model_index, f'{metric}_mean'] = np.nanmean(metrics[metric][model_index].flatten()).round(5)
                        all_models.loc[model_index, f'{metric}_inf'] = np.nanpercentile(metrics[metric][model_index].flatten(), 2.5).round(5)
                        all_models.loc[model_index, f'{metric}_sup'] = np.nanpercentile(metrics[metric][model_index].flatten(), 97.5).round(5)
                all_models.to_csv(Path(path,random_seed,f'best_models_{model}_dev_bca_{scoring}.csv')) if all_models_bool == False else all_models.to_csv(Path(path,random_seed,f'all_models_{model}_dev_bca.csv')) 

                #pickle.dump(outputs_bootstrap,open(Path(path,random_seed,f'outputs_bootstrap_{model}.pkl'),'wb'))
                #pickle.dump(y_dev_bootstrap,open(Path(path,random_seed,f'y_dev_bootstrap_{model}.pkl'),'wb'))
                #pickle.dump(y_pred_bootstrap,open(Path(path,random_seed,f'y_pred_bootstrap_{model}.pkl'),'wb'))
                #pickle.dump(metrics,open(Path(path,random_seed,f'metrics_bootstrap_{model}_bca_{scoring}.pkl'),'wb'))
            except Exception as e:
                print(e)
                continue    