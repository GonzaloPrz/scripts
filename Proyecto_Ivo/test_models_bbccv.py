import pandas as pd
import pickle, itertools, sys

import numpy as np
from  pathlib import Path

from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from xgboost import XGBClassifier as XGB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(Path(__file__).parent.parent,'scripts_generales')))

from utils import *

tasks = ['Animales','P','Animales_P']

n_seeds_train = 10
n_seeds_test = 1
exp_ft = False

kfold_folder = '10_folds'

base_dir = Path(__file__).parent

scaler_name = 'StandardScaler'

y_label = 'Grupo'

hyp_opt_list = [True]
bootstrap_list = [True]

models_dict = {'lr':LR,'svc':SVC,'xgb':XGB,'knn':KNN,'lda':LDA}

scaler = StandardScaler()

metrics_names = ['roc_auc','accuracy','recall','f1','norm_expected_cost','norm_cross_entropy']
for task in tasks:
    for dimension_path in Path(base_dir,task).iterdir():
        if not dimension_path.is_dir():
            continue
        path_to_data = Path(dimension_path,scaler_name,'exp_ft' if exp_ft else 'all_features',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'no_hyp_opt','bootstrap')

        for (hyp_opt,bootstrap) in itertools.product(hyp_opt_list,bootstrap_list):
            path_to_data = Path(str(path_to_data).replace('no_hyp_opt','hyp_opt')) if hyp_opt else path_to_data
            path_to_data = Path(str(path_to_data).replace('bootstrap','')) if not bootstrap else path_to_data
            
            X_dev = pickle.load(open(Path(path_to_data,'X_dev.pkl'),'rb'))
            X_holdout = pickle.load(open(Path(path_to_data,'X_holdout.pkl'),'rb'))
            y_dev = pickle.load(open(Path(path_to_data,'y_dev.pkl'),'rb'))
            y_holdout = pickle.load(open(Path(path_to_data,'y_holdout.pkl'),'rb'))

            models = [file.stem.split('_')[-1] for file in path_to_data.iterdir() if file.is_file() and 'all_outputs_val' in file.stem] 

            for model in models:
                best_models = pd.read_csv(Path(path_to_data,f'best_models_{model}.csv'))
                params = [col for col in best_models.columns if 'inf' not in col and 'mean' not in col and 'sup' not in col and col != 'selected']
                
                all_y_scores = np.empty((X_holdout.shape[0],2,best_models.shape[0],n_seeds_test))
                all_y_preds = np.empty((X_holdout.shape[0],best_models.shape[0],n_seeds_test))
                metrics = dict((metric,np.empty((best_models.shape[0],n_seeds_test))) for metric in metrics_names)
                all_metrics = pd.DataFrame(columns = params + metrics_names)
                for m, best_model in best_models.iterrows():
                    model = Model(models_dict[model](**best_model[params].dropna().to_dict()),scaler)
                    for n_seed_test in range(n_seeds_test):                        
                        met, y_pred,y_scores = test_model(model,pd.DataFrame(X_dev[:,:,n_seed_test]),y_dev[:,n_seed_test],pd.DataFrame(X_holdout[:,:,n_seed_test]),y_holdout[:,n_seed_test],metrics_names)

                        all_y_preds[:,m,n_seed_test] = y_pred
                        all_y_scores[:,:,m,n_seed_test] = y_scores
                        for metric in metrics_names:
                            metrics[metric][m,n_seed_test] = met[metric]
                    

                    metrics_holdout, _, _ = test_model(model,
                                                       np.concatenate([X_holdout[:,:,r_test] for r_test in range(n_seeds_test)]),
                                                       np.concatenate([y_holdout[:,:,r_test] for r_test in range(n_seeds_test)]),
                                                       metrics_names)
                    all_metrics.loc[m] = best_model[params].to_dict() | metrics_holdout
                all_metrics.to_csv(Path(path_to_data,f'all_metrics_{model}.csv'),index = False)
                pickle.dump(all_y_preds,open(Path(path_to_data,f'all_y_preds_{model}.pkl'),'wb'))
                pickle.dump(all_y_scores,open(Path(path_to_data,f'all_y_scores_{model}.pkl'),'wb'))
                pickle.dump(metrics,open(Path(path_to_data,f'metrics_{model}.pkl'),'wb'))