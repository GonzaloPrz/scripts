import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from xgboost import XGBClassifier as XGB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import LeavePOut as LPO
from sklearn.model_selection import LeaveOneOut as LOO
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import pickle
from sklearn.utils import resample

from pathlib import Path
import itertools,sys,pickle

sys.path.append(str(Path(Path.home(),'scripts_generales')))

from utils import *

def get_conf_int(x,metrics_names):
    conf_int = dict()
    for metric in metrics_names:
        inf = np.nanpercentile(x[metric],2.5).round(2)
        mean = np.nanmean(x[metric]).round(2)
        sup = np.nanpercentile(x[metric],97.5).round(2)
        conf_int[f'inf_{metric}'] = inf
        conf_int[f'mean_{metric}'] = mean
        conf_int[f'sup_{metric}'] = sup
    return pd.Series(conf_int)

l2ocv = False

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

n_seeds_train = 10
n_seeds_test = 1
y_label = 'target'
hyp_opt = True
n_boot = 100
project_name = 'MCI_classifier'

results_dir = Path(Path.home(),'results',project_name) if 'Users/gp' in str(Path.home()) else Path('D:','CNC_Audio','gonza','results',project_name)

best_classifiers = pd.read_csv(Path(results_dir,f'best_classifiers_{kfold_folder}_hyp_opt.csv' if hyp_opt else f'best_classifiers_{kfold_folder}_no_hyp_opt.csv'))

dimensions = best_classifiers.dimension.unique()
single_dimensions = list()

for dimension in dimensions:
    single_dimensions += dimension.split('__')

single_dimensions = np.unique(single_dimensions)

tasks = best_classifiers.task.unique()
hyp_opt_list = [True]
bootstrap_list = [True]
feature_selection_list = [True]
scaler_name = 'StandardScaler'
metrics_names = ['roc_auc','accuracy','f1','recall','norm_cross_entropy']

scaler = StandardScaler() if scaler_name == 'StandardScaler' else MinMaxScaler()

models_dict = {'lr':LR,'svc':SVC,'xgb':XGB,'knn':KNN}

best_models = dict((f'{dimension}_{task}',dict()) for dimension,task in itertools.product(single_dimensions,tasks))

loocv = LOO()

all_results = pd.DataFrame(columns=['task','combination','roc_auc','accuracy','precision','recall','f1'])

scoring = 'roc_auc'
extremo = 'sup' if 'norm' in scoring else 'inf'
ascending = True if 'norm' in scoring else False

id_col = 'id'

for task in tasks:
    print(task)
    for hyp_opt,feature_selection,bootstrap in itertools.product(hyp_opt_list,feature_selection_list,bootstrap_list):
        path_to_save = Path(results_dir,'late_fusion',task,y_label,'feature_selection','bootstrap')
        if not feature_selection:
            path_to_save = Path(str(path_to_save).replace('feature_selection',''))
        if not bootstrap:
            path_to_save = Path(str(path_to_save).replace('bootstrap',''))

        path_to_save.mkdir(parents=True,exist_ok=True)
        for dimension in single_dimensions:     
            best_model_type = best_classifiers.loc[(best_classifiers.dimension == dimension) & (best_classifiers.task == task)].model_type.values[0]
            
            path = Path(results_dir,task,dimension,'StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'hyp_opt' if hyp_opt else 'no_hyp_opt','feature_selection','bootstrap')
            path = Path(str(path).replace('feature_selection','')) if not feature_selection else path
            path = Path(str(path).replace('bootstrap','')) if not bootstrap else path
            
            random_seeds_test = [random_seed_folder.name for random_seed_folder in path.iterdir() if random_seed_folder.is_dir()]

            for random_seed_test in random_seeds_test:
                X_dev = pickle.load(open((Path(path,random_seed_test,'X_dev.pkl')),'rb'))
                y_dev = pickle.load(open((Path(path,random_seed_test,'y_dev.pkl')),'rb'))
                IDs_dev = pickle.load(open((Path(path,random_seed_test,'IDs_dev.pkl')),'rb'))

                IDs_test = pickle.load(open((Path(path,random_seed_test,'IDs_test.pkl')),'rb'))
                X_test = pickle.load(open((Path(path,random_seed_test,'X_test.pkl')),'rb'))
                y_test = pickle.load(open((Path(path,random_seed_test,'y_test.pkl')),'rb'))
                
                X_dev.drop(id_col,axis=1,inplace=True)
                X_test.drop(id_col,axis=1,inplace=True)

                all_features = X_test.columns 

                X_dev = X_dev[[col for col in X_dev.columns if dimension in col]]
                X_test = X_test[[col for col in X_test.columns if dimension in col]]

                best_model = pd.read_csv(Path(path,random_seed_test,f'all_performances_{best_model_type}.csv').resolve()).sort_values(f'{extremo}_{scoring}_bootstrap',ascending=ascending)

                params = best_model.loc[0,[col for col in best_model.columns if all(x not in col for x in ['inf','sup','mean']) and col not in all_features and col != id_col]].to_dict()

                features = [col for col in all_features if best_model.loc[0,col] ==1]

                feature_index = [i for i,col in enumerate(X_dev.columns) if col in features]

                X_dev = X_dev[features]
                X_test = pd.DataFrame(columns=features,data=X_test[features])

                if best_model_type == 'knn':
                    params['n_neighbors'] = int(params['n_neighbors'])

                params = pd.DataFrame(params,index=[0]).dropna(axis=1).loc[0,:].to_dict()
                model = Model(models_dict[best_model_type](**params),scaler)

                best_models[f'{dimension}_{task}'] = dict((random_seed_test,dict()) for random_seed_test in random_seeds_test)
                
                model.train(X_dev,y_dev)

                #best_models[f'{dimension}_{task}'][random_seed_test]['trained_model'] = model.train(X_train,y_train)
                best_models[f'{dimension}_{task}'][random_seed_test]['X_train'] = X_test
                best_models[f'{dimension}_{task}'][random_seed_test]['y_true'] = y_test
                best_models[f'{dimension}_{task}'][random_seed_test]['y_score'] = model.eval(X_test)[:,1]
        
        for ndim in range(2,len(single_dimensions)+1):
            if ndim == 2:
                late_fusion_models = dict(('_'.join(dimensions),dict()) for dimensions in itertools.combinations(single_dimensions,ndim))
                X_dev_late_fusion = dict(('_'.join(dimensions),dict()) for dimensions in itertools.combinations(single_dimensions,ndim))
                y_dev_late_fusion = dict(('_'.join(dimensions),dict()) for dimensions in itertools.combinations(single_dimensions,ndim))
            else:
                late_fusion_models.update(dict(('_'.join(dimensions),dict()) for dimensions in itertools.combinations(single_dimensions,ndim)))
                X_dev_late_fusion.update(dict(('_'.join(dimensions),dict()) for dimensions in itertools.combinations(single_dimensions,ndim)))
                y_dev_late_fusion.update(dict(('_'.join(dimensions),dict()) for dimensions in itertools.combinations(single_dimensions,ndim)))
            
            for dimensions in itertools.combinations(single_dimensions,ndim):
                all_scores = pd.DataFrame(columns=['y_scores_' + '_'.join(dimensions),'y_true','y_pred' + '_'.join(dimensions)])

                late_fusion_models['_'.join(dimensions)] = dict((f'{random_seed_test}',dict()) for random_seed_test in random_seeds_test)
                X_dev_late_fusion['_'.join(dimensions)] = dict((f'{random_seed_test}',dict()) for random_seed_test in random_seeds_test)
                y_dev_late_fusion['_'.join(dimensions)] = dict((f'{random_seed_test}',dict()) for random_seed_test in random_seeds_test)
                                    
                y_scores_val = np.empty((0,2))
                y_true_val = np.empty((0,1))
                
                for random_seed_test in random_seeds_test:
                    late_fusion_models['_'.join(dimensions)][random_seed_test] = dict()
                    X_dev_late_fusion['_'.join(dimensions)][random_seed_test] = pd.DataFrame(columns=[f'y_score_{dimension}' for dimension in dimensions])
                    for dimension in dimensions:
                        X_dev_late_fusion['_'.join(dimensions)][random_seed_test][f'y_score_{dimension}'] = best_models[f'{dimension}_{task}'][random_seed_test]['y_score']
                    y_dev_late_fusion['_'.join(dimensions)][random_seed_test] = best_models[f'{dimensions[0]}_{task}'][random_seed_test]['y_true']

                    for train_index, test_index in loocv.split(X_dev_late_fusion['_'.join(dimensions)][random_seed_test]):
                        X_train = X_dev_late_fusion['_'.join(dimensions)][random_seed_test].loc[train_index]
                        y_train = y_dev_late_fusion['_'.join(dimensions)][random_seed_test].loc[train_index]
                        X_test = X_dev_late_fusion['_'.join(dimensions)][random_seed_test].loc[test_index]
                        y_test = y_dev_late_fusion['_'.join(dimensions)][random_seed_test].loc[test_index]

                        m = LR(random_state=42).fit(X_train,y_train)
                        if y_scores_val.shape[0] == 0:
                            y_scores_val = m.predict_proba(X_test)
                            y_true_val = y_test
                        else:
                            y_scores_val = np.vstack((y_scores_val,m.predict_proba(X_test)))
                            y_true_val = np.hstack((y_true_val,y_test))

                late_fusion_models['_'.join(dimensions)]['y_scores'] = y_scores_val
                late_fusion_models['_'.join(dimensions)]['y_true'] = y_true_val
                
                for b in range(np.max((1,n_boot))):
                    scores = pd.DataFrame(columns=['y_scores_' + '_'.join(dimensions),'y_true','y_pred' + '_'.join(dimensions)])

                    boot_index = resample(np.arange(y_scores_val.shape[0]),replace=True,n_samples=y_scores_val.shape[0],random_state=b)
                    metrics, y_pred = get_metrics_clf(y_scores_val[boot_index,:],y_true_val[boot_index],metrics_names)
                    for metric in metrics_names:
                            late_fusion_models['_'.join(dimensions)][metric] = metrics[metric]
                    
                    scores['y_scores_' + '_'.join(dimensions)] = y_scores_val[:,1]
                    scores['y_true'] = y_true_val
                    scores['y_pred' + '_'.join(dimensions)] = y_pred

                    all_scores = pd.concat((all_scores,scores),axis=0)

                    df_append = {'task':task,'combination':'_'.join(dimensions),'bootstrap':b}
                    df_append.update(dict((metric,late_fusion_models['_'.join(dimensions)][metric]) for metric in metrics_names))

                    if all_results.empty:
                        all_results = pd.DataFrame(df_append,index=[0])
                    else:
                        all_results = pd.concat((all_results,pd.DataFrame(df_append,index=[0])),axis=0)
                all_scores.to_csv(Path(path_to_save,'_'.join(dimensions) + '_all_scores.csv'),index=False)
                
    with open(Path(path_to_save,'late_fusion_models.pkl'),'wb') as file:
        pickle.dump(late_fusion_models,file)

conf_int_results = all_results.groupby(['task','combination']).apply(lambda x: get_conf_int(x,metrics_names))

all_results.to_csv(Path(results_dir,'late_fusion','all_results.csv'),index=False,encoding='utf-8',)

conf_int_results.to_csv(Path(results_dir,'late_fusion','conf_int_results.csv'),encoding='utf-8',)