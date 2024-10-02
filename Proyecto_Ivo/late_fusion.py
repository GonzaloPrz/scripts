import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from xgboost import XGBClassifier as XGB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import LeavePOut as LPO
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

from pathlib import Path
import itertools,sys,pickle

sys.path.append(str(Path(__file__).parent.parent,'scripts_generales'))

from machine_learning_module import *

l2ocv = False

if l2ocv:
    kfold_folder = 'l2ocv'
else:
    n_folds = 10
    kfold_folder = f'{n_folds}_folds'

n_seeds_train = 10
n_seeds_test = 10
y_label = 'DCL'

best_classifiers = pd.read_csv(Path(Path(__file__).parent,'best_classifiers_10_folds_hyp_opt_bootstrap.csv'))

dimensions = best_classifiers.dimension.unique()
tasks = best_classifiers.task.unique()
hyp_opt_list = [True]
bootstrap_list = [True]
feature_selection_list = [False]
scaler_name = 'StandardScaler'

scaler = StandardScaler() if scaler_name == 'StandardScaler' else MinMaxScaler()

models_dict = {'lr':LR,'svc':SVC,'xgb':XGB,'knn':KNN}

best_models = dict((f'{dimension}_{task}',dict()) for dimension,task in itertools.product(dimensions,tasks))

l2ocv = LPO(2)

for task in tasks:
    for dimension,hyp_opt,bootstrap,feature_selection in itertools.product(dimensions,hyp_opt_list,bootstrap_list,feature_selection_list):
        if '_' in dimension:
            continue
        
        best_model_type = best_classifiers.loc[best_classifiers.dimension == dimension and best_classifiers.task == task,'model_type']
        
        best_model = pd.read_csv(Path(Path(__file__).parent,'hyp_opt' if hyp_opt else 'no_hyp_opt',f'{n_seeds_test}_seeds_test',y_label,'by_roc_auc_inf_val',kfold_folder,f'{task}_{dimension}_conf_int_val_{best_model_type}.csv')).iloc[0]
        params = dict([(key,value) for key,value in best_model.items() if 'inf' not in key and 'sup' not in key and 'mean' not in key])
        model = Model(models_dict[best_model_type](**params),scaler)

        path_to_data = Path(Path(__file__).parent,dimension,task,'StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'no_hyp_opt','feature_selection','bootstrap')
        if hyp_opt:
            path_to_data = Path(str(path_to_data).replace('no_hyp_opt','hyp_opt'))

        if not feature_selection:
            path_to_data = Path(str(path_to_data).replace('feature_selection',''))
        
        if not bootstrap:
            path_to_data = Path(str(path_to_data).replace('bootstrap',''))
        
        random_seeds_test = [random_seed_folder.name for random_seed_folder in Path(path_to_data).iterdir() if random_seed_folder.is_dir()]

        best_models[f'{dimension}_{task}'] = dict((random_seed_test,dict()) for random_seed_test in random_seeds_test)

        for random_seed_test in random_seeds_test:
            with open(Path(path_to_data,random_seed_test,'X_train.pkl'),'rb') as file:
                X_train = pickle.load(file)
            with open(Path(path_to_data,random_seed_test,'y_train.pkl'),'rb') as file:
                y_train = pickle.load(file)
            with open(Path(path_to_data,random_seed_test,'X_test.pkl'),'rb') as file:
                X_test = pickle.load(file)
            with open(Path(path_to_data,random_seed_test,'y_test.pkl'),'rb') as file:
                y_test = pickle.load(file)
            
            best_models[f'{dimension}_{task}'][random_seed_test]['trained_model'] = model.train(X_train,y_train)
            best_models[f'{dimension}_{task}'][random_seed_test]['y_true'] = y_test
            best_models[f'{dimension}_{task}'][random_seed_test]['y_score'] = model.eval(X_test)[:,1]

    late_fusion_models = {}
    X_dev_late_fusion = {}
    y_dev_late_fusion = {}

    for dimension1,dimension2 in itertools.combinations(dimensions,2):
        X_dev_late_fusion[f'{dimension1}_{dimension2}'] = pd.DataFrame(columns=[f'y_score_{dimension1}',f'y_score_{dimension2}'],data=np.concatenate((best_models[f'{dimension1}_{task}'][random_seeds_test[0]].y_score,best_models[f'{dimension2}_{task}'][random_seeds_test[0]].y_score),axis=1))
        y_dev_late_fusion[f'{dimension1}_{dimension2}'] = best_models[f'{dimension1}_{task}'][random_seeds_test[0]].y_true

        y_scores_val = np.zeros((len(y_dev_late_fusion[f'{dimension1}_{dimension2}']),2))
        y_true_val = np.zeros(len(y_dev_late_fusion[f'{dimension1}_{dimension2}']))
        for train_index, test_index in l2ocv.split(X_dev_late_fusion[f'{dimension1}_{dimension2}']):  
            X_train = X_dev_late_fusion[f'{dimension1}_{dimension2}'].iloc[train_index].reset_index(drop=True)
            y_train = y_dev_late_fusion[f'{dimension1}_{dimension2}'][train_index].reset_index(drop=True)
            X_test = X_dev_late_fusion[f'{dimension1}_{dimension2}'].iloc[test_index].reset_index(drop=True)
            y_test = y_dev_late_fusion[f'{dimension1}_{dimension2}'][test_index].reset_index(drop=True)

            LR(random_state=42).fit(X_train,y_train)
            y_scores_val[test_index] = LR.predict_proba(X_test)
            y_true_val[test_index] = y_test

        late_fusion_models[f'{dimension1}_{dimension2}']['model'] = LR(random_state=42).fit(X_dev_late_fusion[f'{dimension1}_{dimension2}'],y_dev_late_fusion[f'{dimension1}_{dimension2}'])
        late_fusion_models[f'{dimension1}_{dimension2}']['y_scores'] = y_scores_val
        late_fusion_models[f'{dimension1}_{dimension2}']['y_true'] = y_true_val
        late_fusion_models[f'{dimension1}_{dimension2}']['roc_auc'] = roc_auc_score(y_true_val,y_scores_val[:,1])
        late_fusion_models[f'{dimension1}_{dimension2}']['accuracy'] = accuracy_score(y_true_val,np.argmax(y_scores_val,axis=1))
        late_fusion_models[f'{dimension1}_{dimension2}']['precision'] = precision_score(y_true_val,np.argmax(y_scores_val,axis=1))
        late_fusion_models[f'{dimension1}_{dimension2}']['recall'] = recall_score(y_true_val,np.argmax(y_scores_val,axis=1))
        late_fusion_models[f'{dimension1}_{dimension2}']['f1'] = f1_score(y_true_val,np.argmax(y_scores_val,axis=1))

    for dimension1,dimension2,dimension3 in itertools.combinations(dimensions,3):
        X_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}'] = pd.DataFrame(columns=[f'y_score_{dimension1}',f'y_score_{dimension2}',f'y_score_{dimension3}'],data=np.concatenate((best_models[f'{dimension1}_{task}'][random_seeds_test[0]].y_score,best_models[f'{dimension2}_{task}'][random_seeds_test[0]].y_score,best_models[f'{dimension3}_{task}'][random_seeds_test[0]].y_score),axis=1))
        y_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}'] = best_models[f'{dimension1}_{task}'][random_seeds_test[0]].y_true                                                                         
        
        y_scores_val = np.zeros((len(y_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}']),2))
        y_true_val = np.zeros(len(y_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}']))

        for train_index, test_index in l2ocv.split(X_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}']):
            X_train = X_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}'].iloc[train_index].reset_index(drop=True)
            y_train = y_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}'][train_index].reset_index(drop=True)
            X_test = X_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}'].iloc[test_index].reset_index(drop=True)
            y_test = y_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}'][test_index].reset_index(drop=True)

            LR(random_state=42).fit(X_train,y_train)
            y_scores_val[test_index] = LR.predict_proba(X_test)
            y_true_val[test_index] = y_test

        late_fusion_models[f'{dimension1}_{dimension2}_{dimension3}']['model'] = LR(random_state=42).fit(X_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}'],y_dev_late_fusion[f'{dimension1}_{dimension2}_{dimension3}'])
        late_fusion_models[f'{dimension1}_{dimension2}_{dimension3}']['y_scores'] = y_scores_val
        late_fusion_models[f'{dimension1}_{dimension2}_{dimension3}']['y_true'] = y_true_val
        late_fusion_models[f'{dimension1}_{dimension2}_{dimension3}']['roc_auc'] = roc_auc_score(y_true_val,y_scores_val[:,1])
        late_fusion_models[f'{dimension1}_{dimension2}_{dimension3}']['accuracy'] = accuracy_score(y_true_val,np.argmax(y_scores_val,axis=1))
        late_fusion_models[f'{dimension1}_{dimension2}_{dimension3}']['precision'] = precision_score(y_true_val,np.argmax(y_scores_val,axis=1))
        late_fusion_models[f'{dimension1}_{dimension2}_{dimension3}']['recall'] = recall_score(y_true_val,np.argmax(y_scores_val,axis=1))
        late_fusion_models[f'{dimension1}_{dimension2}_{dimension3}']['f1'] = f1_score(y_true_val,np.argmax(y_scores_val,axis=1))

    with open(Path(Path(__file__).parent,dimension,task,'StandardScaler',kfold_folder,f'{n_seeds_train}_seeds_train',f'{n_seeds_test}_seeds_test',y_label,'no_hyp_opt','feature_selection','bootstrap','late_fusion_models.pkl'),'wb') as file:
        pickle.dump(late_fusion_models,file)


print(late_fusion_models)