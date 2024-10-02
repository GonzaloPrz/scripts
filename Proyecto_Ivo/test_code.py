from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

model = SVC(probability=True,random_state=42,kernel='rbf',gamma=4,C=0.9)

data = pd.read_excel('/Users/gp/Proyecto_Ivo/data/data_total.xlsx',sheet_name='properties')

data['Grupo'] = data['Grupo'].map({'CTR':0,'AD':1})

features = [col for col in data.columns if 'Animales_' in col and 'P_' not in col]

sss = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=0)
train_index, test_index = next(sss.split(data[features],data['Grupo']))

data_train = data.iloc[train_index]

y = data_train['Grupo']
X = data_train[features]

LOOCV = LeaveOneOut()
KFold_CV = KFold(n_splits=len(y)-1)

y_scores = np.zeros((0,1))
y_pred = np.zeros(0)
y_true = np.zeros(0)

for train_index, test_index in LOOCV.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    y_true = np.hstack((y_true,y_test))

    scaler = StandardScaler()
    scaler.fit(X_train[features])
    X_train = pd.DataFrame(columns=features,data=scaler.transform(X_train[features]))
    X_test = pd.DataFrame(columns=features,data=scaler.transform(X_test[features]))

    model.fit(X_train,y_train)
    y_pred = np.hstack((y_pred,model.predict(X_test)))

    y_scores = np.vstack((y_scores,model.predict_proba(X_test)[0][1]))

scores = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'y_scores': y_scores[:,0]})

roc_auc = roc_auc_score(y_true,y_scores[:,0])
accuracy = accuracy_score(y_true,y_pred)
print(roc_auc)
print(accuracy)

#Correlations between features and target
for feature in features:
    sns.boxplot(x='Grupo',y=feature,data=data)
    plt.title(f'{feature} vs Grupo')
    plt.show()

