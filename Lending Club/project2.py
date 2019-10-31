# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:27:27 2019

@author: Josh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:06:55 2019

@author: Josh
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

lending_data = pd.read_csv("lending_data_final.csv")

lending_data = lending_data.drop("policy_code", axis = 1)


X = lending_data.drop("loan_status_final", axis = 1)
y = lending_data["loan_status_final"]

#skf = StratifiedKFold(n_splits = 5)


"""for trn_idx, tst_idx in skf.split(X,y):
    X_train, X_test = X.iloc[trn_idx], X.iloc[tst_idx]
    y_train, y_test = y.iloc[trn_idx], y.iloc[tst_idx]"""
    
    
def get_sampled(X, y):
    
    skf = StratifiedKFold(n_splits = 5, random_state = 420)
    
    for trn_idx, tst_idx in skf.split(X,y):
        Xtrain, Xtest = X.iloc[trn_idx], X.iloc[tst_idx]#gets sampled, we want to keep 
        ytrain, ytest = y.iloc[trn_idx], y.iloc[tst_idx]#sample amount of data.
        
    return Xtrain, Xtest, ytrain, ytest

X_train, X_test, y_train, y_test = get_sampled(X, y)

        
    
    

X_train_numeric = X_train._get_numeric_data()
X_test_numeric = X_test._get_numeric_data()


X_train_character = X_train.select_dtypes("object")
X_test_character = X_test.select_dtypes("object")

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_numeric)
X_test_num_scaled = scaler.transform(X_test_numeric)

label_binarizer = LabelBinarizer()
y_train_labeled = label_binarizer.fit_transform(y_train)
y_test_labeled = label_binarizer.transform(y_test)


for column in X_train_character.columns:
    
    column_factor = pd.factorize(X_train_character[column])[0]
    
    X_train_character[column] = column_factor
    
for column in X_test_character.columns:
    
    column_factor = pd.factorize(X_test_character[column])[0]
    
    X_test_character[column] = column_factor

X_train_full = np.column_stack((X_train_num_scaled, X_train_character.values))
X_test_full = np.column_stack((X_test_num_scaled, X_test_character.values))


ridge_clf = RidgeClassifier(alpha = 0, random_state = 420)

ridge_clf.fit(X_train_full, y_train_labeled.ravel())

cv_ridge_score = cross_val_score(ridge_clf, X_train_full, y_train_labeled,
                                 scoring = "roc_auc",
                                 cv = 10,
                                 n_jobs = -1,
                                 verbose = -1)
ridge_tuning_params = {
        "alpha" : 10**np.linspace(5, -10, num = 100)
        }
ridge_clf_gs = RidgeClassifier(random_state = 420)
ridge_gridsearch = GridSearchCV(ridge_clf_gs, ridge_tuning_params,
                                scoring = "roc_auc", cv = 10,
                                n_jobs = -1, verbose = -1)
ridge_gridsearch.fit(X_train_full, y_train_labeled.ravel())
ridge_gs_results = ridge_gridsearch.cv_results_

for roc_score, alpha in zip(ridge_gs_results["mean_test_score"], ridge_gs_results["params"]):
    print("ROC Score: {0}, Alpha: {1}".format(roc_score, alpha))

best_ridge = ridge_gridsearch.best_estimator_
ridge_test_pred = best_ridge.predict(X_test_full)
ridge_test_accuracy = accuracy_score(y_test_labeled, ridge_test_pred)

labels = ["Default", "Not Default"]
conf_matrx = confusion_matrix(label_binarizer.inverse_transform(y_test_labeled),
                              label_binarizer.inverse_transform(ridge_test_pred),
                              labels)
conf_matrx_df = pd.DataFrame(conf_matrx, index = ["Default", "Not Default"],
                      columns = ["Default", "Not Default"])
plt.figure(figsize = (12, 8))
sns.heatmap(conf_matrx_df, annot = True, fmt = 'd')
plt.title("Non-Normalized Confusion Matrix")
plt.show()




