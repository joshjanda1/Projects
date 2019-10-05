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
        _, X_sample = X.iloc[trn_idx], X.iloc[tst_idx]#gets sampled, we want to keep 
        _, y_sample = y.iloc[trn_idx], y.iloc[tst_idx]#sample amount of data.
        
    skf2 = StratifiedKFold(n_splits = 5, random_state = 420)
    
    for trn_idx, tst_idx in skf.split(X_sample, y_sample):
        X_trn, X_tst = X_sample.iloc[trn_idx], X_sample.iloc[tst_idx]
        y_trn, y_tst = y_sample.iloc[trn_idx], y_sample.iloc[tst_idx]
        
    return X_trn, X_tst, y_trn, y_tst

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

#knn = KNeighborsClassifier(n_neighbors = 1)

#knn.fit(X_train_full, y_train_labeled.ravel())

dtc = DecisionTreeClassifier(random_state = 420)

rfc = RandomForestClassifier(random_state = 420,
                             verbose = 100)

rfc.fit(X_train_full, y_train_labeled.ravel())

rfc_predictions = rfc.predict(X_train_full)
rfc_predictions_test = rfc.predict(X_test_full)

rfc_score = cross_val_score(rfc, X_train_full, y_train_labeled,
                            scoring = "accuracy", cv = 10)
rfc_test_score = accuracy_score(y_test_labeled, rfc_predictions_test)

gbc = GradientBoostingClassifier(random_state = 420, verbose = 100)

gbc.fit(X_train_full, y_train_labeled.ravel())

gbc_score = cross_val_score(gbc, X_train_full, y_train_labeled.ravel(),
                            scoring = "accuracy", cv = 10, n_jobs = -1,
                            verbose = 100)
gbc_predictions = gbc.predict(X_test_full)

conf_matrx = confusion_matrix(y_test_labeled, gbc_predictions)
conf_matrx_df = pd.DataFrame(conf_matrx, index = [i for i in range(2)],
                      columns = [i for i in range(2)])
plt.figure(figsize = (12, 8))
sns.heatmap(conf_matrx_df, annot = True)
plt.title("Non-Normalized Confusion Matrix")
plt.show()

lgr = LogisticRegression(random_state = 420, verbose = -1)
lgr.fit(X_train_full, y_train_labeled.ravel())
lgr_preds = lgr.predict(X_train_full)
xlgr_acc = accuracy_score(y_train_labeled, lgr_preds)

lgr_cv_score = cross_val_score(lgr, X_train_full, y_train_labeled,
                               scoring = "accuracy", cv = 10, n_jobs = -1, verbose = -1)
