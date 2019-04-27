# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:01:23 2019

@author: Josh
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from util import get_data
from util import DataFrameSelector, MyLabelBinarizer, MyOneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

def clean_data():
    titanic = get_data()
    titanic['Fare'] = titanic['Fare'].astype(float)
    titanic = titanic.drop(['Name', 'PassengerId'], axis=1)
    #fixes ticket class classification
    return titanic
titanic = clean_data()
titanic['Embarked'] = titanic['Embarked'].fillna('S')#2 null values for embarked, fill with S since overwhelmingly most common
titanic_gender = titanic['Sex']
titanic = titanic.drop('Sex', axis=1)
#build train and test data
Xtrain, Xtest, ytrain, ytest = train_test_split(titanic, titanic_gender, test_size = .25, random_state = 42)
#lets work with train data currently
Xtrain_num = Xtrain.drop(['Embarked', 'Pclass'], axis=1)#drop classes since we will scale numerical data due to age outliers
num_attr = list(Xtrain_num)
cat_attr = ['Embarked']
class_attr = ['Pclass']#separate classes since categorical
gender_attr = ['Sex']

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attr)), 
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler()),
                        ])
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attr)), 
        ('label_binarizer', MyLabelBinarizer()),
                        ])
class_pipeline = Pipeline([
        ('selector', DataFrameSelector(class_attr)),
        ('Encoder', MyOneHotEncoder())
                        ])
full_pipeline = FeatureUnion(transformer_list=([
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
        ('class_pipeline', class_pipeline),
                                                ]))
    
Xtrain_prep = full_pipeline.fit_transform(Xtrain)
Xtest_prep = full_pipeline.transform(Xtest)


#model building


KNC = KNeighborsClassifier(n_neighbors=5)
KNC.fit(Xtrain_prep, ytrain)
scoreknc = KNC.score(Xtest_prep, ytest)
knc_pred = KNC.predict(Xtrain_prep)
print('Accuracy of KNC: {0}'.format(scoreknc))#Best model with ~80% accuracy
knc_f1 = f1_score(ytrain, knc_pred, labels = ['female', 'male'], average=None)
print('F1 Score for KNC: {0}'.format(knc_f1))

sgd_clf = SGDClassifier()
sgd_clf.fit(Xtrain_prep, ytrain)
scoresgd = sgd_clf.score(Xtest_prep, ytest)
sgd_pred = sgd_clf.predict(Xtrain_prep)
print('Accuracy of SGDC: {0}'.format(scoresgd))
sgd_f1 = f1_score(ytrain, sgd_pred, labels = ['female', 'male'], average=None)
print('F1 Score for SGD: {0}'.format(sgd_f1))
"""DTC = DecisionTreeClassifier(random_state=42)
DTC.fit(Xtrain_prep, ytrain)
scoredtc = DTC.score(Xtest_prep, ytest)
print('Accuracy of DTC: {0}'.format(scoredtc))
"""

#search distribution
"""param_distr = {
        'n_estimators' : randint(low=200, high=1000),
        'max_features' : randint(low=1, high=11)
            }
RFC2 = RandomForestClassifier(random_state=42)
RFC_search = RandomizedSearchCV(RFC2, param_distributions = param_distr, scoring='accuracy', n_iter=25, cv=5, random_state=42)
RFC_search.fit(Xtrain_prep, ytrain)

search_res = RFC_search.cv_results_
for score, param in zip(search_res['mean_train_score'], search_res['params']):
    print(score, param)

fin_mod = RFC_search.best_estimator_
score_finmod = fin_mod.score(Xtest_prep, ytest)
print('Accuracy for best RFC Model: {0}'.format(score_finmod))"""

#joblib.dump(RFC_search, 'titanic.pkl')

