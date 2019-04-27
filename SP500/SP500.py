# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:01:40 2019

@author: Josh
"""
#IMPORT PACKAGES
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
#END PACKAGE IMPORT
class DataFrameSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


def get_data(symbol):
    #get tech indicators for stock
    ti = TechIndicators(key= '5FWAPV1GCOE2WQLV', output_format = 'pandas')
    
    sma_, _ = ti.get_sma(symbol=symbol, interval = 'daily')
    macd_, _ = ti.get_macd(symbol=symbol, interval = 'daily')
    rsi_, _ = ti.get_rsi(symbol=symbol, interval = 'daily')
    adx_, _ = ti.get_adx(symbol=symbol, interval = 'daily')

    ts = TimeSeries(key = '5FWAPV1GCOE2WQLV', output_format = 'pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize = 'full')
    
    final_data = pd.concat([data, sma_, macd_, rsi_, adx_], axis=1, sort=True)
    return final_data

def clean_data():
    data = get_data('SPX')
    data = data.reset_index()
    datac = data.copy()
    #we're focused on closing data, so let's drop open, high, low
    datac = datac.drop(['1. open', '2. high', '3. low', '5. volume'], axis=1)
    #now remove null values for first month...
    datac = datac[39:]
    time = datac['index'] #allows us to know time of data while reseting index to integers for easier use
    datac = datac.drop('index', axis=1)
    datac = datac.reset_index(drop = True) #resets index one more time since we dropped initial values in data since null
    closing_data = datac['4. close']
    datac = datac.drop('4. close', axis=1)
    
    return datac, closing_data, time
X, y, dates = clean_data()
#uncomment to get data description, checks to see if any null values mostly
#print(X.describe())
#print(y.describe())
#scatter_matrix(X, figsize = (12, 10))
#looking at scatter matrix, data looks relatively normal
def split_data(n_splits, X, y):
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    for train_index, test_index in kf.split(X):
        Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
    return Xtrain, Xtest, ytrain, ytest

Xtrain, Xtest, ytrain, ytest = split_data(5, X, y) #5 folds
#since we are working with technical indicators, we have no categorical data. We will first build model without time feature
#but we may include that later since this is time series data
num_attr = list(Xtrain)

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attr)),
                         ('imputer', SimpleImputer(strategy = 'mean')),
                         ])
#allows us to fit data unscaled and scale and see which data gives better results
num_pipeline_scaled = Pipeline([('selector', DataFrameSelector(num_attr)),
                                ('imputer', SimpleImputer(strategy = 'mean')),
                                ('std_scaler', StandardScaler()),
                                ])

Xtrain_prep = num_pipeline.fit_transform(Xtrain)
Xtest_prep = num_pipeline.transform(Xtest)
Xtrain_prep_scaled = num_pipeline_scaled.fit_transform(Xtrain)
Xtest_prep_scaled = num_pipeline_scaled.transform(Xtest)

#MODEL SECTION
def displayscores(scores):
    print('Scores from CV: {0}'.format(scores))
    print('Average Score: {0}'.format(np.mean(scores)))
    print('Std. Deviation: {0}'.format(np.std(scores)))
    
dtr = DecisionTreeRegressor()
dtr.fit(Xtrain_prep, ytrain)
dtr_scores = cross_val_score(dtr, Xtrain_prep, ytrain, scoring = 'neg_mean_squared_error', cv=10)#10 fold cross val scores
dtr_rmse_scores = np.sqrt(-dtr_scores)
displayscores(dtr_rmse_scores)

dtr_scaled = DecisionTreeRegressor()
dtr_scaled.fit(Xtrain_prep_scaled, ytrain)
dtr_scaled_scores = cross_val_score(dtr_scaled, Xtrain_prep_scaled, ytrain ,scoring = 'neg_mean_squared_error', cv=10)#10 fold cross val scores
dtr_scaled_rmse_scores = np.sqrt(-dtr_scaled_scores)
displayscores(dtr_scaled_rmse_scores)

svr = SVR(gamma='scale', kernel='linear')
svr.fit(Xtrain_prep, ytrain)
svr_scores = cross_val_score(svr, Xtrain_prep, ytrain, scoring = 'neg_mean_squared_error', cv=10)#10 fold cross val scores
svr_rmse_scores = np.sqrt(-svr_scores)
displayscores(svr_rmse_scores)

svr_scaled = SVR(gamma='scale', kernel='linear')
svr_scaled.fit(Xtrain_prep_scaled, ytrain)
svr_scaled_scores = cross_val_score(svr_scaled, Xtrain_prep_scaled, ytrain, scoring = 'neg_mean_squared_error', cv=10)#10 fold cross val scores
svr_scaled_rmse_scores = np.sqrt(-svr_scaled_scores)
displayscores(svr_scaled_rmse_scores)

rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(Xtrain_prep, ytrain)
rfr_scores = cross_val_score(rfr, Xtrain_prep, ytrain, scoring='neg_mean_squared_error', cv=10)#10 fold cross vals scores
rfr_rmse_scores = np.sqrt(-rfr_scores)
displayscores(rfr_rmse_scores)

rfr_scaled = RandomForestRegressor(n_estimators=200)
rfr_scaled.fit(Xtrain_prep_scaled, ytrain)
rfr_scaled_scores = cross_val_score(rfr_scaled, Xtrain_prep_scaled, ytrain, scoring='neg_mean_squared_error', cv=10)#10 fold cross vals scores
rfr_scaled_rmse_scores = np.sqrt(-rfr_scaled_scores)
displayscores(rfr_scaled_rmse_scores)

"""#build param distribution for randomized cv search for random forest regressor
param_dist = {
        'n_estimators' : randint(low=100, high=1000),
        'max_features' : randint(low=1, high=6),
        }
rfr1 = RandomForestRegressor()
rfr_search = RandomizedSearchCV(rfr1, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)
rfr_search.fit(Xtrain_prep, ytrain)

joblib.dump(rfr_search, 'SP500RFR.pkl')

rfr_search_best = rfr_search.best_estimator_
rfr_search_best_score = rfr_search_best.score(Xtest_prep, ytest)
rfr_search_best_pred = rfr_search_best.predict(Xtrain_prep)
rfr_search_best_mse = mean_squared_error(ytrain, rfr_search_best_pred)
rfr_search_best_rmse = np.sqrt(rfr_search_best_mse)
print('Score for RFR (best estimators): {0}, RMSE: {1}'.format(rfr_search_best_score, rfr_search_best_rmse))"""
#skip step, I have decided that support vector regression results in lowest mean squared error

knr = KNeighborsRegressor(n_neighbors = 20)
knr.fit(Xtrain_prep, ytrain)
knr_scores = cross_val_score(knr, Xtrain_prep, ytrain, scoring='neg_mean_squared_error', cv=10)#10 fold cross val scores
knr_rmse_scores = np.sqrt(-knr_scores)
displayscores(knr_rmse_scores)

knr = KNeighborsRegressor(n_neighbors = 20)
knr.fit(Xtrain_prep, ytrain)
knr_scores = cross_val_score(knr, Xtrain_prep, ytrain, scoring='neg_mean_squared_error', cv=10)#10 fold cross val scores
knr_rmse_scores = np.sqrt(-knr_scores)
displayscores(knr_rmse_scores)


#I can conclude that the best model is the support vector machine regression model that utilized randomized cross validation search.
#With a root mean square error of 8.776

#final mod is unscaled SVR
final_predictions = svr.predict(Xtest_prep)
final_score = mean_squared_error(ytest, final_predictions)
final_rmse = np.sqrt(final_score)
print('RMSE of SVR with test data: {0}'.format(final_rmse))#gives rmse of 8.931, consistent with train data

joblib.dump(svr, 'final_model.pkl')
ytestc = ytest.copy()
ytestc = ytestc.reset_index(drop=True)
plt.plot(final_predictions, c='r', label='Predicted Closing Price')
plt.plot(ytestc, c='g', label='True Closing Price')
plt.xlabel('Index of prediction')
plt.ylabel('Predicted Closing Price / True Closing Price')
plt.title('True Closing Price vs. Predicted Closing Price on Test Set')
plt.legend()
plt.show()















    