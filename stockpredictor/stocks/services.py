import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators


def get_stock(symbol='SPX'):
	"""
		This function gets stock information of required symbol. Default is SP500.
		Info is obtained from Alpha Vantage API
		- Symbol -> Stock Symbol (e.g SPX)
	"""
	
	#get tech indicators for stock
	ti = TechIndicators(key= '5FWAPV1GCOE2WQLV', output_format = 'pandas')
		
	sma, _ = ti.get_sma(symbol=symbol, interval = 'daily')
	macd, _ = ti.get_macd(symbol=symbol, interval = 'daily')
	rsi, _ = ti.get_rsi(symbol=symbol, interval = 'daily')
	adx, _ = ti.get_adx(symbol=symbol, interval = 'daily')

	ts = TimeSeries(key = '5FWAPV1GCOE2WQLV', output_format = 'pandas')
	data, _ = ts.get_daily(symbol=symbol, outputsize = 'full')
		
	intermediate_data = pd.concat([data, sma, macd, rsi, adx], axis=1, sort=True)
	
	data = intermediate_data.reset_index()#reset index so standard..
	data['Open'], data['High'], data['Low'], data['Volume'], data['Close'] = data['1. open'], data['2. high'], data['3. low'], data['5. volume'], data['4. close']
	data = data.drop(['1. open', '2. high', '3. low', '4. close', '5. volume'], axis=1)#gets conventional names for labels
	#now remove null values for first month...
	data = data[39:]
	time = data['index'] #allows us to know time of data while reseting index to integers for easier use
	data = data.drop('index', axis=1)
	data = data.reset_index(drop = True) #resets index one more time since we dropped initial values in data since null
	final_data = data
	return final_data, time
		
    