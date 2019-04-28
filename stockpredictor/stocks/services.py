import pandas as pd
import numpy as np
from iexfinance.stocks import Stock, get_historical_data

def get_stock(symbol):
	"""
		This function gets stock information of required symbol. Default is SP500.
		Info is obtained from IEX Finance
		- Symbol -> Stock Symbol (e.g SPX)
	"""
	stock = Stock(symbol)
	stock_data = get_historical_data(symbol, output_format='pandas')#gets full historical data on stock
	
	stock_data = stock_data.reset_index()#resets index so index is standard integers
	dates = stock_data['date']#gets dates
	stock_data = stock_data.drop('date', axis=1)
	company = stock.get_company_name()
	
	return stock_data, dates, company
	
	
		
    