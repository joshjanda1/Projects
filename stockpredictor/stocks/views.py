from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, Http404
from django.shortcuts import get_object_or_404
from . import services
from django.db.models import Q
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.io import output_file, show, output_notebook
from bokeh.resources import CDN
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np
import pandas as pd

# Create your views here.

def detail_view(request):
	"""
		Gives details on stock symbol.
		Parameters:
		- Request -> Request
		- Symbol -> Trading Symbol, such as SPX for SP500
	"""
	symbol = request.GET.get('usr_search')
	template = loader.get_template('stocks/detail.html')
	
	get_stock_info_api, dates, company = services.get_stock(symbol)#gets stock info
	
	volatility_data = get_stock_info_api['close'].iloc[-10: -1]#10 period volatility calculation
	volatility = np.std(volatility_data)
	volatility = round(volatility, 4)
	
	one_day_percent_change = (np.log(get_stock_info_api['close'].iloc[-1])-np.log(get_stock_info_api['close'].iloc[-2]))*100
	one_day_percent_change = round(one_day_percent_change, 2)#percentage change calculation
	
	close = get_stock_info_api['close']
	dates_close = pd.concat([close, dates], axis=1, sort=True)#create dataframe of close and dates
	dates_close_cds = ColumnDataSource(dates_close)#get data source for bokeh plot ##allows us to add tooltip
	
	
	if get_stock_info_api is None:
		raise Http404("Stock {0} does not exist, try again.".format(symbol))
	hover = HoverTool(tooltips=[('Date', '@date{%F}'), ('Close', '$@{close}{%0.2f}'),],
		formatters={'date': 'datetime', 'close': 'printf'})
		
	graph = figure(plot_width=800, plot_height=250, x_axis_type = 'datetime', 
		title='Closing Data of {0}'.format(symbol))
	graph.line("date", "close", color='green', alpha=.5, source=dates_close_cds)
	graph.add_tools(hover)
	graph.title.align = 'center'
	script, div = components(graph)
	
	stock_info = {
		'symbol': symbol,
		'today_close': get_stock_info_api['close'].iloc[-1],#gets todays closing price
		'today_open': get_stock_info_api['open'].iloc[-1],#gets todays open price
		'today_high': get_stock_info_api['high'].iloc[-1],#gets todays high
		'today_low': get_stock_info_api['low'].iloc[-1],#gets todays low
		'today_volume': get_stock_info_api['volume'].iloc[-1],
		'current_volatility': volatility,
		'one_day_percent_change': one_day_percent_change,
		'company': company,
		'script': script,
		'div': div,
	}
	
	
	return HttpResponse(template.render(stock_info, request))
	

def index(request):
	
	template = loader.get_template('stocks/index.html')
	context = {}
	return HttpResponse(template.render(context, request))
	
def search(request):
	template = loader.get_template('post_list.html')
	
	query = requests.GET.get('usr_search')
	test = {
		'test':'test'
	}
	
	
	return HttpResponse(template.render(test, request))
