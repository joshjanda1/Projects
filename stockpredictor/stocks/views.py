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
	
	get_stock_info_api, time = services.get_stock(symbol)
	datelist = pd.DatetimeIndex(time)
	
	volatility_data = get_stock_info_api['Close'].iloc[-10: -1]#10 period volatility calculation
	volatility = np.std(volatility_data)
	
	if get_stock_info_api is None:
		raise Http404("Stock {0} does not exist, try again.".format(symbol))
		
	graph = figure(plot_width=800, plot_height=250, x_axis_type = 'datetime')
	graph.line(datelist, get_stock_info_api['Close'], color='green', alpha=.5)
	script, div = components(graph)
	
	stock_info = {
		'symbol': symbol,
		'today_close': get_stock_info_api['Close'].iloc[-1],#gets todays closing price
		'today_open': get_stock_info_api['Open'].iloc[-1],#gets todays open price
		'today_high': get_stock_info_api['High'].iloc[-1],#gets todays high
		'today_low': get_stock_info_api['Low'].iloc[-1],#gets todays low
		'today_volume': get_stock_info_api['Volume'].iloc[-1],
		'current_volatility': volatility,
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
