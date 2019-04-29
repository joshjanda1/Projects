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
		- Symbol -> Trading Symbol, such as AAPL for Apple Inc.
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
	"""#adding graphs of popular companies on home page. e.g AAPL, MSFT, SPY (SP500 ETF), GOOGL
	aapl, dates_aapl, company_aapl = services.get_stock(symbol='AAPL')
	googl, dates_googl, company_googl = services.get_stock(symbol='GOOGL')
	msft, dates_msft, company_msft = services.get_stock(symbol='MSFT')
	spy, dates_spy, company_spy = services.get_stock(symbol='SPY')
								#gd for graph data													#cds for ColumnDataSource
	aapl_close = aapl['close']; aapl_gd = pd.concat([aapl_close, dates_aapl], axis=1, sort=True); aapl_cds = ColumnDataSource(aapl_gd)
	googl_close = googl['close']; googl_gd = pd.concat([googl_close, dates_googl], axis=1, sort=True); googl_cds = ColumnDataSource(googl_gd)
	msft_close = msft['close']; msft_gd = pd.concat([msft_close, dates_msft], axis=1, sort=True); msft_cds = ColumnDataSource(msft_gd)
	spy_close = spy['close']; spy_gd = pd.concat([spy_close, dates_spy], axis=1, sort=True); spy_cds = ColumnDataSource(spy_gd)
	
	hover = HoverTool(tooltips = [
						('Date', '@date{%F}'),
						('Close', '$@{close}{%0.2f}'),
						],
					formatters = {
						'date': 'datetime',
						'close': 'printf',
					})
	aapl_graph = figure(plot_width=800, plot_height=250, x_axis_type='datetime',
		title = 'Closing Data of AAPL')
	aapl_graph.line('date', 'close', color='green', alpha=.5, source=aapl_cds)
	aapl_graph.add_tools(hover)
	aapl_graph.title.align='center'
	aapl_script, aapl_div = components(aapl_graph)
	
	default_graphs = {
		'company_aapl': company_aapl,
		'aapl_script': aapl_script,
		'aapl_div': aapl_div,
	}"""
	context = {}
	
	return HttpResponse(template.render(context, request))
	
def contact(request):
	template = loader.get_template('stocks/contact.html')
	context = {}
	return HttpResponse(template.render(context, request))
	
"""def search(request):
	template = loader.get_template('post_list.html')
	
	query = requests.GET.get('usr_search')
	test = {
		'test':'test'
	}

	return HttpResponse(template.render(test, request))"""
