from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, Http404
from django.shortcuts import get_object_or_404
from . import services

# Create your views here.

def detail_view(request, symbol):
	"""
		Gives details on stock symbol.
		Parameters:
		- Request -> Request
		- Symbol -> Trading Symbol, such as SPX for SP500
	"""
	
	template = loader.get_template('stocks.detail.html')
	
	get_stock_info_api = services.get_stock(symbol)
	
	if get_stock_info_api is None:
		raise Http404("Stock {0} does not exist, try again.".format(symbol))
	
	stock_info = {
		'symbol': symbol,
		'today_close': get_stock_info_api['Close'][-1],#gets todays closing price
		'today_open': get_stock_info_api['Open'][-1],#gets todays open price
		'today_high': get_stock_info_api['High'][-1],#gets todays high
		'today_low': get_stock_info_api['Low'][-1],#gets todays low
		'today_volume': get_stock_info_api['Volume'][-1],
	}
	
	return HttpResponse(template.render(stock_quote, request))
	

def index(request):
	
	template = loader.get_template('stocks/index.html')
	context = {}
	return HttpResponse(template.render(context, request))
	