# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:14:36 2019

@author: Josh
"""

"""I built this for a research project in a class, my objective was to
    test whether color makes a difference in car price (specifically the color red),
    using a Welch's T-Test. This script scrapes car listings on cars.com and gives a csv
    file of the cars price and its color."""

from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup

def get_url(page_number):
    return 'https://www.cars.com/for-sale/searchresults.action/?mdId=20823&mkId=20017&mlgId=28872&page='+str(page_number)+'&perPage=100&rd=99999&searchSource=GN_REFINEMENT&shippable-dealers-checkbox=true&showMore=false&sort=relevance&yrId=58487&yrId=30031936&yrId=35797618&yrId=36362520&zc=60542&localVehicles=false'
#this url is searching for 2018-2019 honda civics under 10k miles. You can configure it to whatever car(s) or model(s) you want. 
filename = 'cars.csv' #output filename
f = open(filename, 'w')

headers = 'Price,Color\n'
f.write(headers)
for i in range(1, 51):#loop through pages of cars..
    uClient = ureq(get_url(i))#opens url
    page_html = uClient.read()#reads page html
    uClient.close()#closes url
    page_soup = soup(page_html, 'html.parser')#parses html
    containers = page_soup.findAll('div', {'class':'shop-srp-listings__inner'})#finds all of the listing blocks. I have included a sample picture of what a listing block looks like
    for container in containers: #loops through each listing block on each page. I have it set to 100 listings per page in my link.
        price_container = container.find('div', 'payment-section')
        price = price_container.span.text.strip()
        if price == 'Not Priced': #some prices are not listed, so we do not want to include these since the cars will be of no use
            continue
        price = price.replace(',', '')
        price = price.replace('$', '')#gives price as a readable number
        
        color_container = container.find('ul', 'listing-row__meta')#gives color
        color = color_container.li.text.replace('Ext. Color:', '').strip()#get color as a clean string
        
        f.write(price + ',' + color + ',' + '\n')
f.close()