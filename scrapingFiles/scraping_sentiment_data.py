#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import yfinance as yf

#scrap data method that scraps the data of the companies list for the duration passed
def scrapData(stocklist,duration):
    for stock in stocklist:
       data_df =yf.download(stock,period=duration)
       data_df.to_csv(stock+duration+'.csv')
#scraping google stock data for 30 days
scrapData(["GOOGL"],"30d")
#scraping tesla stock data for 90 days
scrapData(["TSLA"],"90d")

#tesla selected period historical data for sentiment analysis
TSLA = yf.Ticker('TSLA')
oldTSLA = TSLA.history(start="2019-01-01",  end="2019-12-31")
oldTSLA.to_csv("stockTSLA.csv")

