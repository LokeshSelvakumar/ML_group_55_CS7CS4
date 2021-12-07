#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import matplotlib.pyplot as plot

stock = pd.read_csv("stocks_5days.csv",header=0, names = ['date','open','high', 'low', 'close','volume'])
stock
date_df = stock.iloc[:, 0:1]
high_df = stock.iloc[:, 2]
date =[5,6,7,8,9]
plot.plot(date, high_df)

tweets = pd.read_csv("tweets_elon.csv",header=0, names = ['text','name','link','code','date', 'sentiment', 'probability'])
sentiment_df = tweets.iloc[:,5]


plot.plot(date, sentiment_df)

