#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install flair


# In[2]:


import pandas as pd
import flair
from flair.data import Sentence

sentiment = []
probability = []

sentiment_model = flair.models.TextClassifier.load('en-sentiment')

df = pd.read_csv("tweets.csv",header=1, names = ['text','username','link','code','date'])
#print(df.head())

tweets = df.iloc[ : , 0 ]

for tweet in tweets.to_list():
    # make prediction
    sentence = Sentence(str(tweet))
    sentiment_model.predict(sentence)
    # extract sentiment prediction
    probability.append(sentence.labels[0].score)  # numerical score 0-1
    sentiment.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'
    

df['sentiment'] = sentiment
df['probability'] = probability

#print(df.head())
df.to_csv('sentiment'+'.csv')


# In[ ]:




