# -*- coding: utf-8 -*-
"""ML_group.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b6JktKhJ2xs3ETw5qJWyh1mB4bp9Zg4n
"""

pip install pytrends



import numpy as np
import pandas as pd
import yfinance as yf


# to scrap the the stock data from yahoo finance 
def scrapStocks(stocklist,duration):
  for stock in stocklist:
    data_df=yf.download(stock,period=duration)
    data_df.to_csv(stock+'.csv')

# It will create the files for all the stocks
stocksName=['ADANIPOWER.NS','TATAPOWER.NS','RPOWER.NS','JKCEMENT.NS','AMBUJACEM.NS','ACC.NS']
scrapStocks(stocksName,"90d")

import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression  
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score


# Reading the consolidated file of stocks
df= pd.read_csv('stockData.csv')

"""# New Section"""

# droping the null values
df=df.dropna()

# with six input features
data = df[['OPEN','LOW','CLOSE','NO. OF SHARES', 'NO. OF TRADES', 'Tata powers']]
output =df[['HIGH']]

print(len(data))
#Splitting between training and test 80,20 split

Xtrain= data[: int(0.8*len(data))]
Ytrain= output[: int(0.8*len(data))]
Xtest=data[int(0.8*len(data)) :]
Ytest= output[int(0.8*len(data)):]

model = LinearRegression()
model.fit(Xtrain,Ytrain)
model.coef_


# calculating Mean Squared Error 
print('MSE test %.3f' % (mean_squared_error(model.predict(Xtest), Ytest)))
# calculating R-Squared
print('R^2  test: %.3f' % (r2_score(model.predict(Xtest), Ytest)))
# calculating absolute error
print('MAE test: %.3f' % (mean_absolute_error(model.predict(Xtest), Ytest)))
# calculating absolute percentage error
print('MAE test: %.3f' % (mean_absolute_percentage_error(model.predict(Xtest), Ytest)))

# adding new featires with rolling 3point and 5 point average

df['MVA3'] = df['OPEN'].shift(1).rolling(window=3).mean() 
df['MVA5']= df['OPEN'].shift(1).rolling(window=5).mean()

df=df.dropna()
df.head(10)

data = df[['OPEN','LOW','CLOSE','NO. OF SHARES', 'NO. OF TRADES', 'Tata powers','MVA3','MVA5']]
output =df[['HIGH']]

data

data
print(len(data))
#Splitting between training and test 90,10 split

Xtrain= data[: int(0.9*len(data))]
Ytrain= output[: int(0.9*len(data))]
Xtest=data[int(0.9*len(data)) :]
Ytest= output[int(0.9*len(data)):]

# training the Linear Regression model

model = LinearRegression()
model.fit(Xtrain,Ytrain)

model.coef_

# Error estimation with new features
# calculating Mean Squared Error 
print('MSE test %.3f' % (mean_squared_error(model.predict(Xtest), Ytest)))
# calculating R-Squared
print('R^2  test: %.3f' % (r2_score(model.predict(Xtest), Ytest)))
# calculating absolute error
print('MAE test: %.3f' % (mean_absolute_error(model.predict(Xtest), Ytest)))
# calculating absolute percentage error
print('MAE test: %.3f' % (mean_absolute_percentage_error(model.predict(Xtest), Ytest)))

#plotting the results

date1=df['DATE'][int(0.9*len(df)):]
import matplotlib.pyplot as plt
plt.plot(figsize=(10,5))
plt.plot(date1,model.predict(Xtest))
plt.plot(date1,Ytest)
plt.legend(['predicted_price','actual_price'])
plt.xlabel('Date') 
plt.ylabel('Stock price')
plt.gcf().autofmt_xdate()
plt.show



# training lasso regression
dictParm={}
dictIntercept={}
mean_error=[]; std_error=[]
Ci_range = [ 0.5, 1, 5, 10,100, 500]
from sklearn.linear_model import Lasso
for Ci in Ci_range:
  model3 = Lasso(alpha=1/(2*Ci))
  model3.fit(Xtrain,Ytrain)
  ypred = model3.predict(Xtest)
  print("Lasso Regression for C= "+str(Ci))
  print("Coefficient =" +str(model3.coef_))
  print("Theta0 =" +str(model3.intercept_))
  param=model3.coef_
  dictParm[Ci]=param
  dictIntercept[Ci]=model3.intercept_

# selecting the best value from above calculation
Ci=70
model3 = Lasso(alpha=1/(2*Ci))
model3.fit(Xtrain,Ytrain)
ypred = model.predict(Xtest)

# calculating Mean Squared Error with new features
print('MSE test %.3f' % (mean_squared_error(model3.predict(Xtest), Ytest)))
# calculating R-Squared
print('R^2  test: %.3f' % (r2_score(model3.predict(Xtest), Ytest)))
# calculating absolute error
print('MAE test: %.3f' % (mean_absolute_error(model3.predict(Xtest), Ytest)))
# calculating absolute percentage error
print('MAE test: %.3f' % (mean_absolute_percentage_error(model3.predict(Xtest), Ytest)))

import matplotlib.pyplot as plt
plt.plot(figsize=(10,5))
plt.plot(date1,model3.predict(Xtest))
plt.plot(date1,Ytest)
plt.legend(['predicted_price','actual_price'])
plt.xlabel('Date') 
plt.ylabel('Stock price')
plt.gcf().autofmt_xdate()
plt.show

# training the support vector regression model

from sklearn.svm import SVR
model2 = SVR(kernel = 'rbf')
model2.fit(Xtrain, Ytrain.values.ravel())

# calculating Mean Squared Error with new features
print('MSE test %.3f' % (mean_squared_error(model2.predict(Xtest), Ytest)))
# calculating R-Squared
print('R^2  test: %.3f' % (r2_score(model2.predict(Xtest), Ytest)))
# calculating absolute error
print('MAE test: %.3f' % (mean_absolute_error(model2.predict(Xtest), Ytest)))
# calculating absolute percentage error
print('MAE test: %.3f' % (mean_absolute_percentage_error(model2.predict(Xtest), Ytest)))

#predicted_price = pd.DataFrame(predicted_price,index=Ytest.index,columns = ['price']) 
predicted_price = pd.DataFrame(model2.predict(Xtest),date1,columns = ['price'])
predicted_price.plot(figsize=(10,5))  
plt.legend(['predicted_price','actual_price'])  
plt.ylabel("Price")  
plt.show()

#SVM is not for regression it for classification

test11

stockprofile = []
stocksName=['ADANIPOWER.NS','TATAPOWER.NS','RPOWER.NS','JKCEMENT.NS','AMBUJACEM.NS','ACC.NS']
for stock in stocksName:
  info=[]
  ticker = yf.Ticker(stock)
  info.append(stock)
  info.append(ticker.info['industry'])
  info.append(ticker.info['totalRevenue'])
  info.append(ticker.info['volume'])
  info.append(ticker.info['sector'])
  stockprofile.append(info)
df1 = pd.DataFrame(stockprofile, columns = ['Name','industry','totalRevenue','volume','sector'])
df1.to_csv('StockInfo.csv')

df1

df1['industry']=df1['industry'].map({'Utilities—Independent Power Producers':'1', 'Utilities—Renewable':'2', 'Building Materials': '3'})
df1['sector']=df1['sector'].map({'Utilities':'1','Basic Materials':'2'})

df1

from sklearn import svm
model4 = svm.SVC()
X =df1[['totalRevenue','volume','sector']][1:]
y=df1[['industry']][1:]
xtest =df1[['totalRevenue','volume','sector']][:1]

model4.fit(X,y.values.ravel())

model4.predict(xtest)