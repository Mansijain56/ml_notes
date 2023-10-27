import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# reading the data set csv file
df = pd.read_csv("homeprice.csv")
# print(df)

# ploting the graph 
plt.scatter(df['area'], df['price'], color='red',marker='*')
plt.plot(df['area'].values, df['price'].values, color='blue', label='Line Plot')
plt.xlabel('area')
plt.ylabel('price')
plt.legend()
plt.title('Scatter Plot of area vs. price')


# applyng the linear regression as well
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# testing out the the prediction of the same dataset
result = reg.predict( [ [3300] ] )
#  f = wx + b
print(result)
# x
print(reg.coef_)
# b
print(reg.intercept_)


# reading new data set without price column 
df1 = pd.read_csv("homeprice_empty.csv")
# print(df1)
# to predict the values of price in the dataset
p = reg.predict(df1)
# added the column p in the dataset
df1['p'] = p
print(df1)
# saved the csv file
df1.to_csv("prediction.csv")

#  plotted the predicted line 
plt.scatter(df1['area'], df1['p'], color='green', marker='o', label='Predicted Data')
plt.plot(df1['area'].values, df1['p'].values, color='yellow', label='predicted Line Plot')

plt.show()



