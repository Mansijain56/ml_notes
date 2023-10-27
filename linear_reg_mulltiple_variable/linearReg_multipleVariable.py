import pandas as pd
import numpy as np
from sklearn import linear_model
# reading the csbv file
df = pd.read_csv("homeprices.csv")
# calculating the median of the column
median = int(df.bedroom.median())
# fill the empty rows in the column
df.bedroom = df.bedroom.fillna(median)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedroom','age']],df.price)
# f = m1*area+m2*bedroom+m3*age+b
# m is coef
# b is intercept
print(reg.coef_)
print(reg.intercept_)

prediction = reg.predict([[4563,3,45]])
print(prediction)