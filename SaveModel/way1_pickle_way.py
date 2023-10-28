import pandas as pd
from sklearn import linear_model

# reading the data set csv file
df = pd.read_csv("homeprice.csv")
# applyng the linear regression as well
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

#  saving object in the a Serializated file
import pickle
with open('house_pickel','wb') as f:
    pickle.dump(reg,f)
