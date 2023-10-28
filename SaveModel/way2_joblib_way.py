#  used the split method , to test  only some amount of data set
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Reading the data set csv file
df = pd.read_csv("homeprice.csv")

# Define your feature matrix (X) and target vector (y)
X = df[['area']]  
y = df['price']

# Split the data into training and testing sets
# used 0.5 because of small data, use 0.2 for test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# print(len(X_train))
# print(len(X_test))
# Create a LinearRegression model
reg = LinearRegression()

# Fit the model to the training data
reg.fit(X_train, y_train)
print(reg.predict(X_test))
print(y_test)
print(reg.score(X_test,y_test))
# Saving the model to a joblib file
joblib.dump(reg, 'house_joblib2')
