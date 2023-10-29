# σ(x) = 1 / (1 + e^(-x))
# e is the base of the natural logarithm, approximately equal to 2.71828.
# x is the input value or a linear combination of input features.


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("insurance.csv")
# print(df)
# df.shape -- to see the columns and rows 
# print(df.shape)
# plt.scatter(df.age,df.insured,marker="*")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.insured, test_size = 0.1)
print(X_test)
model = LogisticRegression()
model.fit(X_train,y_train)
print(model.predict(X_test))
# accuracy of model-score , 1 i.e.model is perfect
print(model.score(X_test,y_test))
# probability
# print(model.predict_proba(X_test,))
predicted_probabilities = model.predict_proba(X_test)
results = pd.DataFrame({'X_test_values': X_test.values.flatten(), 'Probability': predicted_probabilities[:, 1]})
print(results)
#  one can see the the (problity of both positive and negative as well)
print(model.predict_proba(X_test))

