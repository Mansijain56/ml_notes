from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn


iris = load_iris()
print(dir(iris))
# o/p = ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']

# print(len(iris))
# print(iris.data[0])

# Let's just print the data for the first 5 samples as an example
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.5)
# print(len(X_train))
# print(len(X_test))
model = LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(model.predict([iris.data[62]]))

# as the score is the 0.96, 
# to see the where it went wrong
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test,y_predicted)
print(cm)

# seaborn to see the graph 
plt.figure(figsize=(10,10))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()