import joblib
# Load the trained model from the joblib file
mp = joblib.load('house_joblib2')
result = mp.predict([[3300]])
print(result)