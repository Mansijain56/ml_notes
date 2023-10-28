# to testing out pickle file
import pickle

with open('house_pickel','rb') as f:
    mp = pickle.load(f)

result = mp.predict([[3300]])
print(result)