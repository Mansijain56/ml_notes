import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv("salaries.csv")
# print(df.head())
#droping the salary_more_then_100k column
df1 = df.drop('salary_more_then_100k',axis='columns')
df2 = df['salary_more_then_100k']
# to convert group in digits ,, like company abc,xyz,asd = 1,2,3
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

# adding the columns in the 
df1["company_number"] =  le_company.fit_transform(df1['company'])
df1["job_number"] =  le_company.fit_transform(df1['job'])
df1["degree_number"] =  le_company.fit_transform(df1['degree'])
df1 = df1.drop(['company','job','degree'],axis='columns') 
# print(df1)

model = tree.DecisionTreeClassifier()
model.fit(df1,df2)
# print(model.score(df1,df2))
print(model.predict([[2,0,0]]))

