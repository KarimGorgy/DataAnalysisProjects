import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
""""
def func(t):
    return t*2
x =np.array([1,2,3,4,4])
plt.figure(1)
plt.subplot(211)
plt.plot(x,func(x))
plt.subplot(212)
plt.plot([1,2,3])
plt.plot([4,5,6,6,6])
plt.show()
"""
file1 = "C:\\Users\\karim\\OneDrive\\Desktop\\Github\\DataAnalysisProjects\\egypt-external-debt-stock.csv"
file2 = "C:\\Users\\karim\\OneDrive\\Desktop\\Github\\DataAnalysisProjects\\PALLFNFINDEXQ.csv"
file3 = "C:\\Users\\karim\\OneDrive\\Desktop\\Github\\DataAnalysisProjects\\InflationsEgyptHistory.csv"
df1 = pd.read_csv(file1,sep=',',skiprows=list(range(0,17)),names=['date','dollars','change'])
df1['change'].loc[0] = 1.
df2 = pd.read_csv(file2,sep=',',names=['date','index'],skiprows=1)
#Date,Headline (y/y),Core (y/y),Regulated Items (y/y),Fruits and Vegetables (y/y)

df3 = pd.read_csv(file3,sep=',',names=['date','headline','core','regulated_items','fruits_and_veggies'])
df3 = df3.dropna()
df3 = df3.drop([1])
print(df3.head())
def parse_date(string):
    new_string = string[-4:]
    return new_string

df3['date'] = df3['date'].apply(parse_date)
df3['headline'] = df3['headline'].str.replace("%","")
def parse_string(string):
    parsed = string[:-6]
    return parsed


#df3['date'] = df3['date'].apply(parse_string)
df2['date'] = df2['date'].apply(parse_string)
df1['date'] = df1['date'].apply(parse_string)
df1['dollars'] = df1['dollars']/1e9
df2 = df2.drop_duplicates(subset=['date'])
df = pd.merge(df1,df2,on='date',how='outer')
df = pd.merge(df,df3,on='date',how='outer')
df['date'] = df['date'].astype(float)

plt.figure(1)
plt.subplot(211)
plt.plot(df['date'],df['index'],label='Commodities price index')
plt.ylabel('Index of Global Commodities price')
plt.xlabel('Year')
plt.subplot(212)
plt.plot(df['date'].loc[30:],df['dollars'].loc[30:],label='External Debt')
plt.ylabel('Debt in Billions of US $')
plt.xlabel('Year')
#plt.legend()
#plt.show()

df = df.dropna()
features = ['dollars','index','date']
target = 'headline'
dates = np.array(df['date']).reshape(-1,1)
modelIndex = LinearRegression()
#modelIndex.fit(df['date'],df['index'])
modelIndex.fit(dates,df['index'])

yIndex = modelIndex.coef_*df['date']+modelIndex.intercept_
plt.subplot(211)
plt.plot(df['date'],yIndex,label='expected Index')
modelDebt = LinearRegression()
#modelDebt.fit(df['dollars'],df['dollars'])
modelDebt.fit(dates,df['dollars'])
yDebt = modelDebt.coef_*df['date']+modelDebt.intercept_
plt.subplot(212)
plt.plot(df['date'],yDebt,label='expected Debt')
plt.legend()
plt.show()

model = LinearRegression()
model.fit(df[features],df[target])
dates = dates.flatten()
new_data = pd.DataFrame({'dollars':yDebt,'index':yIndex,'date':dates})

prediction = model.predict(new_data)
new_data['predicted_inflation'] = prediction
plt.figure(2)
plt.plot(new_data['date'],prediction)
plt.show()
print(new_data)