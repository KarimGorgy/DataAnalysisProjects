import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime,timedelta
#from sklearn.linear_model import LinearRegression

# Assigning file path to variable
file = "C:\\Users\\karim\\Downloads\\Inflations Historical (1).csv"
# assign data read from file to variable data
# gave it: file path, separator here as a comma, list of column names, and which rows to skip
data = pd.read_csv(file,sep = ',',names=['Date','Headline','Core','Regulated Items','Fruits and Vegetables'],skiprows=[0,1]+list(range(282,294)))


#print(data.shape)
#print(data.columns)
data = data.dropna()
print(data.loc[0])
#print(data.shape)
#print(data)
#print(pd.to_datetime(data['Date']))
data['Date'] = pd.to_datetime(data['Date'],format="%b %Y")
data['Headline'] = data['Headline'].str.replace("%","").astype(float)
plt.figure(1)
plt.subplot(211)
plt.plot(data['Date'],data['Headline'])
plt.xlabel("Year")
plt.ylabel("Inflation rate (%)")

plt.subplot(212)
#print(data[data.columns[1]])
for col in data.columns[2:]:
                 data[col] = data[col].str.replace("%","").astype(float)
plt.plot(data['Date'],data['Core'],data['Date'],data['Regulated Items'],data['Date'],data['Fruits and Vegetables'])
#plt.show()

file2 = "C:\\Users\\karim\\Downloads\\egypt-external-debt-stock.csv"

data2 = pd.read_csv(file2,sep=',',names=['date','debt','Percent increase'],skiprows=[0,1]+list(range(1,48)))
data2['date'] = pd.to_datetime(data2['date'],format='%Y-%m-%d')
data2 = data2.dropna()
print(data2)
data2['debt'] = data2['debt'].astype(float)
plt.subplot(313)
plt.plot(data2['date'],data2['debt'])
plt.yticks(np.arange(0,1.7e+11,step=2e+10))
plt.show()
"""
fig,ax1 =  plt.subplots()
#data = data.dropna()
#Cleaning columns depending on their content
# For columns with percentages, get rid of the percentage sign
# For column with date, format it.
data['Headline'] = data['Headline'].str.replace("%","").astype(float)
data['Core'] = data['Core'].str.replace("%","").astype(float)
data['Regulated Items'] = data['Regulated Items'].str.replace("%","").astype(float)
data['Fruits and Vegetables'] = data['Fruits and Vegetables'].str.replace("%","").astype(float)
data['Date'] =pd.to_datetime(data['Date'], format = "%b %Y")

# Plot the different curves as a function of times
ax1.plot(data['Date'],data['Headline'], label='Headline')
ax1.plot(data['Date'],data['Core'],label= 'Core')



ax2 = ax1.twinx()
file2 = "C:\\Users\\karim\\Downloads\\egypt-external-debt-stock.csv"

data2 = pd.read_csv(file2,sep=',',names=['date','debt','Percent increase','Extra'],skiprows=[0,1]+list(range(1,48)))
data2['date'] = pd.to_datetime(data2['date'],format='%Y-%m-%d')

xaxis = np.array(range(0,21))

print(np.dtype(data2['debt']))
print(np.dtype(xaxis[1]))

ax2.plot(data2['date'],data2['debt'])
first_date = data2['date'].min()
data2['days'] = (data2['date'] - first_date).dt.days/365

# Calculate derivative
dy_dx = np.gradient(data2['debt'], data2['days'])
ax2.plot(data2['date'],dy_dx,label='derivative')

# Show legend and plot
ax2.legend()
ax1.legend()
plt.legend()
plt.show()
"""