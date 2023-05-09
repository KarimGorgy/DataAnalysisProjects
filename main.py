import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime,timedelta
from sklearn.linear_model import LinearRegression


# Assigning file path to variable
file = "C:\\Users\\karim\\Downloads\\Inflations Historical (1).csv"
# assign data read from file to variable data
# gave it: file path, separator here as a comma, list of column names, and which rows to skip
data = pd.read_csv(file,sep = ',',names=['Date','Headline','Core','Regulated Items','Fruits and Vegetables'],skiprows=[0,1]+list(range(282,294)))


#print(data.shape)
#print(data.columns)
data = data.dropna()
#print(data.loc[0])
#print(data.shape)
#print(data)
#print(pd.to_datetime(data['Date']))
data['Date'] = pd.to_datetime(data['Date'],format="%b %Y")
data['Headline'] = data['Headline'].str.replace("%","").astype(float)
plt.figure(1)
plt.subplot(211)
plt.plot(data['Date'],data['Headline'],label='Headline')
plt.xlabel("Year")
plt.ylabel("Inflation rate (%)")

#plt.subplot(212)
#print(data[data.columns[1]])
for col in data.columns[2:]:
                 data[col] = data[col].str.replace("%","").astype(float)
#plt.plot(data['Date'],data['Core'],data['Date'],data['Regulated Items'],data['Date'],data['Fruits and Vegetables'])
plt.plot(data['Date'],data['Core'], label='Core')
plt.plot(data['Date'],data['Regulated Items'],label='Regulated Items')
plt.plot(data['Date'],data['Fruits and Vegetables'],label='Fruits and Veggies')
plt.legend()

#plt.show()

file2 = "C:\\Users\\karim\\Downloads\\egypt-external-debt-stock.csv"

data2 = pd.read_csv(file2,sep=',',names=['date','debt','Percent increase'],skiprows=[0,1]+list(range(1,48)))
data2['date'] = pd.to_datetime(data2['date'],format='%Y-%m-%d')
data2 = data2.dropna()
#print(data2)
data2['debt'] = data2['debt'].astype(float)
#print((data2['date']-data2['date'].iloc[0]).dt.total_seconds()/86400/365)

y = np.array(data2['debt'])#.reshape(-1,1)
x = np.array((data2['date']-data2['date'].iloc[0]).dt.total_seconds()/86400/365).reshape(-1,1)
model = LinearRegression()
model.fit(x,y)
r_sq = model.score(x,y)
y1 = model.intercept_+model.coef_*x
#print(r_sq)
plt.subplot(313)
plt.plot(data2['date'],y1,label='predicted debt')
#plt.subplot(414)
plt.plot(data2['date'],data2['debt'],label='debt')
plt.yticks(np.arange(0,1.7e+11,step=2e+10))
plt.legend()
#plt.show()
model.fit(x,y1)
future_years = np.arange(2024,2031,1)
print(future_years)
new_dates = np.zeros(7,dtype= pd.Timestamp)
for year in future_years:
        new_dates[year-2024] = pd.to_datetime(f"{year}-12-31")
print(new_dates)
new_years = np.zeros(7)
i = 0
for date in new_dates:
        print(new_dates[i]-data2['date'].iloc[0])
        new_year = (new_dates[i]-data2['date'].iloc[0]).days.total_seconds()/86400/365
        new_years[i] = new_year
        i = i+1

future_years = future_years.reshape(-1,1)
print(future_years)
predicted = model.predict(new_years)
print(predicted)
new_rows =[]# pd.DataFrame(columns=['date','debt'])
print(future_years)
for year in future_years:
         
         new_row = pd.DataFrame({'date':pd.to_datetime(f"{year[0]}-12-31"),'debt': predicted[year[0]-2024][0]},index = [0])
         data2 = data2._append(new_row,ignore_index=True)
plt.plot(data2['date'],data2['debt'],label = 'new Expected debt')
plt.show()
#print(data2.shape)
#data2 = pd.concat([data2]+new_rows,ignore_index=True)
print(data2)

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