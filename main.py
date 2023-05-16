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

# get rid of incomplete rows
data = data.dropna()

#Convert dates as object types to TimeStamp types
data['Date'] = pd.to_datetime(data['Date'],format="%b %Y")

# Clean the percentages in column headline, getting rid of the '%' and converting values to floats
data['Headline'] = data['Headline'].str.replace("%","").astype(float)
# Create a first figure
plt.figure(1)
# Create a new subplot
plt.subplot(211)
# Plot the column date with headline
plt.plot(data['Date'],data['Headline'],label='Headline')
# Define labels
plt.xlabel("Year")
plt.ylabel("Inflation rate (%)")

# Clean all other columns with percentages
for col in data.columns[2:]:
                 data[col] = data[col].str.replace("%","").astype(float)

#Plot the remaining columns with respect to time
plt.plot(data['Date'],data['Core'], label='Core')
plt.plot(data['Date'],data['Regulated Items'],label='Regulated Items')
#plt.plot(data['Date'],data['Fruits and Vegetables'],label='Fruits and Veggies')

# Show legend
plt.legend()


# Now the external debt file
file2 = "C:\\Users\\karim\\Downloads\\egypt-external-debt-stock.csv"
# read the csv file
data2 = pd.read_csv(file2,sep=',',names=['date','debt','Percent increase'],skiprows=[0,1]+list(range(1,48)))

# Converting the date object values to TimeStamp values
data2['date'] = pd.to_datetime(data2['date'],format='%Y-%m-%d')

# Get rid of incomplete rows
data2 = data2.dropna()

# Convert Column debt to float
data2['debt'] = data2['debt'].astype(float)

# Create a numpy array using the columns, x and y
y = np.array(data2['debt'])#.reshape(-1,1)

# As we cannot calculate the linear regression with non numerical values for dates, 
#the difference between the current date and the first date is taken, in terms of years

x = np.array((data2['date']-data2['date'].iloc[0]).dt.total_seconds()/86400/365).reshape(-1,1)

# Create linear regression model
model = LinearRegression()
# Fit the curve
model.fit(x,y)
r_sq = model.score(x,y)

# Get the function
y1 = model.intercept_+model.coef_*x

# New subplot
plt.subplot(313)
# Add the predicted debt to subplot
plt.plot(data2['date'],y1,label='predicted debt')

# Add the normal debt to subplot
plt.plot(data2['date'],data2['debt'],label='debt')

# Correct the yticks for clear steps on the y axis
plt.yticks(np.arange(0,1.7e+11,step=2e+10))

# Show legend
plt.legend()

#
model.fit(x,y1)

# Creating new numpy array holding year values from 2022 to 2030, these years are the years we want to predict
future_years = np.arange(2022,2031,1)

#Initialize a numpy array with 0s that will hold the new timestamps for the new years
new_dates = np.zeros(9,dtype= pd.Timestamp)

#add each date to the new datese array
for year in future_years:
        new_dates[year-2022] = pd.to_datetime(f"{year}-12-31")
# create a new float numpy array that will hold the numerical difference in years with the base year, this is 
# used for the modeling
new_years = np.zeros(9)
i = 0 # Counter for index in new_years array
for date in new_dates:
        new_year = (new_dates[i]-data2['date'].iloc[0])
        new_years[i] = new_year.total_seconds()/(86400*365)
        i = i+1
new_years = new_years.reshape(-1,1)
future_years = future_years.reshape(-1,1)
# Use the predict function to get the values for the new years
predicted = model.predict(new_years)

new_rows =[]#
#loop through each new year
for year in future_years:
         # create row, with corresponding values
         new_row = pd.DataFrame({'date':pd.to_datetime(f"{year[0]}-12-31"),'debt': predicted[year[0]-2022][0]},index = [0])
         #add the row to the end of the data frame
         data2 = data2._append(new_row,ignore_index=True)
# Plot with the new Values
plt.plot(data2['date'].iloc[20:],data2['debt'].iloc[20:],label = 'New Expected debt')
plt.legend()
plt.show()
