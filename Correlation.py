# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:17:39 2021

@author: rache
"""
import pandas as pd
from Google import Create_Service
import datetime
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.seasonal import seasonal_decompose

CLIENT_SECRET_FILE  = 'credentials.json'
API_NAME = 'sheets'
API_VERSION = 'v4'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
spreadsheetID= '1JYK5ckijfCUds-uhU0qpCOtDim8HO6CPzzX9GSit0BU'

service= Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
gs=service.spreadsheets()
rows1=gs.values().get(spreadsheetId=spreadsheetID, range='HackbridgeTemperatureData!A2:C').execute()
rows2=gs.values().get(spreadsheetId=spreadsheetID, range='WeatherAPIData!A2:D').execute()
rows3=gs.values().get(spreadsheetId=spreadsheetID, range='TrafficData!A2:C').execute()
data1=rows1.get('values')
data2=rows2.get('values')
data3=rows3.get('values')
df1=pd.DataFrame(data1)
df2=pd.DataFrame(data2)
df3=pd.DataFrame(data3)
# print(df1)
# print(df2)
# print(df3)

######## On Google Sheets #######
#I need to remove the humidity column
# Any missing data I need to add this in with night and a 0 next to it 

#I need to do a regression analysis 



####### Turning the weather column into numbers so that they can be compared ######
Weather = {'clear sky': 1, 'haze': 2, 'overcast clouds': 6, 'broken clouds': 5, 'scattered clouds': 4, 'few clouds': 3, 'mist': 7, 'fog': 8, 'light rain': 9, 'light intensity drizzle': 10, 'light intensity drizzle rain': 11, 'drizzle': 12, 'drizzle rain': 13, 'moderate rain': 14, 'shower rain': 16, 'smoke': 18, 'light intensity shower rain': 15, 'thunderstorm': 19, 'light snow': 20}
df2.iloc[:,2] = [Weather[item] for item in df2.iloc[:,2]]

####### Getting the date and time in the appropriate format ######

df1date = [int(0)] * len(df1.iloc[:,0])
for i in range(0,len(df1.iloc[:,0])):
    df1date[i] = datetime.datetime(int(df1.iloc[i,0][4:8]), int(df1.iloc[i,0][2:4]), int(df1.iloc[i,0][0:2]),int(df1.iloc[i,0][9:11]),int(df1.iloc[i,0][11:13]),int(df1.iloc[i,0][13:15]))

df3date = [int(0)] * len(df3.iloc[:,0])
for i in range(0,len(df3.iloc[:,0])):
    df3date[i] = datetime.datetime(int(df3.iloc[i,0][4:8]), int(df3.iloc[i,0][2:4]), int(df3.iloc[i,0][0:2]),int(df3.iloc[i,0][9:11]),int(df3.iloc[i,0][11:13]),int(df3.iloc[i,0][13:15]))


####### Putting the cars in 5 minutes buckets ######
oricarBin = [int(0)] * len(df1.iloc[:,0])
oricarDayBin = [int(0)] * len(df1.iloc[:,0])
tempCarIndex = 0
for i in range(0,len(df1.iloc[:,2])-1):
    tempCounter = 0;
    binEndTime = df1date[i+1];
    oricarDayBin[i] = df3.iloc[tempCarIndex,2]
    
    for j in range(tempCarIndex,len(df3date)):
        if (df3date[j]<binEndTime):
                tempCounter = tempCounter+int(df3.iloc[j,1]);
        else:
            tempCarIndex=j
            break
    oricarBin[i]=tempCounter

####combine all of the datasets together#####
#only one date time module 
# carbin=pd.Series(oricarBin)
# df4 = pd.concat([df1, df2], axis=1)
# df5 = pd.concat([df4, carbin], axis=1)
# #print(df5)

col0=pd.Series(df1date[:], name='Seconds')
col1=pd.Series(df1.iloc[:,1], name='HackbridgeTemp',dtype="float64")
col2=pd.Series(df2.iloc[:,2], name='WeatherAPI',dtype="int64")
col3=pd.Series(df2.iloc[:,3], name='WeatherAPItemp',dtype="float64")
col4=pd.Series(oricarBin, name='Cars',dtype="int64")
col5=pd.Series(oricarDayBin, name='Day/Night')

df6 = pd.concat([col0, col1, col2, col3, col4, col5], axis=1)
#print(df6)
#### remove all variables when it is too dark to take a photo ######
df7 = df6[~(df6 == "Night").any(axis=1)]
#print(df7)




######## removing outliers ###########
df7 = df7.iloc[:,0:5]


arbitrary_date = datetime.datetime(2021,1,1)
df7.iloc[:,0] = [(d - arbitrary_date).total_seconds() for d in df7.iloc[:,0]]


for x in range(1,5):
    for y in range(len(df7)):
        df7.iloc[y,x] = float(df7.iloc[y,x])
     
df7.boxplot(column=(['HackbridgeTemp','WeatherAPItemp']))
# dfboxplot.boxplot(column=(['Cars']))

# boxplot.set_size_inches(20,8)


I1Q1 = df7.iloc[:,1].quantile(0.00)
I1Q3 = df7.iloc[:,1].quantile(0.95)
# I4Q1 = df7.iloc[:,4].quantile(0)
# I4Q3 = df7.iloc[:,4].quantile(0.98)


I1index = df7[(df7.iloc[:,1]>= I1Q3)|(df7.iloc[:,1] <= I1Q1)].index
# I4index = df7[(df7.iloc[:,4]>= I4Q3)|(df7.iloc[:,4] <= I4Q1)].index


df7.drop(I1index, inplace=True)


################# seasonlity ############



TempCorr = ['DateTime', 'HackbridgeTemp', 'WeatherAPI','WeatherAPItemp', 'Cars']

dateXLabels = [""] * len(df6.iloc[:,2])
dateXLabels[0]="02 Jan"
dateXLabels[158]="03 Jan"
dateXLabels[325]="04 Jan"

dateXLabels[448]="05 Jan"

dateXLabels[618]="06 Jan"

dateXLabels[713]="07 Jan"
dateXLabels[858]="08 Jan"
dateXLabels[989]="09 Jan"
dateXLabels[1071]="10 Jan"




# for i in range(1021):
#     test[i]=1021-i
trend_series = []
for i in range(0,5):
    # print(i)
    decomposed = seasonal_decompose(df7.iloc[:,i],period=round(1163/9))
    trend_series.append(decomposed.trend)
    figure = decomposed.plot()
    figure.axes[0].set_title(TempCorr[i])
    # figure.axes[0].set_xticks(len(df8.iloc[:,i]))
    figure.set_figheight(6)
    figure.set_figwidth(15)
    figure.tight_layout()
    
    for j in range(4):
        figure.axes[j].set_xticks(np.arange(len(dateXLabels)))
    
        figure.axes[j].set_xticklabels([])
        
    figure.axes[3].set_xticklabels(dateXLabels)


trends = pd.concat(trend_series, axis=1)
plt.show()


#######correlation against time #########
Time_HackbridgeTemp, _ = pearsonr(df7.iloc[:,0] , df7.iloc[:,1])
print('time vs hackbridge temp Pearsons correlation: %.3f' % Time_HackbridgeTemp)
Time_APIWeather, _ = pearsonr(df7.iloc[:,0] , df7.iloc[:,2])
print('time vs Weather Number Pearsons correlation: %.3f' % Time_APIWeather)
Time_APIWeatherTemp, _ = pearsonr(df7.iloc[:,0] , df7.iloc[:,3])
print('time vs API Temperature Pearsons correlation: %.3f' % Time_APIWeatherTemp)
Time_cars, _ = pearsonr(df7.iloc[:,0] , df7.iloc[:,4])
print('time vs Traffic Data Pearsons correlation: %.3f' % Time_cars)

######correlation agains hackbridge temp ######
HackbridgeTemp_WeatherNumber, _ = pearsonr(df7.iloc[:,1], df7.iloc[:,2])
print('Hackbridge Temp vs Weather Number Pearsons correlation: %.3f' % HackbridgeTemp_WeatherNumber)
HackbridgeTemp_TempAPI , _ = pearsonr(df7.iloc[:,1], df7.iloc[:,3])
print('Hackbridge Temp vs TempAPI Pearsons correlation: %.3f' % HackbridgeTemp_TempAPI)
HackbridgeTemp_TrafficData, _ = pearsonr(df7.iloc[:,1], df7.iloc[:,4])
print('Hackbridge Temp vs Traffic Data Pearsons correlation: %.3f' % HackbridgeTemp_TrafficData)

######correlation agains Weather Number ######
WeatherNumber_WeatherAPI, _ = pearsonr(df7.iloc[:,2], df7.iloc[:,3])
print('Weather Number vs TempAPI Pearsons correlation: %.3f' % WeatherNumber_WeatherAPI)
WeatherNumber_TrafficData, _ = pearsonr(df7.iloc[:,2], df7.iloc[:,4])
print('Weather Number vs Traffic Data Pearsons correlation: %.3f' % WeatherNumber_TrafficData)

TempAPI_TrafficData, _ = pearsonr(df7.iloc[:,3], df7.iloc[:,4])
print('Temp API vs Traffic Data Pearsons correlation: %.3f' % TempAPI_TrafficData)


################ linier regression #########

linear_regressor = LinearRegression()  # create object for the class

####################################
model = linear_regressor.fit(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,1].values.reshape((-1, 1)))
r_sq = model.score(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,1].values.reshape((-1, 1)))
print('Time vs Hackbridge Temperature coefficient of determination:', r_sq)
Y_pred = linear_regressor.predict(df7.iloc[:,0].values.reshape((-1, 1)))  # make predictions

plt.scatter(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,1].values.reshape((-1, 1)))
plt.plot(df7.iloc[:,0].values.reshape((-1, 1)), Y_pred, color='red')
plt.title('Time vs Hackbridge Temperature')
plt.show()

####################################
model = linear_regressor.fit(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,2].values.reshape((-1, 1)))
r_sq = model.score(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,2].values.reshape((-1, 1)))
print('Time vs Weather Number coefficient of determination:', r_sq)
Y_pred = linear_regressor.predict(df7.iloc[:,0].values.reshape((-1, 1)))  # make predictions

plt.scatter(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,2].values.reshape((-1, 1)))
plt.plot(df7.iloc[:,0].values.reshape((-1, 1)), Y_pred, color='red')
plt.title('Time vs Weather Number')
plt.show()

####################################
model = linear_regressor.fit(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,3].values.reshape((-1, 1)))
r_sq = model.score(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,3].values.reshape((-1, 1)))
print('Time vs API Temperature coefficient of determination:', r_sq)
Y_pred = linear_regressor.predict(df7.iloc[:,0].values.reshape((-1, 1)))  # make predictions

plt.scatter(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,3].values.reshape((-1, 1)))
plt.plot(df7.iloc[:,0].values.reshape((-1, 1)), Y_pred, color='red')
plt.title('Time vs API Temperature')
plt.show()

####################################
model = linear_regressor.fit(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
r_sq = model.score(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
print('Time vs Traffic Data coefficient of determination:', r_sq)
Y_pred = linear_regressor.predict(df7.iloc[:,0].values.reshape((-1, 1)))  # make predictions

plt.scatter(df7.iloc[:,0].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
plt.plot(df7.iloc[:,0].values.reshape((-1, 1)), Y_pred, color='red')
plt.title('Time vs Traffic Data')
plt.show()

####################################
model = linear_regressor.fit(df7.iloc[:,1].values.reshape((-1, 1)), df7.iloc[:,2].values.reshape((-1, 1)))
r_sq = model.score(df7.iloc[:,1].values.reshape((-1, 1)), df7.iloc[:,2].values.reshape((-1, 1)))
print('Hackbridge Temperature vs Weather Number coefficient of determination:', r_sq)
Y_pred = linear_regressor.predict(df7.iloc[:,1].values.reshape((-1, 1)))  # make predictions

plt.scatter(df7.iloc[:,1].values.reshape((-1, 1)), df7.iloc[:,2].values.reshape((-1, 1)))
plt.plot(df7.iloc[:,1].values.reshape((-1, 1)), Y_pred, color='red')
plt.title('Hackbridge Temperature vs Weather Number ')
plt.show()

####################################
model = linear_regressor.fit(df7.iloc[:,1].values.reshape((-1, 1)), df7.iloc[:,3].values.reshape((-1, 1)))
r_sq = model.score(df7.iloc[:,1].values.reshape((-1, 1)), df7.iloc[:,3].values.reshape((-1, 1)))
print('Hackbridge Temperature vs API Temperature coefficient of determination:', r_sq)
Y_pred = linear_regressor.predict(df7.iloc[:,1].values.reshape((-1, 1)))  # make predictions

plt.scatter(df7.iloc[:,1].values.reshape((-1, 1)), df7.iloc[:,3].values.reshape((-1, 1)))
plt.plot(df7.iloc[:,1].values.reshape((-1, 1)), Y_pred, color='red')
plt.title('Hackbridge Temperature vs API Temperature')
plt.show()

####################################
model = linear_regressor.fit(df7.iloc[:,1].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
r_sq = model.score(df7.iloc[:,1].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
print('Hackbridge Temperature vs Traffic Data coefficient of determination:', r_sq)
Y_pred = linear_regressor.predict(df7.iloc[:,1].values.reshape((-1, 1)))  # make predictions

plt.scatter(df7.iloc[:,1].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
plt.plot(df7.iloc[:,1].values.reshape((-1, 1)), Y_pred, color='red')
plt.title('Hackbridge Temperature vs Traffic Data')
plt.show()

####################################
model = linear_regressor.fit(df7.iloc[:,2].values.reshape((-1, 1)), df7.iloc[:,3].values.reshape((-1, 1)))
r_sq = model.score(df7.iloc[:,2].values.reshape((-1, 1)), df7.iloc[:,3].values.reshape((-1, 1)))
print('Weather Number vs API Temperature coefficient of determination:', r_sq)
Y_pred = linear_regressor.predict(df7.iloc[:,2].values.reshape((-1, 1)))  # make predictions

plt.scatter(df7.iloc[:,2].values.reshape((-1, 1)), df7.iloc[:,3].values.reshape((-1, 1)))
plt.plot(df7.iloc[:,2].values.reshape((-1, 1)), Y_pred, color='red')
plt.title('Weather Number vs API Temperature')
plt.show()

####################################
model = linear_regressor.fit(df7.iloc[:,2].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
r_sq = model.score(df7.iloc[:,2].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
print('Weather Number vs Traffic Data coefficient of determination:', r_sq)
Y_pred = linear_regressor.predict(df7.iloc[:,2].values.reshape((-1, 1)))  # make predictions

plt.scatter(df7.iloc[:,2].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
plt.plot(df7.iloc[:,2].values.reshape((-1, 1)), Y_pred, color='red')
plt.title('Weather Number vs Traffic Data')
plt.show()


####################################
model = linear_regressor.fit(df7.iloc[:,3].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
r_sq = model.score(df7.iloc[:,3].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
print('Temp API vs Traffic Data coefficient of determination:', r_sq)
Y_pred = linear_regressor.predict(df7.iloc[:,3].values.reshape((-1, 1)))  # make predictions

plt.scatter(df7.iloc[:,3].values.reshape((-1, 1)), df7.iloc[:,4].values.reshape((-1, 1)))
plt.plot(df7.iloc[:,3].values.reshape((-1, 1)), Y_pred, color='red')
plt.title('Temp API vs Traffic Data')
plt.show()

df7.to_csv (r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\GITHUB.csv', index = False, header=False)