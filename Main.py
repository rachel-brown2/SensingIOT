# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:12:14 2020

@author: rache
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 20:59:48 2020
@author: rache
"""
#https://docs.google.com/spreadsheets/d/1JYK5ckijfCUds-uhU0qpCOtDim8HO6CPzzX9GSit0BU/edit#gid=1008245518

#https://github.com/makersdigest/T03-DHTXX-Temp-Humidity/blob/master/raspberry-pi/dhtxx_example.py
## using the DHT11 sensor which is good for 0-50 °C temperature readings +-2 °C accuracy
## using my own photos and also photos from here xxx to code the ML model
## using the waether API avalible here xxx to get london weather data 


from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
from utils import (get_hog_features, extract_features)
from utils import (find_cars, draw_boxes,draw_labeled_bboxes, add_heat, apply_threshold, process_frame,londonWeather,updateSheets)
from picamera import PiCamera
from time import sleep
import time
import datetime
from Google import Create_Service
import sys
import Adafruit_DHT     
# from sklearn.externals import joblib
import pickle
import requests
import Adafruit_GPIO.SPI as SPI
import Adafruit_MCP3008

####################### SETUP ########################
pin = 4                 # Set tempreture sensor to pin to pin 4
tempsensdly = 2         # Temp Sensor is slow so needs a 2 second delay
SPI_PORT   = 0 # the port the LDR is connected to 
SPI_DEVICE = 0
mcp = Adafruit_MCP3008.MCP3008(spi=SPI.SpiDev(SPI_PORT, SPI_DEVICE))
LDR_value=[0] #sets the initial value of the LDR to 0

LDR_value = mcp.read_adc(0) #getting the value from the LDR which is connected to a 1k resistor

with open('/home/pi/Documents/model1.pkl', 'rb') as f:
    svc=pickle.load(f) #the daytime model
with open('/home/pi/Documents/model1.pkl', 'rb') as f:
    svcN=pickle.load(f) #the nightime model 

    
    


################### Operational code starts here ###########################

camera = PiCamera()

timeout = time.time() + 60*60  #running for 10 days 

# Setting up google sheets

CLIENT_SECRET_FILE  = 'client_secret_london_weather_vs_traffic.json'
API_NAME = 'sheets'
API_VERSION = 'v4'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

service= Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
spreadsheetID= '1JYK5ckijfCUds-uhU0qpCOtDim8HO6CPzzX9GSit0BU'



range_ = 'WeatherAPIData!A1:Z1000'

clear_values_request_body = {} #CHECK THIS
        
request = service.spreadsheets().values().clear(spreadsheetId=spreadsheetID, range=range_, body=clear_values_request_body).execute()
    
cell_range_insert='WeatherAPIData!A1'
Headings = (('Date','Time','Weather','London Temperature'),('','','',''))
value_range_body = {'majorDimension': 'ROWS','values': Headings}
    
request=service.spreadsheets().values().append(spreadsheetId=spreadsheetID, range=cell_range_insert, valueInputOption='USER_ENTERED', body=value_range_body)
request.execute()

range_ = 'TrafficData!A1:B500000'

clear_values_request_body = {}
    
request = service.spreadsheets().values().clear(spreadsheetId=spreadsheetID, range=range_, body=clear_values_request_body).execute()

cell_range_insert='TrafficData!A1'
Headings = (('Date Time','Number of Cars'),('',''))
value_range_body = {'majorDimension': 'ROWS','values': Headings}

request=service.spreadsheets().values().append(spreadsheetId=spreadsheetID, range=cell_range_insert, valueInputOption='USER_ENTERED', body=value_range_body)
request.execute()

range_ = 'HackbridgeTemperatureData!A1:B50000'

clear_values_request_body = {}
    
request = service.spreadsheets().values().clear(spreadsheetId=spreadsheetID, range=range_, body=clear_values_request_body).execute()

cell_range_insert='HackbridgeTemperatureData!A1'
Headings = (('Date Time','Tempreture', 'Humidity'),('','',''))
value_range_body = {'majorDimension': 'ROWS','values': Headings}

request=service.spreadsheets().values().append(spreadsheetId=spreadsheetID, range=cell_range_insert, valueInputOption='USER_ENTERED', body=value_range_body)
request.execute()

# camera.start_preview()

weatherAPItime = datetime.datetime(1999,1,1,00,00,00)
cell_range_insert='TrafficData!A1'

while (time.time()<timeout):
    current_time = datetime.datetime.now()
    
    # ***** CAMERA CODE ****
    camera.capture('/home/pi/Desktop/ori_%s.jpg' % current_time.strftime("%d%m%Y_%H%M%S"))
    current_image = '/home/pi/Desktop/ori_%s.jpg' % current_time.strftime("%d%m%Y_%H%M%S")
    if (LDR_value >100):
        img0, rect = process_frame(mpimg.imread(current_image),svc)
        plt.imsave('/home/pi/Desktop/ML_%s.jpg' % current_time.strftime("%d%m%Y_%H%M%S"),img0,cmap="gray")
        CameraData = ((current_time.strftime("%d%m%Y_%H%M%S"), len(rect),'Day'),('','',''))
    else:
        img0, rect = process_frame(mpimg.imread(current_image),svcN)
        plt.imsave('/home/pi/Desktop/ML_%s.jpg' % current_time.strftime("%d%m%Y_%H%M%S"),img0,cmap="gray")
        CameraData = ((current_time.strftime("%d%m%Y_%H%M%S"), len(rect),'Night'),('','',''))
    
    value_range_body = {'majorDimension': 'ROWS','values': CameraData}
    request=service.spreadsheets().values().append(spreadsheetId=spreadsheetID, range=cell_range_insert, valueInputOption='USER_ENTERED', body=value_range_body)
    request.execute()
    
    
    # ****TEMPERATURE API CODE***
    
    if (current_time.timestamp()-weatherAPItime.timestamp()>300): #the delay for the tempreture api and the hackbridge tempreture sensor to record and upload to google sheets
        cell_range_insert='WeatherAPIData!A1'
        LondonTempC, LondonWeatherDescription=londonWeather()
        updateSheets(current_time, LondonTempC, LondonWeatherDescription, service, spreadsheetID, cell_range_insert)      
        weatherAPItime = datetime.datetime.now() 
        
        # ****HACKBRIDGE TEMPERATURE SENSOR CODE***
        humidity, temperature = Adafruit_DHT.read_retry(11, pin) # Read from sensor
        if humidity is not None or temperature is not None:
            cell_range_insert='HackbridgeTemperatureData!A1'
            Bodyinfo = ((current_time.strftime("%d%m%Y_%H%M%S"),'{0:0.1f}'.format(temperature),humidity),('','',''))
            value_range_body = {'majorDimension': 'ROWS','values': Bodyinfo}
            request=service.spreadsheets().values().append(spreadsheetId=spreadsheetID, range=cell_range_insert, valueInputOption='USER_ENTERED', body=value_range_body)
            request.execute()
        else:
            print ('Cannot read from device')

        cell_range_insert='TrafficData!A1'
        
    sleep(2)
