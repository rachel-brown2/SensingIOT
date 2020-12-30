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
pin = 4                 # Set tempreture sensor to pin to pin 4
tempsensdly = 2         # Temp Sensor is slow so needs a 2 second delay



#pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
#need to coppy across the london weather jason file onto the pi

# Get car and non-vehicle images
car_images = glob.glob(r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\Machine Learning\vehicles\**\*.png', recursive=True)
noncar_images = glob.glob(r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\Machine Learning\non-vehicles\**\*.png', recursive=True)

# Print lengths
print("Car images:")
print(len(car_images))
print()
print("Non-vehicle images:")
print(len(noncar_images))

# # Get sample images from training set
# car_img    = mpimg.imread(car_images[1])
# not_car_img = mpimg.imread(noncar_images[1])

# # Plot sample images from training set
# print("Car and not-car sample images from the training set:")
# fig = plt.figure()
# fig.add_subplot(2,2,1)
# plt.imshow(car_img.squeeze(), cmap="gray")
# fig.add_subplot(2,2,2)
# plt.imshow(not_car_img.squeeze(), cmap="gray")

# # Get Sample Images
# car_img = mpimg.imread(car_images[0])
# not_car_img = mpimg.imread(noncar_images[0])

# # Get HOG features from images
# _, car_dst = get_hog_features(car_img[:,:,2], 9, 8, 8, vis=True, feature_vec=True)
# _, not_car_dst = get_hog_features(not_car_img[:,:,2], 9, 8, 8, vis=True, feature_vec=True)

# # Display images next to their hog features 
# fig = plt.figure()
# fig.add_subplot(2,2,1)
# plt.imshow(car_img.squeeze(), cmap="gray")
# fig.add_subplot(2,2,2)
# plt.imshow(car_dst.squeeze(), cmap="gray")
# fig.add_subplot(2,2,3)
# plt.imshow(not_car_img.squeeze(), cmap="gray")
# fig.add_subplot(2,2,4)
# plt.imshow(not_car_dst.squeeze(), cmap="gray")

# Feature extraction parameters
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

# Get features for images with cars
print ('Get features for images with cars')
car_features = extract_features(car_images, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

# Get features for images without cars
print ('Get features for images without cars')

notcar_features = extract_features(noncar_images, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)  

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

# Print feature details
print()
print('Using',orient,'orientations with',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

print ('Running SVC')
# Create a linear SVC 
svc = LinearSVC()

# Train the SVC Classifer using the .fit() method
svc.fit(X_train, y_train)

# Check the accuracy of the SVC
print('Test Accuracy =', round(svc.score(X_test, y_test), 4))


################### Operational code starts here ###########################

camera = PiCamera()

timeout = time.time() + 60*60  #running for 60 mins

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

camera.start_preview()

weatherAPItime = datetime.datetime(1999,1,1,00,00,00)

while (time.time()<timeout):
    current_time = datetime.datetime.now()
    
    # ***** CAMERA CODE ****
    camera.capture('/home/pi/Desktop/ori_%s.jpg' % current_time.strftime("%d%m%Y_%H%M%S"))
    current_image = glob.glob('/home/pi/Desktop/ori_%s.jpg' % current_time.strftime("%d%m%Y_%H%M%S"))
    img0, rect = process_frame(mpimg.imread(current_image,svc))
    plt.imsave('/home/pi/Desktop/ML_%s.jpg' % current_time.strftime("%d%m%Y_%H%M%S"),img0,cmap="gray")

    Bodyinfo = ((current_time.strftime("%d%m%Y_%H%M%S"), rect),('',''))
    value_range_body = {'majorDimension': 'ROWS','values': Bodyinfo}
    request=service.spreadsheets().values().append(spreadsheetId=spreadsheetID, range=cell_range_insert, valueInputOption='USER_ENTERED', body=value_range_body)
    request.execute()
    
    
    # ****TEMPERATURE API CODE***
    
    if (current_time.timestamp()-weatherAPItime.timestamp()>300):
        cell_range_insert='WeatherAPIData!A1'
        LondonTempC, LondonWeatherDescription=londonWeather()
        # print(LondonTempC)
        updateSheets(current_time, LondonTempC, LondonWeatherDescription, service, spreadsheetID, cell_range_insert)      
        weatherAPItime = datetime.datetime.now() 
        
###### not really sure what to do with this #########
    # sleep(tempsensdly)# delay because tempreture sensor is slow
        humidity, temperature = Adafruit_DHT.read_retry(11, pin) # Read from sensor
    #####################################################
    
        if humidity is not None or temperature is not None:
            #print 'Temperature: ({0:0.1f}C) Humidity: {2:0.1f}%'.format( temperature, humidity)
            cell_range_insert='HackbridgeTemperatureData!A1'
            Bodyinfo = ((current_time.strftime("%d%m%Y_%H%M%S"),temperature('({0:0.1f}C)'),humidity('{2:0.1f}%')),('','',''))
            value_range_body = {'majorDimension': 'ROWS','values': Bodyinfo}
            request=service.spreadsheets().values().append(spreadsheetId=spreadsheetID, range=cell_range_insert, valueInputOption='USER_ENTERED', body=value_range_body)
            request.execute()
        else:
            print ('Cannot read from device')

        cell_range_insert='TrafficData!A1'

    
    
    sleep(2)


camera.stop_preview()




