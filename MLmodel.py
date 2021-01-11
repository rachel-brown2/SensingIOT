# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:10:04 2021

@author: rache
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:32:36 2020

@author: Rachel Brown
"""

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
import glob
from utils import (extract_features)
import pickle


# Get car and non-vehicle images
car_images = glob.glob(r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\Machine Learning\vehicles\vehicles\**\*.png', recursive=True)
noncar_images = glob.glob(r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\Machine Learning\non-vehicles\**\*.png', recursive=True)


# Feature extraction parameters
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

# Extract the hog features for images with cars
car_features = extract_features(car_images, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

# Extract the hog features for images without cars
notcar_features = extract_features(noncar_images, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)  

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=rand_state)

# Create a linear SVC 
svc = LinearSVC()

# Train the SVC Classifer using the .fit() method
svc.fit(X_train, y_train)

# Check the accuracy of the SVC
print('Test Accuracy =', round(svc.score(X_test, y_test), 4))

with open('Model.pkl','wb') as f:
    pickle.dump(svc,f)