# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 13:56:09 2020

@author: rache
"""

import cv2
import glob
import numpy as np


car_images = glob.glob(r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\all_together\car_at_night - Copy\**\*.png', recursive=True)

for x in range(len(car_images)):
    img = cv2.imread(car_images[x])
   
    result1 = cv2.resize(img,(64, 64))
    # save result
    cv2.imwrite(car_images[x] + '.png', result1)

noncar_images = glob.glob(r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\all_together\not_car_at_night\**\*.png', recursive=True)

Hackbridgecar_images = glob.glob(r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\noncarimageshackbridge - Copy\**\*.jpg', recursive=True)

test_images = glob.glob(r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\test_images\*.jpg', recursive=True)
test_images1 = glob.glob(r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\Machine Learning\test_images\*.jpg', recursive=True)

emptystreet = glob.glob(r'C:\Users\rache\OneDrive - Imperial College London\DE4 Uni Work\Sensing and IOT\Machine Learning\emptystreet2\*.png', recursive=True)



    

# # read image
# for x in range(len(emptystreet)):
#     img = cv2.imread(emptystreet[x])
#     ht, wd, cc= img.shape
    
#     # create new image of desired size and color (blue) for padding
#     ww = 150
#     hh = 150
#     color = (0,0,0)
#     result = np.full((hh,ww,cc), color, dtype=np.uint8)
    
#     # compute center offset
#     xx = (ww - wd) // 2
#     yy = (hh - ht) // 2
    
#     # copy img image into center of result image
#     result[yy:yy+ht, xx:xx+wd] = img
    
#     # # view result
#     # cv2.imshow("result", result)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     result1 = cv2.resize(result,(64,64))
#     # save result
#     cv2.imwrite(emptystreet[x], result1)
    
    # read image
