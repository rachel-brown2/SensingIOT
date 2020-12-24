# -*- cod--ing: utf-8 -*-
"""
Created on Thu Dec 24 16:36:15 2020

@author: rache
"""

from picamera import PiCamera
from time import sleep
import time

camera = PiCamera()

timeout = time.time() + 60  #running for 60 seconds

i=1        

camera.start_preview()
while (time.time()<timeout):
   camera.capture('/home/pi/Desktop/image%s.jpg' % i)
   i=i+1
   sleep(5)
camera.stop_preview()