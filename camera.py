from picamera import PiCamera
from time import sleep
import time

camera = PiCamera()

timeout = time.time() + 60*60*3  #running for 3 hours

i=1        

camera.start_preview()
camera.resolution = (1280, 720)
while (time.time()<timeout):
   camera.capture('/home/pi/Desktop/image%s.jpg' % i)
   i=i+1
   sleep(2)
camera.stop_preview()
