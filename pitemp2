##
 # Maker's Digest
 # DHTxx Temperature and Humidity Example.
 #
 # Dont forget to install Adafruit_DHT! See README.md for details.
##
import sys
from time import sleep  # Import Sleep module from time library
import Adafruit_DHT     # Import the Adafruit_DHT module

pin = 4                 # Set pin to pin 4
dly = 2                 # Set delay to 2000ms (2 seconds) Can be changed to 1 for DHT22
sensor_type = 22        # Sensor type: Change this to 22 if using DHT22, or leave
                        # at 11 for DHT11


try: 
    while True:
        # Introduce our delay
        sleep(dly)
        
        # Read from sensor
        humidity, temperature = Adafruit_DHT.read_retry(sensor_type, pin)

        # Sensor defaults to Celsius, convert to Fahrenheit. 
        fahrenheit = (temperature * 1.8) + 32
        celsius = temperature

        # Check if data from sensor read is valid. Print data if it is. Else, fail.
        if humidity is not None or temperature is not None:
            print 'Temperature: ({0:0.1f}C) ({1:0.1f}F) Humidity: {2:0.1f}%'.format(celsius, fahrenheit, humidity)
        else:
            print 'Cannot read from device'

except KeyboardInterrupt:
    sys.exit()
