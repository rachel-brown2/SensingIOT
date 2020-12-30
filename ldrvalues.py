# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:47:26 2020

@author: rache
"""

###need to figure out how to do a low pass filter####

import time

# Import SPI library (for hardware SPI) and MCP3008 library.
import Adafruit_GPIO.SPI as SPI
import Adafruit_MCP3008

SPI_PORT   = 0 # the port the LDR is connected to 
SPI_DEVICE = 0
mcp = Adafruit_MCP3008.MCP3008(spi=SPI.SpiDev(SPI_PORT, SPI_DEVICE))
LDR_value = mcp.read_adc(0)

while True:
    LDR_value[0] = mcp.read_adc(0)
    if (LDR_value >1200):
        print('| {0:>4} |'.format(LDR_value))
    else:
        print('too small wahhh')
    time.sleep(0.5)
