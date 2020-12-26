##
 # Maker's Digest
 #
 # MCP3008 ADC Example
 #
 # Dont forget to install the libraries! See README.md for details.
##
import sys                      # Import sys module
from time import sleep          # Import sleep from time
import Adafruit_GPIO.SPI as SPI # Import Adafruit GPIO_SPI Module
import Adafruit_MCP3008         # Import Adafruit_MCP3008

# We can either use Software SPI or Hardware SPI. For software SPI we will
# use regular GPIO pins. Hardware SPI uses the SPI pins on the Raspberry PI
# Set the following variable to either HW or SW for Hardware SPI and Software
# SPI respectivly.
SPI_TYPE = 'HW'
dly = .5         # Delay of 1000ms (1 second)

# Software SPI Configuration
CLK     = 18    # Set the Serial Clock pin
MISO    = 23    # Set the Master Input/Slave Output pin
MOSI    = 24    # Set the Master Output/Slave Input pin
CS      = 25    # Set the Slave Select

# Hardware SPI Configuration
HW_SPI_PORT = 0 # Set the SPI Port. Raspi has two.
HW_SPI_DEV  = 0 # Set the SPI Device

# Instantiate the mcp class from Adafruit_MCP3008 module and set it to 'mcp'. 
if (SPI_TYPE == 'HW'):
    # Use this for Hardware SPI
    mcp = Adafruit_MCP3008.MCP3008(spi=SPI.SpiDev(HW_SPI_PORT, HW_SPI_DEV))
elif (SPI_TYPE == 'SW'):
    # Use this for Software SPI
    mcp = Adafruit_MCP3008.MCP3008(clk = CLK, cs = CS, miso = MISO, mosi = MOSI)

# Check to see if we have input from command line. Bail if we dont.
if ( len(sys.argv) <= 1):
    print "Usage: MCP3008-example.py <Analog Port>"
    sys.exit(1)
else:
    analogPort = int(sys.argv[1])

print 'Reading MCP3008 values on pin: %d' % analogPort

try:
    while True:
        # Read the value from the MCP3008 on the pin we specified in analogPort
        val = mcp.read_adc(analogPort)

        # print out the value
        print val

        # Sleep for dly
        sleep(dly)
except KeyboardInterrupt:
    sys.exit()
