# File: trying_exposures.py
# Description: Script for testing different exposure times and gains with Picamera2
# To be run on Raspberry Pi
# Author: Shivesh Prakash

from picamera2 import Picamera2
import timeit
import time

# Initialize Picamera2 instance
picam2 = Picamera2()

# Configure capture settings
capture_config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(capture_config)

# Starting exposure time in milliseconds
exp = 5000

# Maximum exposure time to try
max_exp = 60000

# Maximum gain to try
max_gain = 6.0

# Loop through different exposure times
while exp <= max_exp:
    # Starting analogue gain, it theoretically goes from 1.0 to 16.0
    # on my machine the miminum possible value was 1.123
    g = 1.123
    # Loop through different gains
    while g <= max_gain:
        # Configure camera controls
        # except analogue gain and exposure time, all other filters, focus, white-balance are disabled
        picam2.set_controls(
            {
                "AeEnable": False,
                "ExposureTime": exp,
                "AwbEnable": False,
                "AnalogueGain": g,
                "NoiseReductionMode": 0,
                "Saturation": 1.0,
                "Contrast": 1.0,
                "Brightness": 0.0,
            }
        )
        picam2.start()

        # Capture image and save with exposure time and gain as filename
        name = str(exp) + "_" + str(g)
        picam2.capture_file(name + ".jpg")
        picam2.capture_file(name + ".dng", "raw")

        # Increment gain
        g += 1.0
    # Increment exposure time
    exp += 5000
