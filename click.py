# File: click.py
# Description: Script for capturing raw images in all colors using Picamera2 and ESP32 board
# To be run on Raspberry Pi
# Author: Shivesh Prakash

from picamera2 import Picamera2
import timeit
import time
import serial

# defining all the colours
colour = [
    "royal_blue",
    "blue",
    "cyan",
    "green",
    "lime",
    "amber",
    "red_orange",
    "red",
    "deep_red",
    "far_red",
    "violet",
]

picam2 = Picamera2()
capture_config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(capture_config)

# serial used to communicate with the ESP32 board (ChromaFlash)
# the serial port may be /dev/ttyACM1 check the raspberry pi to see which new serial port is now available
ser = serial.Serial(port="/dev/ttyACM0")
# sleep might be needed here if the camera taked time to start up
# time.sleep(0.2)

# set the values for chosen exp time and gain
exp = 10000
g = 2.0

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

for i in range(11):
    # sometimes the first and second images were slow so more wait time might be needed here sometimes
    # if i == 1:
    # time.sleep(0.1)
    # this writes the index of image colour being clicked so the corresponding LED lights up through the ESP32 board
    ser.write(str(i).encode())
    name = colour[i] + "_" + str(exp) + "_" + str(g)
    # I use time and wait here so the click happens sometime after the led stops and waits enough for the next LED to light up
    start = timeit.default_timer()
    time.sleep(0.3)
    picam2.capture_file(name + ".jpg")
    picam2.capture_file(name + ".dng", "raw")
    print("time taken: ", timeit.default_timer() - start)

    # Most images were clicked in about 1.2 sec so I set the LED to be on for 2 sec and so we wait here until 2 sec have passed since the LED lit up
    t = 2 - timeit.default_timer() + start
    if t > 0:
        time.sleep(t)

    print("time now: ", timeit.default_timer() - start)

ser.close()
