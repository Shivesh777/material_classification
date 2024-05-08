# File: process_raw.py
# Description: Script for processing raw DNG images to generate linear RGB images
# Authors: Shivesh Prakash, Dhruv Verma

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import rawpy
from skimage import exposure
from scipy.signal import convolve2d
import imageio


def normalize_image(image):
    """
    Normalize the intensity values of an image to the range [0, 1].

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Normalized image.
    """
    normalized_image = exposure.rescale_intensity(
        image, in_range="image", out_range=(0, 1)
    )
    return normalized_image


def readDNG(dng_path):
    """
    Read a Digital Negative (DNG) file and return raw data.

    Args:
        dng_path (str): Path to the DNG file.

    Returns:
        rawpy._rawpy.RawPy: rawpy object.
        np.ndarray: Raw image data.
    """
    dng = rawpy.imread(dng_path)
    raw_image = dng.raw_image.astype(int)
    return dng, raw_image


def printImage(image, dpi=200):
    """
    Display the image using matplotlib.

    Args:
        image (np.ndarray): Image to display.
        dpi (int, optional): Dots per inch in the output figure. Defaults to 200.
    """
    figure(dpi=dpi)
    plt.imshow(image, cmap="gray")
    plt.show()


def saveImage(image, dpi=200, path="output"):
    """
    Save the image as a plot using matplotlib.

    Args:
        image (np.ndarray): Image to save.
        dpi (int, optional): Dots per inch in the output figure. Defaults to 200.
        path (str, optional): File path to save the image. Defaults to "output".
    """
    plt.figure(dpi=dpi)
    plt.imshow(image, cmap="gray")
    plt.savefig(path)
    plt.close()


# Linearize image
def linearize(raw, dng):
    """
    Linearize the raw image data.

    Args:
        raw (np.ndarray): Raw image data.
        dng (rawpy._rawpy.RawPy): rawpy object.

    Returns:
        np.ndarray: Linearized image.
    """
    black = dng.black_level_per_channel[0]
    saturation = dng.white_level
    raw -= black
    uint10_max = 2**10 - 1
    raw = raw * int(uint10_max / (saturation - black))
    raw = np.clip(raw, 0, uint10_max)
    return raw


def bilinear_demosaic(dng, raw):
    """
    Perform bilinear demosaicing on the raw image data.

    Args:
        dng (rawpy._rawpy.RawPy): rawpy object.
        raw (np.ndarray): Raw image data.

    Returns:
        np.ndarray: Demosaiced RGB image.
    """
    bayer = dng.raw_colors
    assert dng.color_desc == b"RGBG"
    raw = normalize_image(raw)

    k1 = np.array([[0.5, 0, 0.5]])
    k2 = k1.T
    k3 = np.array([[0.25, 0, 0.25], [0, 0, 0], [0.25, 0, 0.25]])
    k4 = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])

    # Red
    red_idx = np.where(bayer == 0)
    red = np.zeros(raw.shape)
    red[red_idx] = raw[red_idx]
    f1 = convolve2d(red, k1, mode="same")
    f2 = convolve2d(red, k2, mode="same")
    f3 = convolve2d(red, k3, mode="same")
    fill = f1 + f2 + f3
    red += fill

    # Green
    green_idx = np.where((bayer == 1) | (bayer == 3))
    green = np.zeros(raw.shape)
    green[green_idx] = raw[green_idx]
    fill = convolve2d(green, k4, mode="same")
    green += fill

    # Blue
    blue_idx = np.where(bayer == 2)
    blue = np.zeros(raw.shape)
    blue[blue_idx] = raw[blue_idx]
    f1 = convolve2d(blue, k1, mode="same")
    f2 = convolve2d(blue, k2, mode="same")
    f3 = convolve2d(blue, k3, mode="same")
    fill = f1 + f2 + f3
    blue += fill

    rgb = np.stack([red, green, blue], axis=2)

    return rgb


def readAndProcess(path_to_raw, display=False):
    """
    Read and process a DNG file.

    Args:
        path_to_raw (str): Path to the DNG file.
        display (bool, optional): Whether to display the processed image. Defaults to True.

    Returns:
        np.ndarray: Raw image data.
        np.ndarray: Processed RGB image data.
    """
    dng, raw = readDNG(path_to_raw)
    raw = linearize(raw, dng)

    lin_rgb = bilinear_demosaic(dng, raw)

    if display:
        printImage(lin_rgb)

    return raw, lin_rgb


def process_data():
    items = [
        "blue_plastic",
        "white_plastic",
        "brown_board",
        "white_board",
        "green_ceramic",
        "white_ceramic",
        "green_towel",
        "white_towel",
        "yellow_paper",
        "white_paper",
    ]
    experiments = ["exp1", "exp2", "exp3", "exp4", "exp5"]
    colours = [
        "amber",
        "blue",
        "cyan",
        "deep_red",
        "far_red",
        "green",
        "lime",
        "red",
        "red_orange",
        "royal_blue",
        "violet",
    ]
    for item in items:
        for exp in experiments:
            for col in colours:
                dng_name = item + "/" + exp + "/" + col + "_10000_2.0.dng"
                png_name = item + "/" + exp + "/" + col + "_10000_2.0.png"
                path = "/Users/material/data_collection/" + dng_name
                raw, lin_rbg = readAndProcess(path)
                rgb_array = np.uint8(lin_rbg * 255)

                imageio.imwrite(
                    "/Users/material/data_collection_out/" + png_name, rgb_array
                )


# Usage:
process_data()
