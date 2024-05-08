# File: crop.py
# Description: Script for cropping images in a rectangular shape
# Author: Shivesh Prakash

from PIL import Image

def crop_images(folder_path, xstart, xend, ystart, yend):
    """
    Crop images in a folder to a specified rectangular shape.

    Args:
        folder_path (str): Path to the folder containing the images.
        xstart (int): Starting x-coordinate of the crop.
        xend (int): Ending x-coordinate of the crop.
        ystart (int): Starting y-coordinate of the crop.
        yend (int): Ending y-coordinate of the crop.
    """
    image_names = ['amber', 'blue', 'cyan', 'deep_red', 'far_red', 'green', 'lime', 'red', 'red_orange', 'royal_blue', 'violet']
    
    for name in image_names:
        # Open the image
        image_path = folder_path + '/' + name + '_10000_2.0.png'
        img = Image.open(image_path)
        
        # Crop the image
        cropped_img = img.crop((xstart, ystart, xend, yend))
        
        # Save the cropped image with the same name
        cropped_img.save(folder_path + '/' + name + '.png')

# Example usage:
folder_path = "/Users/material/data_collection_out/yellow_paper/exp5"
xstart = 60
xend = 1505
ystart = 1
yend = 864

crop_images(folder_path, xstart, xend, ystart, yend)
