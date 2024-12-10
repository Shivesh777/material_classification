# Material Classification Project

This project is designed to process images, extract features, and train basic machine learning models for the material classification tasks. Below is an overview of the files in this project and their functionalities:

## Files Overview:

1. **process_raw.py**: Processes raw DNG images to generate linear RGB images. It includes functions for reading DNG files, linearizing raw data, performing bilinear demosaicing, and saving processed images.

2. **crop.py**: Crops images in a rectangular shape. It includes a function to crop images based on specified coordinates.

3. **extract_and_analyze.py**: Extracts patches from images, normalizes them, extracts statistical features, and trains basic machine learning models. It includes functions for feature extraction, statistical feature calculation, model training, and evaluation.

## How to Use:

1. **Processing Raw DNGs**:
   - Place your raw DNG images in a folder.
   - Run `process_raw.py` script.
   - Provide the path to the folder containing raw DNG images.
   - Processed images will be saved in the same folder.

2. **Cropping Images**:
   - Ensure you have images in the specified folder.
   - Run `crop_images.py` script.
   - Provide the folder path and coordinates for cropping.
   - Cropped images will be saved with the same names in the same folder.

3. **Extracting Patches and Training Models**:
   - Organize your image dataset into folders according to different classes.
   - Run `extract_and_train.py` script.
   - Ensure the script has access to the image folders.
   - Models will be trained and evaluated, and the confusion matrix will be displayed.
