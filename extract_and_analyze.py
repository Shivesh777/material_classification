# File: extract_and_analyze.py
# Description: Script for extracting patches, normalizing, extracting statistical features, and training basic ML models
# Author: Shivesh Prakash

import cv2
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def extract_patches(image, patch_size=(20, 20), overlap=(5, 5)):
    """
    Extract patches from an image with specified size and overlap.

    Args:
        image (np.ndarray): Input image.
        patch_size (tuple, optional): Size of the patch (height, width). Defaults to (20, 20).
        overlap (tuple, optional): Overlap between patches (vertical, horizontal). Defaults to (5, 5).

    Returns:
        list: List of extracted patches.
    """
    patches = []
    height, width = image.shape[:2]

    # Iterate over the image with the specified overlap
    for y in range(0, height - patch_size[0] + 1, patch_size[0] - overlap[0]):
        for x in range(0, width - patch_size[1] + 1, patch_size[1] - overlap[1]):
            patch = image[y:y+patch_size[0], x:x+patch_size[1]]
            patches.append(patch)

    return patches

def extract(folder_path, t=0, x=0, y=0, norm=1):
    """
    Extract patches, normalize, and extract statistical features from images in a folder.

    Args:
        folder_path (str): Path to the folder containing the images.
        t (int, optional): Target label for the images. Defaults to 0.
        x (list, optional): List to store feature vectors. Defaults to 0.
        y (list, optional): List to store target labels. Defaults to 0.
        norm (int, optional): Normalize flag. Defaults to 1.

    Returns:
        list, list: Feature vectors, target labels.
    """
    colours = ['amber', 'blue', 'cyan', 'deep_red', 'far_red', 'green', 'lime', 'red', 'red_orange', 'royal_blue', 'violet']
    r_patches = []
    g_patches = []
    b_patches = []

    if not x or not y:
        x = []
        y = []
    
    for color in colours:
        # Read the color image
        image_path = folder_path + '/' + color + '.png'
        image = cv2.imread(image_path)

        b, g, r = cv2.split(image)

        r_patch = extract_patches(r)
        g_patch = extract_patches(g)
        b_patch = extract_patches(b)

        num_patches = len(r_patch)

        r_patches.append(r_patch)
        g_patches.append(g_patch)
        b_patches.append(b_patch)

    r_patches = np.array(r_patches)
    g_patches = np.array(g_patches)
    b_patches = np.array(b_patches)

    if norm == 1:
        r_patches = r_patches / np.max(r_patches)
        g_patches = g_patches / np.max(g_patches)
        b_patches = b_patches / np.max(b_patches)
        pass

    for i in range(num_patches):
        r = []
        g = []
        b = []
        for j in  range(len(r_patches)):
            r.append(r_patches[j][i])
            g.append(g_patches[j][i])
            b.append(b_patches[j][i])
        r = np.array(r)
        g = np.array(g)
        b = np.array(b)
        stats = calculate_statistics(r, g, b)
        x.append(stats)
        y.append(t)
        
    return x, y

def calculate_statistics(r_patches, g_patches, b_patches):
    """
    Calculate statistical features from RGB patches.

    Args:
        r_patches (np.ndarray): Red channel patches.
        g_patches (np.ndarray): Green channel patches.
        b_patches (np.ndarray): Blue channel patches.

    Returns:
        list: List of statistical features.
    """
    # Initialize a list to store statistics for each patch
    patch_statistics = []

    # Calculate statistics for red channel
    mean_values = np.mean(r_patches)
    std_dev_values = np.std(r_patches)
    percentile_25 = np.percentile(r_patches, 25)
    percentile_75 = np.percentile(r_patches, 75)
    rms_values = np.sqrt(np.mean(np.square(r_patches)))
    coefficient_of_variation = std_dev_values / mean_values

    patch_statistics.extend([mean_values, std_dev_values, percentile_25, percentile_75, rms_values, coefficient_of_variation])

    # Calculate statistics for green channel
    mean_values = np.mean(g_patches)
    std_dev_values = np.std(g_patches)
    percentile_25 = np.percentile(g_patches, 25)
    percentile_75 = np.percentile(g_patches, 75)
    rms_values = np.sqrt(np.mean(np.square(g_patches)))
    coefficient_of_variation = std_dev_values / mean_values

    patch_statistics.extend([mean_values, std_dev_values, percentile_25, percentile_75, rms_values, coefficient_of_variation])

    # Calculate statistics for blue channel
    mean_values = np.mean(b_patches)
    std_dev_values = np.std(b_patches)
    percentile_25 = np.percentile(b_patches, 25)
    percentile_75 = np.percentile(b_patches, 75)
    rms_values = np.sqrt(np.mean(np.square(b_patches)))
    coefficient_of_variation = std_dev_values / mean_values

    patch_statistics.extend([mean_values, std_dev_values, percentile_25, percentile_75, rms_values, coefficient_of_variation])

    return patch_statistics

def svm_linear(X_train, y_train, X_test, y_test):
    """
    Train and evaluate SVM with linear kernel.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        np.ndarray: Predicted labels.
    """
    # Initialize and train SVM model
    svm_model = SVC(kernel='linear', C=100)
    svm_model.fit(X_train, y_train)

    # Predict on validation set
    svm_val_predictions = svm_model.predict(X_test)

    rounded_predictions = np.round(svm_val_predictions).astype(int)

    # Calculate accuracy
    svm_val_accuracy = accuracy_score(y_test, rounded_predictions)
    print("SVM Validation Accuracy:", svm_val_accuracy)

    return rounded_predictions

# Similar functions for other classifiers (svm_rbf, logistic_reg, random_forest, xg_boost) omitted for brevity

def conf_matrix(rounded_predictions, y_test):
    """
    Display confusion matrix.

    Args:
        rounded_predictions (np.ndarray): Predicted labels.
        y_test (np.ndarray): True labels.
    """
    cm = confusion_matrix(y_test, rounded_predictions, labels=[0,1,2,3,4], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
    print(cm)
    print(classification_report(y_test, rounded_predictions, labels=[0,1,2,3,4]))
    disp.plot()
    plt.show()

def svm_linear_grid_search(X_train, y_train):
    """
    Perform grid search for SVM with linear kernel.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
    """
    # Define parameter grid
    C_range = np.logspace(0, 2, 10)
    param_grid = {'C': C_range}

    # Define cross-validation strategy
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    # Perform grid search
    grid = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=cv)
    grid.fit(X_train, y_train)

    # Print best parameters and score
    print("The best parameter C is %0.2f with a score of %0.2f" % (grid.best_params_['C'], grid.best_score_))

def xg_boost_grid_search(X_train, y_train):
    """
    Perform grid search for xgboost.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
    """
    # Create the XGBoost Classifier
    xgb_model = XGBClassifier()

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    # Perform Grid Search
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Print the best parameters found
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

# Usage:

# for all items
items = ['blue_plastic', 'white_plastic', 'brown_board', 'white_board', 'green_ceramic', 'white_ceramic', 'green_towel', 'white_towel', 'yellow_paper', 'white_paper']

# for coloured items only
# items = ['blue_plastic', 'brown_board', 'green_ceramic', 'green_towel', 'yellow_paper']

# for white items only
# items = ['white_plastic', 'white_board', 'white_ceramic', 'white_towel', 'white_paper']


experiments = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']

X_train = []
y_train = []

X_test = []
y_test = []

val_ex = 'exp5'

experiments.remove(val_ex)

for item in items:
    if 'plastic' in item:
        t = 0
    if 'board' in item:
        t = 1
    if 'ceramic' in item:
        t = 2
    if 'towel' in item:
        t = 3
    if 'paper' in item:
        t = 4
    for exp in experiments:
        folder_path = '/Users/material/data_collection_out/' + item + '/' + exp
        X_train, y_train = extract(folder_path, t, X_train, y_train)
    folder_path = '/Users/material/data_collection_out/' + item + '/' + val_ex
    X_test, y_test = extract(folder_path, t, X_test, y_test)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Choose one of the models, let's say svm linear
# If it is tunable, use grid search to find the best parameters
svm_linear_grid_search(X_train, y_train)

# train and evalaute the model
rounded_predictions = svm_linear(X_train, y_train, X_test, y_test)

# draw the confusion matrix
conf_matrix(rounded_predictions, y_test)
