# Import libraries
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# # # INITIALISE VARIABLES # # #


# # # CREATING A k-NEAREST NEIGHBOUR CLASSIFIERs

# Generate a random numpy seed

def SIFT_FeatureExtraction(grayscale_image):
    

    return grayscale_image;

# # # MAIN APPLICATION
img_grayscale = cv2.imread('102_su_000_0744.jpg', 0);
corners = cv2.cornerHarris(img_grayscale, 3, 15, 0.04)
plt.imshow(corners, cmap='gray');