# Import libraries
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# # # MAIN APPLICATION
img_grayscale = cv2.imread('102_su_000_0744.jpg', 0);
# Create sift features
surf = cv2.xfeatures2d.SURF_create();
# Detect the key points in the image
key_points = surf.detect(img_grayscale);
# Create array of zeros with the same shape and type as a given array
image_key_points = np.zeros_like(img_grayscale);
# Draw key points on the image
image_key_points = cv2.drawKeypoints(img_grayscale, key_points, image_key_points, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Plot Image
plt.imshow(image_key_points);
# Create feature discriptors
key_points, feature_descriptors = surf.compute(img_grayscale, key_points);
print(feature_descriptors.shape);