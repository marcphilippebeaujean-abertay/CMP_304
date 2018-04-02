# Import libraries
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

    
# # # CASCADE VARIABLES AND FUNCTION

# Load cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
# Cascade detection definitions
scale_reduction = 1.5;
min_accepted_neighbour_zones = 10;

def haar_cascade_detection(grayscale_img, example):
    # Detect faces in rectangles- function returns the definitions of the the detected rectangle in tuples
    faces = face_cascade.detectMultiScale(grayscale_img, scale_reduction, min_accepted_neighbour_zones);
    # For each detected face, generate the coordinates
    for (x, y, width, height) in faces:
        # Create subset of the grayscale, that contains only the detected face
        area_of_interest = grayscale_img[y:y+height, x:x+width];
        # area_of_interest = cv2.rectangle(grayscale_img, (x, y), (x+width, y+height), (255, 0, 0), 2)
    # Check if we want to plot the result of this function
    if example:
        # Return the final cropped version of the grayscale image
        plt.imshow(area_of_interest);
    return area_of_interest;

# # # GETTING THE FEATURE VECTOR OF AN IMAGE

def get_SURF_feature_vector(input_img, example):     
    # Apply the haar cascade to the grayscale
    img_grayscale = haar_cascade_detection(input_img, example);
    # Create sift features
    surf = cv2.xfeatures2d.SURF_create();
    # Detect the key points in the image
    key_points = surf.detect(img_grayscale);
    # Create array of zeros with the same shape and type as a given array
    image_key_points = np.zeros_like(img_grayscale);
    # Draw key points on the image
    image_key_points = cv2.drawKeypoints(img_grayscale, key_points, image_key_points, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Create feature discriptors
    key_points, feature_descriptors = surf.compute(img_grayscale, key_points);
    # Check if we want to print out the results
    if example:
        # Plot Image and descriptors
        plt.imshow(image_key_points);
        print(feature_descriptors.shape);
    

# # # MAIN APPLICATION
    

get_SURF_feature_vector(cv2.imread('006_an_000_0024.jpg', 0), True);