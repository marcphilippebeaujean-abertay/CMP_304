import glob
import os
import numpy as np
import cv2
import pandas as pd

# # # CREATE LISTS FOR OUR DATA

# Labels are from 0 - 5, corresponding to the index of the imagess
list_of_labels = []
# Store pre-process iamges for fisherface anlysis
list_of_images = []
# Create list of feature vectors
list_of_features = []
# Create list to assign each label value to a string
emotion_classes = ['Happy', 'Sad', 'Fear', 'Angry', 'Surprised', 'Disgust']

# Create area of interest dimensions
hroi = 100
wroi = 100

rand_seed = 40

# Define to what decimal place the final data needs to be rounded
decimal_place = 2

# # # CREATE DEPENDANCIES FOR HOG FEATURE EXTRACTION

# Create specifications for HOG
win_size = (48, 96)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9
# Create the HOG descriptor
hog = cv2.HOGDescriptor(win_size, block_size,
                        block_stride, cell_size, num_bins)

# # # CASCADE VARIABLES AND FUNCTION

# Load cascades
face_cascade_1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade_2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade_3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_cascade_4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
# Cascade detection definitions
scale_reduction = 1.1
min_accepted_neighbour_zones = 10


def extract_aoi_features(detected_faces, grayscale_img, label):
    # For each detected face, generate the coordinates and dimensions of the face
    for x, y, height, width in detected_faces:
        # Crop the detected face (area of interest)
        area_of_interest = grayscale_img[y:(y+width), x:(x+height)]
        try:
            # Try to resize the area of interest to create uniform face sizes
            resized_aoi = cv2.resize(area_of_interest, (wroi, hroi))
            # Extract hog features from the area of interest
            list_of_features.append(hog.compute(resized_aoi, (64, 64)))
            # Create reference to the image
            list_of_images.append(resized_aoi)
            # Add index of directory as a label
            list_of_labels.append(label)
        except:
            # Don't do anything, if resizing fails - skip the feature
            pass


def create_data_set(set_directory):
    # Print notifier
    print("Pre-processing data and extracting features...!")
    # Create debugging tracker for cascades
    haar_failed = 0
    # Use OS Walk to get the name of each sub directory
    for root, directories, files in os.walk(set_directory):
        # Loop through sub directorys in main directory to get their names
        for directory_index, directory in enumerate(directories):
            # Create variable that tracks number of files in each directory
            FileNumberInDir = 0
            # Create path name based on the current directory and the directories we are reading from
            path = "{}/{}".format(set_directory, directory)
            # Loop through all the files in that directory
            for file in glob.glob('%s/*' % path, recursive=True):
                # Increment file count
                FileNumberInDir += 1
                # Locate the image from the list and read it using OpenCV
                grayscale_img = cv2.imread(file, 0)
                # Apply histogram equalisation to expose features
                grayscale_img = cv2.equalizeHist(grayscale_img)
                # Detect faces in image using Haar Cascade - function returns the definitions of the the detected rectangle in tuples
                faces1 = face_cascade_1.detectMultiScale(
                    grayscale_img, scale_reduction, min_accepted_neighbour_zones, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
                # Check if we found a face, then use it to extract the corresponding features
                if len(faces1) == 1:
                    extract_aoi_features(
                        faces1, grayscale_img, directory_index)
                else:
                    faces2 = face_cascade_2.detectMultiScale(
                        grayscale_img, scale_reduction, min_accepted_neighbour_zones, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
                    if len(faces2) == 1:
                        extract_aoi_features(
                            faces2, grayscale_img, directory_index)
                    else:
                        faces3 = face_cascade_3.detectMultiScale(
                            grayscale_img, scale_reduction, min_accepted_neighbour_zones, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
                        if len(faces3) == 1:
                            extract_aoi_features(
                                faces3, grayscale_img, directory_index)
                        else:
                            faces4 = face_cascade_4.detectMultiScale(
                                grayscale_img, scale_reduction, min_accepted_neighbour_zones, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
                            if len(faces4) == 1:
                                extract_aoi_features(
                                    faces4, grayscale_img, directory_index)
                            else:
                                # We failed to find a face!
                                haar_failed += 1
            # Print all the sub-directories of our  training set
            print(directory + " ( " + str(directory_index) + " ) : " +
                  str(FileNumberInDir - haar_failed))
    # Print how many haar cascades we failed to find
    print("Failed Haar Cascades: " + str(haar_failed))

# Load our images into open cv by creating references to each image
create_data_set("Training_Set_Final")
# Convert lists into data types that are compatible with OpenCV
list_of_labels = np.int32(list_of_labels)
# Convert list of features to numpy array with float precision
list_of_features = np.array(list_of_features, dtype=np.float32)
# Hog gives us the features in shape 1038, 1980, 1 - we can just reshape it into a 2D array
list_of_features = list_of_features.reshape(1038, 1980)
print(list_of_features.shape)
# Print the extracted matrices
print("Total Data: " + str(len(list_of_images)) + " Images")
# Save list of features as csv
df = pd.DataFrame(list_of_features)
df.to_csv("image_hog_features")