# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from sklearn import metrics
import glob
import os


# # # SVM Classification Parameters
kernels = [cv2.ml.SVM_LINEAR, cv2.ml.SVM_INTER, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_RBF];

# # # CREATE LISTS FOR OUR DATA

# Create list of integers that will be used
initial_list_of_labels = [];
# Since some images get removed with Haar Cascade, a new lable array is created
final_set_labels = [];
# Create labels for each image (which are integers from 0 - 6)
list_of_images = [];
# Create list of feature vectors
list_of_features = [];
# Get the maximum of features to extract
max_surf_features = 50;

# Create area of interest dimensions
hroi = 170;
wroi = 160;

def create_image_set(set_directory):
    # Use OS Walk to get the name of each sub directory
    for root, directories, files in os.walk(set_directory):
        # Loop through sub directorys in main directory to get their names
        for index, directory in enumerate(directories):
            # Create variable that tracks number of files in each directory
            FileNumberInDir = 0;
            # Print loop index
            # Create path name based on the current directory and the directories we are reading from
            path = "%s/%s"%(set_directory, directory);
            # Get all the files in that directory
            for file in glob.glob('%s/*'%path, recursive = True):
                # Create reference to each image
                list_of_images.append(file);
                # Increment file count
                FileNumberInDir += 1;
                # Create corresponding label array based on the directory index
                initial_list_of_labels.append(index);
            # Print all the sub-directories of our  training set
            print(directory + " ( " + str(index) + " ) : " + str(FileNumberInDir));
        
            
# # # CASCADE VARIABLES AND FUNCTION

# Load cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
# Cascade detection definitions
scale_reduction = 1.5;
min_accepted_neighbour_zones = 10;

def extract_feature(image_list_index, example, hog):
    # Locate the image from the list of images using the index
    grayscale_img = cv2.imread(list_of_images[image_list_index]);
    # Detect faces in image using Haar Cascade - function returns the definitions of the the detected rectangle in tuples
    faces = face_cascade.detectMultiScale(grayscale_img, scale_reduction, min_accepted_neighbour_zones);
    # Check if we found a face
    if len(faces) > 0:
        # For each detected face, generate the coordinates and dimensions of the face
        for x, y, height, width in faces:
            # Crop the detected face (area of interest) with approximate values for the faces
            area_of_interest = grayscale_img[y:(y+width), x:(x+height)];
            # Resize the area of interest to create uniform face sizes
            # Check if we want to plot the result of this function
            if example:
                # Return the final cropped version of the grayscale image
                plt.imshow(area_of_interest);
            # Extract hog features from the area of interest
            list_of_features.append(hog.compute(area_of_interest, (64, 64)));
            # Add label to the final label set from labels that were assigned at the beginning
            final_set_labels.append(initial_list_of_labels[image_list_index]);
    # Otherwise, don't extract the image's features if haar cascade failed
    else:
        print("Failed to find face! Removing file " + list_of_images[image_list_index] + " from data base.");

# # # CLASSIFICATION VARIABLES
        
def train_svm(training_features, training_labels):
    # Initiate the classifier object from OpenCV
    svm = cv2.ml.SVM_create();
    # Train the classifier using the training set
    svm.train(training_features, cv2.ml.ROW_SAMPLE, training_labels); 
    # Return the trained classifier     
    return svm;

def score_svm(svm, testing_features, testing_labels):
    # Make prediction based on the testing features
    _, predicted_labels = svm.predict(testing_features);
    # Score the classifier based on its accuracy
    return metrics.accuracy_score(testing_labels, predicted_labels);

# # # MAIN APPLICATION
        
# Load our images into open cv by creating references to each image
create_image_set("Training_Set_Large");
# Create specifications for HOG
win_size = (48, 96);
block_size = (16, 16);
block_stride = (8, 8);
cell_size = (8, 8)
num_bins = 9
# Create the HOG descriptor
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins);
print("Number of pictures: " + str(len(list_of_images)) + " Number of Labels: " + str(len(initial_list_of_labels)));
# Iterate through the data set of images
for x in range(0, len(initial_list_of_labels)):
    # Create example condition
    example = False;
    # Check if this is the first iteration of the function
    if x == 0:
        # We want to visualise the process with the first image
        example = True;
    # Extract featuers o feach image
    aoi = extract_feature(x, example, hog);
# Check how many images are left after the elimination process
print("Number of features: " + str(len(list_of_features)) + " Number of Labels: " + str(len(final_set_labels)));
# Convert lists into data types that are compatible with OpenCV
final_set_labels = np.int32(final_set_labels);
list_of_features = np.array(list_of_features, dtype = np.float32);
# Print information about the features
print(list_of_features.shape, final_set_labels.shape);
# Using sklearn function, divide the data into testing and training sets
training_features, testing_features, training_labels, testing_lables = ms.train_test_split(list_of_features, final_set_labels, test_size=0.2, random_state = 42);
# Create the svm
my_svm = train_svm(training_features, training_labels);
# Now that we have trained the SVM, make sure it can classify the training set
print(score_svm(my_svm, training_features, training_labels));
# See how it performs on the test set
print(score_svm(my_svm, testing_features, testing_lables));