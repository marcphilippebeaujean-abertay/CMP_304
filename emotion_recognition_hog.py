# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from sklearn import metrics
import xlsxwriter
import glob
import os
import itertools


# # # SVM Classification Parameters
kernels = [cv2.ml.SVM_LINEAR, cv2.ml.SVM_INTER, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_RBF];

# # # CREATE LISTS FOR OUR DATA

# Since some images get removed with Haar Cascade, a new lable array is created
list_of_labels = [];
# Create labels for each image (which are integers from 0 - 6)
list_of_images = [];
# Create list of feature vectors
list_of_features = [];
# Create list to assign each label value to a string
emotion_classes = ['Happy', 'Sad', 'Fear', 'Angry', 'Surprised', 'Disgust'];

# Create area of interest dimensions
hroi = 100;
wroi = 100;


# # # CREATE DEPENDANCIES FOR HOG FEATURE EXTRACTION

# Create specifications for HOG
win_size = (48, 96);
block_size = (16, 16);
block_stride = (8, 8);
cell_size = (8, 8)
num_bins = 9
# Create the HOG descriptor
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins);

# # # CASCADE VARIABLES AND FUNCTION

# Load cascades
face_cascade_1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
face_cascade_3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml");
face_cascade_4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml");
# Cascade detection definitions
scale_reduction = 1.1;
min_accepted_neighbour_zones = 10;

def extract_aoi_features(detected_faces, grayscale_img, file, label):
    # For each detected face, generate the coordinates and dimensions of the face
    for x, y, height, width in detected_faces:
        # Crop the detected face (area of interest)
        area_of_interest = grayscale_img[y:(y+width), x:(x+height)];
        try:
            # Try to resize the area of interest to create uniform face sizes
            resized_aoi = cv2.resize(area_of_interest, (wroi, hroi));
            # Extract hog features from the area of interest
            list_of_features.append(hog.compute(resized_aoi, (64, 64)));
            # Create reference to the image
            list_of_images.append(file);
            # Add index of directory as a label
            list_of_labels.append(label);
        except:
            # Don't do anything, if resizing fails - skip the feature
            pass
    

def create_data_set(set_directory):
    # Print notifier
    print("Pre-processing data and extracting features...!");
    # Use OS Walk to get the name of each sub directory
    for root, directories, files in os.walk(set_directory):
        # Loop through sub directorys in main directory to get their names
        for directory_index, directory in enumerate(directories):
            # Create debugging trackers
            haar_failed = 0;
            # Create variable that tracks number of files in each directory
            FileNumberInDir = 0;
            # Create path name based on the current directory and the directories we are reading from
            path = "%s/%s"%(set_directory, directory);
            # Loop through all the files in that directory
            for file in glob.glob('%s/*'%path, recursive = True):
                # Increment file count
                FileNumberInDir += 1;
                # Locate the image from the list and read it using OpenCV
                grayscale_img = cv2.imread(file, 0);
                # Apply histogram equalisation to expose features
                grayscale_img = cv2.equalizeHist(grayscale_img);
                # Detect faces in image using Haar Cascade - function returns the definitions of the the detected rectangle in tuples
                faces1 = face_cascade_1.detectMultiScale(grayscale_img, scale_reduction, min_accepted_neighbour_zones, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
                faces3 = face_cascade_3.detectMultiScale(grayscale_img, scale_reduction, min_accepted_neighbour_zones, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
                faces4 = face_cascade_4.detectMultiScale(grayscale_img, scale_reduction, min_accepted_neighbour_zones, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
                # Check if we found a face, then use it to extract the corresponding features
                if len(faces1) == 1:
                    extract_aoi_features(faces1, grayscale_img, file, directory_index);
                elif len(faces3) == 1:
                    extract_aoi_features(faces3, grayscale_img, file, directory_index);
                elif len(faces4) == 1:
                    extract_aoi_features(faces4, grayscale_img, file, directory_index);
                else:
                    # We failed to find a face!
                    haar_failed += 1;
            # Print how many haar cascades we failed to find
            print("Failed Haar Cascades: " + str(haar_failed));
            # Print all the sub-directories of our  training set
            print(directory + " ( " + str(directory_index) + " ) : " + str(FileNumberInDir - haar_failed));


# # # CLASSIFICATION VARIABLES
    
def train_svm(training_features, training_labels):
    # Print notifier
    print("Training classifier - please wait!");
    # Initiate the classifier object from OpenCV
    svm = cv2.ml.SVM_create();
    # Specify a kernel
    svm.setKernel(cv2.ml.SVM_LINEAR);
    # Train the classifier using the training set
    svm.train(training_features, cv2.ml.ROW_SAMPLE, training_labels); 
    # Return the trained classifier
    return svm;

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap = plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Disclaimer: the plot function was not written by me, and is available at "http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html"
    """
    if normalize:
        # Normalize data
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis];
    
    # Create graph
    plt.imshow(cm, interpolation='nearest', cmap=cmap);
    plt.title(title);
    plt.colorbar();
    # Create labels for the values
    tick_marks = np.arange(len(classes));
    plt.xticks(tick_marks, classes, rotation=45);
    plt.yticks(tick_marks, classes);
    
    #Colour the graph based on how accurate the model was    
    fmt = '.2f' if normalize else 'd';
    thresh = cm.max() / 2.;
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black");
    
    # Label eaxis
    plt.tight_layout();
    plt.ylabel('True label');
    plt.xlabel('Predicted label');

def score_svm(svm, testing_features, testing_labels, plot_matrix):
    # Make prediction based on the testing features
    _, predicted_labels = svm.predict(testing_features);
    # Create a confusion matrix
    conf_matrix = metrics.confusion_matrix(testing_labels, predicted_labels);
    # Plot the confusion matrix
    if plot_matrix == True:
        plot_confusion_matrix(conf_matrix, emotion_classes, True, 'Confusion Matrix');
    # Score the classifier based on its accuracy
    return metrics.accuracy_score(testing_labels, predicted_labels) * 100;

# # # ACCURACY MEASUREMENT TECHNIQUES

def train_test_split():
    # Using sklearn function, divide the data into testing and training sets
    training_features, testing_features, training_labels, testing_lables = ms.train_test_split(list_of_features, list_of_labels, test_size=0.2, random_state = 36);
    # Create the svm
    svm = train_svm(training_features, training_labels);
    # Now that we have trained the SVM, make sure it can classify the training set
    print("Training Accuracy: " + str(score_svm(svm, training_features, training_labels, False)) + "%");
    # See how it performs on the test set
    print("Testing Accuracy: " + str(score_svm(svm, testing_features, testing_lables, True)) + "%");
    
    return svm;

def two_fold_cross_validation():
    # Using sklearn function, divide the data into two folds (50/50)
    fold1_features, fold2_features, fold1_labels, fold2_labels = ms.train_test_split(list_of_features, list_of_labels, test_size=0.5, random_state = 36);
    # Train the svm with the first fold
    svm_fold1 = train_svm(fold1_features, fold1_labels);
    # Now that we have trained the SVM, make sure it can classify the training set
    print("Fold 1 Accuracy: " + str(score_svm(svm_fold1, fold2_features, fold2_labels, False)) + "%");
    # Train the svm with the second fold
    svm_fold2 = train_svm(fold2_features, fold2_labels);
    # See how it performs on the test set
    print("Fold 2 Accuracy: " + str(score_svm(svm_fold2, fold1_features, fold1_labels, True)) + "%");
    
    return svm_fold1;

# # # MAIN APPLICATION

# Load our images into open cv by creating references to each image
create_data_set("Training_Set_Large");
# Convert lists into data types that are compatible with OpenCV
list_of_labels = np.int32(list_of_labels);
list_of_features = np.array(list_of_features, dtype = np.float32);
# Print the extracted matrices
print("Number of pictures: " + str(len(list_of_images)))
print("Number of Labels: " + str(len(list_of_labels)));
print("Feature Array: " + str(list_of_features.shape));
# Print information about the features
print(list_of_features.shape, list_of_labels.shape);
# Create classifier
my_svm = two_fold_cross_validation();
my_svm = train_test_split();

# # # INITIATE VIDEO CAPTURE DETECTION

'''
video_capture = cv2.VideoCapture(0);
shouldCapture = True;
if shouldCapture:
    # Start creating loop, using webcam image every frame
    while True:
        # Read method returns two elements, the last being the last frame captured by the camera - the _ allows us to ignore the first element
        _, frame = video_capture.read();
        # Convert the camera frame into a grayscale version
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces1 = face_cascade_1.detectMultiScale(grayscale_frame, scale_reduction, min_accepted_neighbour_zones);
        # Create dependancies for live detection
        features_extracted = False;
        current_frame_features = [];
        # Use histograme equalization on current frame
        aoi = cv2.equalizeHist(grayscale_frame);
        # Check for detected faces
        if len(faces1) == 1:
            # For each detected face, generate the coordinates and dimensions of the face
            for x, y, height, width in faces1:
                # Crop the detected face (area of interest) with approximate values for the faces
                area_of_interest = grayscale_frame[y:(y+width), x:(x+height)];
                # Draw rectangle on face area in the camera frame
                cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2);
                try:
                    # Try to resize the area of interest to create uniform face sizes
                    resized_aoi = cv2.resize(area_of_interest, (wroi, hroi));
                    # Extract hog features from the area of interest
                    current_frame_features.append(hog.compute(resized_aoi, (64, 64)));
                    current_frame_features = np.array(current_frame_features, dtype = np.float32);
                    # Indicate that we successfully extracted features
                    features_extracted = True;
                    break;
                except:
                    pass
        if features_extracted == True:
            # Use classifier to make a prediction
            _, predicted_label = my_svm.predict(current_frame_features);
            # Print the prediction on our frame
            display_text = "Emotion: " + str(predicted_label);
            font = cv2.FONT_HERSHEY_SIMPLEX;
            cv2.putText(frame, display_text, (10, 50), font, 1, (255,255,255), 2, cv2.LINE_AA);
        # Use open CV to display the final camera frame
        cv2.imshow('Video', frame);
        # implement way to discontinue the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Stop capturing video
    video_capture.release();
    # Destroy windows created by OpenCV
    cv2.destroyAllWindows();
'''