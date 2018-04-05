# Import libraries
import cv2
from matplotlib import pyplot as plt

# # # CREATE LISTS FOR OUR DATA

# Since some images get removed with Haar Cascade, a new lable array is created
list_of_labels = [];
# Create labels for each image (which are integers from 0 - 6)
list_of_images = [];
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

def extract_aoi_features(detected_faces, grayscale_img, file):
    # For each detected face, generate the coordinates and dimensions of the face
    for x, y, height, width in detected_faces:
        # Crop the detected face (area of interest)
        area_of_interest = grayscale_img[y:(y+width), x:(x+height)];
        list_of_images.append(area_of_interest);
        try:
            # Try to resize the area of interest to create uniform face sizes
            resized_aoi = cv2.resize(area_of_interest, (wroi, hroi));
            list_of_images.append(resized_aoi);
            # Create reference to the image
            list_of_images.append(file);
            print(len(list_of_images));
            # to hide tick values on X and Y axis
            plt.subplot(2, 2, 3)
            plt.imshow(list_of_images[1], cmap = 'gray', interpolation = 'bicubic');
            plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        except:
            # Don't do anything, if resizing fails - skip the feature
            pass
    

def create_data_set():
    # Print notifier
    print("Pre-processing data...");
    # Locate the image from the list and read it using OpenCV
    grayscale_img = cv2.imread("006_an_000_0024.jpg", 0);
    list_of_images.append(grayscale_img);
    # Apply histogram equalisation to expose features
    grayscale_img = cv2.equalizeHist(grayscale_img);
    list_of_images.append(grayscale_img);
    # Detect faces in image using Haar Cascade - function returns the definitions of the the detected rectangle in tuples
    faces1 = face_cascade_1.detectMultiScale(grayscale_img, scale_reduction, min_accepted_neighbour_zones, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    faces3 = face_cascade_3.detectMultiScale(grayscale_img, scale_reduction, min_accepted_neighbour_zones, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    faces4 = face_cascade_4.detectMultiScale(grayscale_img, scale_reduction, min_accepted_neighbour_zones, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    # Check if we found a face, then use it to extract the corresponding features
    if len(faces1) == 1:
        extract_aoi_features(faces1, grayscale_img, "006_an_000_0024.jpg");
    elif len(faces3) == 1:
        extract_aoi_features(faces3, grayscale_img, "006_an_000_0024.jpg");
    elif len(faces4) == 1:
        extract_aoi_features(faces4, grayscale_img, "006_an_000_0024.jpg");
        
create_data_set();