# Importing libraries
import cv2

# # # INITIALISE VARIABLES # # #

# Load cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Cascade detection definitions
scale_reduction = 1.3
min_accepted_neighbour_zones = 5

# # # DETECT FEATURES USING HAARCASCADE # # #

# Define functions for detections
def detect(grayscale_image, current_cam_frame):
    # Detect faces in rectangles- function returns the definitions of the the detected rectangle in tuples
    faces = face_cascade.detectMultiScale(grayscale_image, scale_reduction, min_accepted_neighbour_zones)
    # Create for loop that iterates through face tupples - x + y are the coordinates of the bottom left corner of the rectangle, w = width, h = height
    for (x_coord_f, y_coord_f, width_f, height_f) in faces:
        # Open CV rectangle draw function - get the current camera frame, find the bottom left coordintes, find bottom right coordinates by adding dimesions, define rgb colour of rectangle, final thickness of rectangle
        cv2.rectangle(current_cam_frame, (x_coord_f, y_coord_f), (x_coord_f+width_f, y_coord_f+height_f), (255, 0, 0), 2)
    # Return the frame with our function - it will have all rectangles applied to it
    return current_cam_frame
# Create capture class from Open CV - the integer that is passed, references the camera that is supposed to be used ('0' is native camera, any others are extra/external)
video_capture = cv2.VideoCapture(0)

# # # START LIVE DETECTION VIA WEBCAM # # #

# Repeate infinitely, until 'break' is called (function in python)
while True:
    # read method returns two elements, the last being the last frame captured by the camera - the _ allows us to ignore the first element
    _, frame = video_capture.read();
    # convert the camera frame into a grayscale version
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # add rectangles to get the final image
    final_img = detect(grayscale_frame, frame)
    # use open CV the final video
    cv2.imshow('Video', final_img)
    # implement way to discontinue the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 
video_capture.release()
# destroy window
cv2.destroyAllWindows()
