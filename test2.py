import cv2
import os
import numpy as np

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Set the width and height of the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Set the delay between frames (in milliseconds)
delay = 200

# Specify the directory of images
image_dir = r'russs'

# Get the list of all subfolders in the image directory
subfolders = [f.path for f in os.scandir(image_dir) if f.is_dir()]

# Initialize the index of the current subfolder and image
subfolder_index = 0
image_index = 0

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the Haar cascade classifier for license plate detection
plate_cascade = cv2.CascadeClassifier('haarcascade_license_plate_rus_16stages.xml')

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Load the next image from the current subfolder
    images = os.listdir(subfolders[subfolder_index])
    image_path = os.path.join(subfolders[subfolder_index], images[image_index])
    image = cv2.imread(image_path)

    # Resize the image and webcam frame to the same size
   
    resized_frame = cv2.resize(frame, (960, 540))

    # Detect faces and blur them
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        # Blur the face
        face_roi = resized_frame[y:y+h, x:x+w]
        face_roi = cv2.GaussianBlur(face_roi, (25, 25), 0)
        resized_frame[y:y+h, x:x+w] = face_roi

    # Detect license plates and blur them
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    if(len(plates)>0):
        print("plate"+str(len(plates))+image_path)
    for (x, y, w, h) in plates:
        # Blur the license plate
        plate_roi = image[y:y+h, x:x+w]
        plate_roi = cv2.GaussianBlur(plate_roi, (25, 25), 0)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 4)
        image[y:y+h, x:x+w] = plate_roi

    image = cv2.resize(image, (960, 540))
    # Combine the frame and image
    combined_frame = cv2.hconcat([resized_frame, image])
    # Add text to the final frame
    text = "Blurred faces and license plates"
    cv2.putText(combined_frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the combined frame
    cv2.imshow('Combined', combined_frame)

    # Increment the image index
    image_index += 1

    # If the end of the current subfolder is reached, move to the next subfolder
    if image_index == len(images):
        subfolder_index += 1
        image_index = 0

    # If the end of the subfolders is reached, start from the beginning
    if subfolder_index == len(subfolders):
        subfolder_index = 0

    # Exit on 'q' key press
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
