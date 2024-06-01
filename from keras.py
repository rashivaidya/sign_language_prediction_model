# from keras.models import load_model
# import cv2
# import numpy as np

# # Load the model
# model = load_model("keras_Model.h5", compile=False)

# # Load the labels
# class_names = open("labels.txt", "r").readlines()

# # CAMERA can be 0 or 1 based on the default camera of your computer
# camera = cv2.VideoCapture(0)

# # Confidence threshold for saving images
# confidence_threshold = 0.8

# # Folder to save images
# save_folder = "Data/"

# while True:
#     # Grab the web camera's image.
#     ret, image = camera.read()

#     # Resize the raw image into (224-height,224-width) pixels
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

#     # Show the image in a window
#     cv2.imshow("Webcam Image", image)

#     # Make the image a numpy array and reshape it to the model's input shape.
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

#     # Normalize the image array
#     image = (image / 127.5) - 1

#     # Predicts the model
#     prediction = model.predict(image)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]

#     # Print prediction and confidence score
#     print("Class:", class_name[2:], end="")
#     print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

#     # Check if the predicted class is above the confidence threshold
#     if confidence_score > confidence_threshold:
#         # Save the image with the predicted class name
#         image_name = f"{save_folder}/{class_name[2:]}_{confidence_score:.2f}.jpg"
#         cv2.imwrite(image_name, image[0])
#         print(f"Image saved as: {image_name}")

#     # Listen to the keyboard for presses.
#     keyboard_input = cv2.waitKey(1)

#     # 27 is the ASCII for the esc key on your keyboard.
#     if keyboard_input == 27:
#         break

# camera.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np
import math
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tensorflow as tf
print(tf.__version__)
import keras
print(keras.__version__)
# Initialize video capture
cap = cv2.VideoCapture(0)

# Hand detector with a maximum of 1 hand to detect
detector = HandDetector(maxHands=1)

# Load classifier with the model and label files
classifier = Classifier("C:/Users/91797/Downloads/HandSignDetection-master/Model/keras_model.h5", "C:/Users/91797/Downloads/HandSignDetection-master/Model/labels.txt")

# Constants
offset = 20
imgSize = 300
folder = "Data/"  # Base folder for storing images
counter = 0  # Image counter

# Define labels
labels = ["A", "B", "C", "D", "E", "F", "G", "X", "Y"]

while True:
    # Capture webcam frame
    success, img = cap.read()
    imgOutput = img.copy()

    # Detect hands
    hands, img = detector.findHands(img)
    
    if hands:
        # Extract the bounding box of the first hand
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop the image around the bounding box
        imgCrop = img[y - offset: y + h + offset, x - offset:x + w + offset]

        # Get the shape of the cropped image
        imgCropShape = imgCrop.shape

        # Adjust aspect ratio and resize
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Classifier prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Display the predicted label on the webcam output
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Display the cropped image and the white background
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        # Save images when 's' is pressed
        key = cv2.waitKey(1)
        if key == ord("s"):
            class_folder = folder + labels[index]  # Determine the folder based on the predicted class
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            # Save the image with a unique name based on the class and counter
            img_path = f"{class_folder}/{labels[index]}_{counter}.jpg"
            cv2.imwrite(img_path, imgWhite)  # Save the white background image
            print(f"Image saved: {img_path}")
            counter += 1  # Increment the image counter

    # Display the webcam output
    cv2.imshow("Image", imgOutput)
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
