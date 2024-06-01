import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

# Initialize video capture for the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Hand detector with a maximum of 2 hands
detector = HandDetector(maxHands=2)

# Constants for image processing and saving
offset = 20
imgSize = 300
folder = "Data/Y"
counter = 0

while True:
    # Capture frame from webcam
    success, img = cap.read()
    
    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        totalFingers = 0  # Initialize total finger count

        for hand in hands:
            # Get the bounding box for the hand
            x, y, w, h = hand['bbox']

            # Create a white canvas
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Crop the image around the bounding box
            imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

            # Get the aspect ratio and resize
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
                imgWhite[hGap : hCal + hGap, :] = imgResize

            # Get the count of extended fingers
            fingers = detector.fingersUp(hand)
            count = fingers.count(1)  # Count fingers that are extended
            totalFingers += count  # Add to the total finger count

            # Display total fingers on the webcam feed
            cv2.putText(
                imgWhite, 
                f"Fingers: {totalFingers}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 0), 
                2
            )

            # Display the cropped and white background images
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # Display the webcam output
    cv2.imshow("Image", img)

    # Capture image when 's' is pressed
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        # Save the current white background image
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print("Image saved:", f"{folder}/Image_{time.time()}.jpg")

    # Exit on 'q'
    if key == ord("q"):
        break

# Release webcam and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import math
# import time
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier

# # Initialize video capture for the webcam
# cap = cv2.VideoCapture(0)  # Default webcam

# # Hand detector with a maximum of 2 hands
# detector = HandDetector(maxHands=2)

# # Classifier for hand signs (optional)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# # Constants for image processing and saving
# offset = 20
# imgSize = 300
# folder = "Data/Y"
# counter = 0
# labels = ["A", "B", "C", "D", "E", "F", "G", "X", "Y"]  # Labels for classifier

# while True:
#     # Capture frame from webcam
#     success, img = cap.read()
    
#     # Detect hands
#     hands, img = detector.findHands(img)

#     if hands:
#         totalFingers = 0  # Initialize total finger count

#         for hand in hands:
#             # Get the bounding box for the hand
#             x, y, w, h = hand['bbox']

#             # Create a white canvas
#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#             # Crop the image around the bounding box
#             imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

#             # Get the aspect ratio and resize
#             aspectRatio = h / w
#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap:wCal + wGap] = imgResize
#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap:hCal + hGap, :] = imgResize

#             # Get the count of extended fingers
#             fingers = detector.fingersUp(hand)
#             count = fingers.count(1)  # Count fingers that are extended
#             totalFingers += count  # Add to the total finger count

#             # Draw a background box for the text
#             cv2.rectangle(img, (5, 10), (140, 40), (0, 0, 0), cv2.FILLED)
            
#             # Display total fingers on the webcam feed
#             cv2.putText(
#                 img, 
#                 f"Fingers: {totalFingers}", 
#                 (10, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 1, 
#                 (255, 255, 255), 
#                 2  # White text on a black box
#             )

#             # Display bounding box around hand
#             cv2.rectangle(img, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 2)

#             # If you have a classifier, you can get predictions and display labels
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             cv2.putText(
#                 img, 
#                 f"Class: {labels[index]}", 
#                 (x - offset, y - offset - 20), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 1, 
#                 (255, 255, 255), 
#                 2
#             )

#     # Display the webcam output with all features integrated
#     cv2.imshow("Webcam Feed", img)

#     # Capture image when 's' is pressed
#     key = cv2.waitKey(1)
#     if key == ord("s"):
#         counter += 1
#         # Save the current white background image
#         cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", img)
#         print("Image saved:", f"{folder}/Image_{time.time()}.jpg")

#     # Exit on 'q'
#     if key == ord("q"):
#         break

# # Release webcam and destroy OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

