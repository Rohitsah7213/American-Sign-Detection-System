import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model # type: ignore

# Initialize webcam capture
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load the model using tensorflow.keras
model_path = "C:\\Users\\rohit\\Desktop\\Project\\my_model.keras"
labels_path = "C:\\Users\\rohit\\Desktop\\Project\\converted_keras\\labels.txt"

# Load the model
model = load_model(model_path)

# Optionally, read the labels from the file
with open(labels_path, "r") as file:
    labels = file.read().splitlines()

offset = 20
imgSize = 300
counter = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white canvas for resizing the cropped hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        aspectRatio = h / w

        # Resize the cropped image to fit within imgWhite
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Directly show the resized or cropped image first
        # Ensure it is in the correct type
        imgResize = np.uint8(imgResize)  # Ensure it's in uint8 format

        # Check dimensions before displaying
        print(f"imgResize shape: {imgResize.shape}, dtype: {imgResize.dtype}")

        if imgResize.shape[0] > 0 and imgResize.shape[1] > 0:
            cv2.imshow('Resized Image', imgResize)  # Display resized image for debugging

        # Expand dimensions for the model prediction
        imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension
        prediction = model.predict(imgWhite)  # Predict using your model

        # Get the index of the highest predicted value
        index = np.argmax(prediction)

        # Ensure the index is within the bounds of the label list
        if 0 <= index < len(labels):
            predicted_label = labels[index]  # Get the corresponding label
        else:
            print(f"Index {index} is out of range!")
            predicted_label = "sorry"

        # Display the prediction on the image
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, predicted_label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

    # Show the main image with the prediction
    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
