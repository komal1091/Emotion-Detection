
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
import os


# Load the pre-trained emotion detection model
model = load_model("/home/codezeros/Documents/Emotion Detection/emotion_model.h5")

# Load the video
video_path = "/home/codezeros/Documents/Emotion Detection/5 Emotional Ads that You will LOVE _ WHY & WHAT (online-video-cutter.com).mp4"
cap = cv2.VideoCapture(video_path)

# Load the Haar Cascades classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the emotion labels
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

save_dir = "/home/codezeros/Documents/Emotion Detection/output/Frame"
frame_count = 0  # Counter for frame filenames


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region and resize for emotion detection
        face_region = gray[y:y+h, x:x+w]
        face_region_resized = cv2.resize(face_region, (48, 48))

        # Preprocess the face region for prediction
        face_region_resized = np.expand_dims(face_region_resized, axis=0)
        face_region_resized = np.expand_dims(face_region_resized, axis=-1)
        face_region_resized = face_region_resized.astype('float32') / 255  

        # Perform emotion prediction
        result = model.predict(face_region_resized)
        predicted_emotion_index = np.argmax(result)
        predicted_emotion_label = label_dict[predicted_emotion_index]

        # Add predicted emotion label text
        cv2.putText(frame, predicted_emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Save the frame with detected face and emotion label
        frame_filename = f"{save_dir}/frame_{frame_count}_{predicted_emotion_label}.jpg"
        cv2.imwrite(frame_filename, frame)
        frame_count += 1  # Increment frame counter

        # out.write(frame)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
