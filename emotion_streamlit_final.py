import streamlit as st
import cv2
import os
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import time
import numpy as np
from tensorflow.keras.models import load_model

st.title('Emotion Detection App')

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    model_emotion = load_model("/home/codezeros/Pictures/Emotion Detection/emotion_model.h5")

    frames = []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'MP4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            face_region_resized = cv2.resize(face_region, (48, 48))
            face_region_resized = np.expand_dims(face_region_resized, axis=0)
            face_region_resized = np.expand_dims(face_region_resized, axis=-1)
            face_region_resized = face_region_resized.astype('float32') / 255

            result = model_emotion.predict(face_region_resized)
            predicted_emotion_index = np.argmax(result)
            predicted_emotion_label = label_dict[predicted_emotion_index]

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Display predicted emotion label text
            cv2.putText(frame, predicted_emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        out.write(frame)
        frames.append(frame)

    cap.release()
    out.release()


def convert_to_mp4(input_file, output_file):
    video = VideoFileClip(input_file)
    video.write_videofile(output_file, codec='libx264', audio_codec='aac')
    video.close()

uploads_dir = 'uploads'
os.makedirs(uploads_dir, exist_ok=True)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file is not None:
    video_path = os.path.join(uploads_dir, uploaded_file.name)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if st.button("Process Video"):
        start_time = time.time()
        output_video_path = os.path.join(uploads_dir, "output_video.mp4")
        process_video(tfile.name, output_video_path)
        st.markdown("---")
        st.subheader("Output Video")

        output_video_path_conv = os.path.join(uploads_dir, "new_output_video.mp4")
        convert_to_mp4(output_video_path, output_video_path_conv)
        end_time = time.time()
        print("Total time consumed:", end_time-start_time)

        st.video(output_video_path_conv)
        st.session_state['video_bytes'] = open(output_video_path, 'rb').read()
        st.write(f"Output video saved at: {output_video_path}")

    if 'video_bytes' in st.session_state:
        st.markdown("---")
        st.download_button(label="Download output video",
                            data=st.session_state['video_bytes'],
                            file_name="output_video.mp4",
                            mime="video/mp4")





