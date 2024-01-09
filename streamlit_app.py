import numpy as np
import cv2
import tensorflow as tf
import streamlit as st

# for better result use highe quality camera which enough light at place.

model = tf.keras.models.load_model('emotion_detection_model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Function to perform emotion detection on an image
def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion_text = emotion_dict[maxindex]
        cv2.putText(image, emotion_text, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

    return image

# Streamlit application
def main():
    st.title("Emotion Detection App")

    # Button to start the app
    start_button = st.button("Start")

    if start_button:
        # Open webcam
        cap = cv2.VideoCapture(0)

        # Display the webcam feed using st.video
        video_placeholder = st.empty()
        stop_button = st.button("Stop")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform emotion detection on each frame
            emotion_image = detect_emotion(frame)

            # Update the displayed video feed
            video_placeholder.image(emotion_image, channels="BGR", use_column_width=True)

            # Break the loop if the user clicks the "Stop" button
            if stop_button:
                cap.release()
                video_placeholder.empty()
                st.stop()

if __name__ == "__main__":
    main()
