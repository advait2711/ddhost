import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import time

# Streamlit config
st.set_page_config(page_title="Drowsiness Detection", layout="centered")

st.title("🛡️ Drowsiness Detection with MediaPipe")
FRAME_THRESHOLD = 6
EAR_THRESHOLD = 0.25
sleep_counter = 0

# Load alarm sound
def play_alarm():
    with open("mixkit-alert-alarm-1005.wav", "rb") as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav", start_time=0)

# EAR calculation
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, indices):
    left = landmarks[indices[0]]
    right = landmarks[indices[3]]
    top = (euclidean_dist(landmarks[indices[1]], landmarks[indices[5]]) +
           euclidean_dist(landmarks[indices[2]], landmarks[indices[4]])) / 2.0
    horizontal = euclidean_dist(left, right)
    return top / horizontal if horizontal != 0 else 0

# Eye landmark indices (MediaPipe)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   refine_landmarks=True, min_detection_confidence=0.5)

# Start webcam
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])
status_placeholder = st.empty()

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Failed to capture video")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    status = "✅ Active"
    color = (0, 255, 0)

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

        left_ear = calculate_ear(landmarks, LEFT_EYE_IDX)
        right_ear = calculate_ear(landmarks, RIGHT_EYE_IDX)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            sleep_counter += 1
            if sleep_counter >= FRAME_THRESHOLD:
                status = "🛑 Drowsy"
                color = (0, 0, 255)
                play_alarm()
            else:
                status = "⚠️ Blinking"
                color = (0, 165, 255)
        else:
            sleep_counter = 0

        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            cv2.circle(frame, landmarks[idx], 2, color, -1)

    cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    status_placeholder.markdown(f"### Status: {status}")

cap.release()
cv2.destroyAllWindows()
