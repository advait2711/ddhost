import streamlit as st
import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
from PIL import Image
import os
import urllib.request

# Constants
MODEL_URL = "https://raw.githubusercontent.com/advait2711/ddhost/main/shape_predictor_68_face_landmarks.dat"
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
ALARM_SOUND = "mixkit-alert-alarm-1005.wav"

# Download the model file if not present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading face landmarks model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# Initialize pygame mixer
pygame.mixer.init()

# Functions
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(ALARM_SOUND)
        pygame.mixer.music.play(-1)

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# Streamlit UI
st.title("🛌 Drowsiness Detection System")
frame_window = st.image([])
status_text = st.empty()

run = st.checkbox('Start Camera')

cap = None
sleep = drowsy = active = 0

if run:
    cap = cv2.VideoCapture(0)

while run and cap is not None:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera read error!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = active = 0
            if sleep > 6:
                status_text.markdown("### 🔴 SLEEPING !!!")
                play_alarm()
        elif left_blink == 1 or right_blink == 1:
            drowsy += 1
            sleep = active = 0
            if drowsy > 6:
                status_text.markdown("### 🟡 Drowsy !")
                play_alarm()
        else:
            active += 1
            sleep = drowsy = 0
            if active > 6:
                status_text.markdown("### 🟢 Active")
                stop_alarm()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(img_rgb)

if not run and cap is not None:
    cap.release()
    stop_alarm()
