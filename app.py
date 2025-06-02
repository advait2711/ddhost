import streamlit as st
import cv2
import numpy as np
import dlib
from imutils import face_utils
import tempfile
import pygame

# Load Dlib's shape predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Initialize Pygame mixer
pygame.mixer.init()

# Alarm flag
alarm_playing = False

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif ratio > 0.21:
        return 1
    else:
        return 0

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        pygame.mixer.music.load("mixkit-alert-alarm-1005.wav")
        pygame.mixer.music.play(-1)
        alarm_playing = True

def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False

def main():
    st.title("Drowsiness Detection System")
    st.markdown("Detects whether the person is Active, Drowsy or Sleeping using a webcam")

    run = st.checkbox('Start Webcam')

    FRAME_WINDOW = st.image([])
    status_text = st.empty()

    sleep = 0
    drowsy = 0
    active = 0

    cap = None

    if run:
        cap = cv2.VideoCapture(0)

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status_text.markdown("### 😴 Sleeping !!!")
                    play_alarm()
            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status_text.markdown("### 😵 Drowsy !")
                    play_alarm()
            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status_text.markdown("### 😀 Active")
                    stop_alarm()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    if cap:
        cap.release()
        stop_alarm()

if __name__ == "__main__":
    main()
