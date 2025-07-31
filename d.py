import mysql.connector
import datetime
import cv2
import numpy as np
import dlib
from imutils import face_utils
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pygame
import base64
import datetime


conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='advait1234',
    database='drowsiness_db'
)
cursor = conn.cursor()


cursor.execute('''CREATE TABLE IF NOT EXISTS drowsiness (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE,
                    timestamp TIME,
                    status VARCHAR(50),
                    frame MEDIUMBLOB
                )''')
conn.commit()

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

alarm_playing = False

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

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


def store_data(status, frame):
    date = datetime.date.today().strftime('%Y-%m-%d')
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    
    _, encoded_frame = cv2.imencode('.png', frame)
    
    print(f"Encoded frame type: {type(encoded_frame)}")  
    print(f"Encoded frame size: {len(encoded_frame)}")  
    
    try:
        cursor.execute("INSERT INTO drowsiness (date, timestamp, status, frame) VALUES (%s, %s, %s, %s)",
                       (date, timestamp, status, encoded_frame.tobytes()))
        conn.commit()
        print("Data stored successfully!")
    except mysql.connector.Error as err:
        print(f"Error: {err}")



def update_frame():
    global sleep, drowsy, active
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame from the camera.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status_var.set("SLEEPING !!!")
                status_label.config(foreground="red")
                play_alarm()
                store_data("SLEEPING", face_frame)

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status_var.set("Drowsy !")
                status_label.config(foreground="blue")
                play_alarm()
                store_data("Drowsy", face_frame)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status_var.set("Active")
                status_label.config(foreground="green")
                stop_alarm()
                

        update_image(face_frame)

    root.after(10, update_frame)

def update_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    panel.img = img
    panel.config(image=img)

root = tk.Tk()
root.title("Drowsiness Detection")

status_var = tk.StringVar()
status_label = ttk.Label(root, textvariable=status_var, font=("Helvetica", 16))
status_label.pack(pady=10)

panel = ttk.Label(root)
panel.pack()


pygame.mixer.init()

update_frame()

root.protocol("WM_DELETE_WINDOW", lambda: root.destroy())
root.mainloop()

cap.release()
conn.close()