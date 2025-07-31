'''
import streamlit as st
import mysql.connector
from PIL import Image
import io
import cv2
import numpy as np

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='advait1234',
    database='drowsiness_db'
)
cursor = conn.cursor()

st.title("Drowsiness Detection Data")

cursor.execute("SELECT * FROM drowsiness")
data = cursor.fetchall()

for row in data:
    id_, date, timestamp, status, frame_encoded = row
    frame_decoded = np.frombuffer(frame_encoded, dtype=np.uint8)
    frame = cv2.imdecode(frame_decoded, 1) 
    
    st.write(f"### ID: {id_}")
    st.write(f"**Date:** {date}")
    st.write(f"**Timestamp:** {timestamp}")
    st.write(f"**Status:** {status}")
    
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(frame, channels="BGR", caption="Face Frame", use_column_width=True)
    with col2:
        st.write(f"### Status: {status}")
        if status == "SLEEPING":
            st.error("This person is sleeping!")
        elif status == "Drowsy":
            st.warning("This person is drowsy!")

conn.close()

'''




import streamlit as st
import mysql.connector
from PIL import Image
import io
import cv2
import numpy as np

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='advait1234',
    database='drowsiness_db'
)
cursor = conn.cursor()


st.title("Drowsiness Detection Data")


cursor.execute("SELECT * FROM drowsiness")
data = cursor.fetchall()


for row in data:
    id_, date, timestamp, status, frame_encoded = row
    frame_decoded = np.frombuffer(frame_encoded, dtype=np.uint8)
    frame = cv2.imdecode(frame_decoded, 1) 
    
    st.write(f"ID: {id_}")
    st.write(f"Date: {date}")
    st.write(f"Timestamp: {timestamp}")
    st.write(f"Status: {status}")

    
    st.image(frame, channels="BGR", caption="Face Frame", use_column_width=True)

conn.close()
