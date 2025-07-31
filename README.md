# Drowsiness Detection System

This project implements a Drowsiness Detection System using Python. It leverages OpenCV and dlib for facial landmark detection, with data stored in a MySQL database. The system monitors drowsiness in real-time using a webcam and provides visual feedback through a Streamlit application.

## Features

- Real-time detection of drowsiness and sleep status
- Storage of detection data (date, time, status, and frame) in a MySQL database
- Streamlit app to display captured frames and statuses

## Requirements

- Python 3.x
- OpenCV
- dlib
- NumPy
- mysql-connector-python
- Pillow
- Pygame
- imutils
- Streamlit

## Drowsiness Detection
Run the detection script:
   - python d.py

## Data Visualization
Run the Streamlit application:
   - streamlit run ddata.py
