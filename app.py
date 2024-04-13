# app.py
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow
from tensorflow import keras
from keras.models import load_model

#Load the model
model = load_model('model.h5')

st.title("Sign Language Recognition")

# Start the webcam
cap = cv2.VideoCapture(0)

# Get the expected input shape of the model
input_shape = model.input_shape[1:]

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Preprocess the image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(frame)
    img = img.resize(input_shape[:2])  # Resize the image to match the input shape
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    # Reshape the input data to match the expected input shape
    img = img.reshape((1, *input_shape))

    # Make a prediction
    prediction = model.predict(img)

    # Display the prediction
    predicted_class = np.argmax(prediction)
    st.write(f'Prediction: {predicted_class}')

    # Display the image
    st.image(frame, channels="GRAY")

cap.release()