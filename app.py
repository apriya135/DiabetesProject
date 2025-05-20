import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
import os

# Session state to control page flow
if 'page' not in st.session_state:
    st.session_state.page = 'instructions'

def show_instructions():
    st.title("Welcome to the Non-Invasive Diabetes Screening App")
    st.markdown("""
    ### Please read the instructions carefully:
    1. Upload or capture a clear **tongue image**.
    2. Upload a short **PPG video** (30 seconds, with finger over flashlight).
    3. Click **Predict** to see your result.
    4. You can also **download a report**.
    """)
    if st.button("Continue"):
        st.session_state.page = 'main'

def show_main_app():
    st.title("Diabetes Screening")

    name = st.text_input("Enter your name")
    age = st.text_input("Enter your age")

    tongue_image = st.file_uploader("Upload Tongue Image", type=["jpg", "png", "jpeg"])
    ppg_video = st.file_uploader("Upload PPG Video", type=["mp4", "avi", "mov"])

    if st.button("Predict"):
        if tongue_image and ppg_video:
            # Load model and scaler
            model = load_model("diabetes_classifier.keras")
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

            mobilenet = MobileNetV2(weights=None, include_top=False, pooling='avg', input_shape=(224, 224, 3))
            mobilenet.load_weights("mobilenetv2_weights.h5")

            # Process tongue image
            img = keras_image.load_img(tongue_image, target_size=(224, 224))
            img_array = preprocess_input(np.expand_dims(keras_image.img_to_array(img), axis=0))
            tongue_features = mobilenet.predict(img_array)

            # Process PPG video
            with open("temp_video.mp4", "wb") as f:
                f.write(ppg_video.read())
            cap = cv2.VideoCapture("temp_video.mp4")
            green_means = [np.mean(frame[:, :, 1]) for ret, frame in iter(lambda: cap.read(), (False, None)) if ret]
            cap.release()

            stats = np.array([
                np.mean(green_means), np.std(green_means),
                np.max(green_means), np.min(green_means),
                np.median(green_means),
                np.max(green_means) - np.min(green_means),
                (3 * (np.mean(green_means) - np.median(green_means))) / (np.std(green_means) or 1)
            ]).reshape(1, -1)

            combined = np.concatenate([tongue_features, stats], axis=1)
            scaled = scaler.transform(combined)
            pred = model.predict(scaled)[0][0]
            result = "Diabetic" if pred >= 0.5 else "Non-Diabetic"
            st.success(f"Prediction: {result}")
        else:
            st.warning("Please upload both image and video")

# Display pages
if st.session_state.page == 'instructions':
    show_instructions()
else:
    show_main_app()
