# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('aphid_detection_model.h5')

# Set up the Streamlit app
st.title("Cotton Plant Aphid Detection")

st.write("Upload an image of a cotton plant leaf to check if it has aphids.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    return "Aphids Detected" if prediction[0] < 0.5 else "Healthy"

# Process and display prediction
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    label = predict_image(uploaded_file)
    st.write(f"Prediction: **{label}**")
