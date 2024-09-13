import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model (ensure 'pneumonia_model.h5' is in the same directory)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('pneumonia_model.h5')
    return model

model = load_model()

# Function to make predictions on uploaded images
def predict_pneumonia(image):
    # Resize and preprocess the image
    size = (150, 150)  # Assuming input size of the model is 150x150
    image = ImageOps.fit(image, size, Image.ANTIALIAS)  # Resize the image to match the input size
    img_array = np.asarray(image)                       # Convert image to numpy array
    img_array = img_array / 255.0                       # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)       # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    return "Pneumonia" if prediction[0][0] > 0.5 else "Not Pneumonia"

# Streamlit app layout
st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to predict if it shows signs of pneumonia.")

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Make prediction on the image
    st.write("Classifying...")
    prediction = predict_pneumonia(image)

    # Display the prediction
    st.write(f"Prediction: **{prediction}**")
else:
    st.write("Please upload an image to classify.")
