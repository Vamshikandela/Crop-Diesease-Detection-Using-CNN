import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(256,activation='relu'),

    Dense(15,activation='softmax')
])

model.load_weights("crop_weights.weights.h5")




with open("class_labels.json") as f:
    class_labels = json.load(f)

labels = {v:k for k,v in class_labels.items()}





### streamlit Ui
# Page Config
st.set_page_config(
    page_title="🌿 Crop Disease Detection",
    page_icon="🌱",
    layout="wide"
)

# Custom Theme
st.markdown("""
<style>

.main {
    background-color: #f4fff4;
}

.stApp {
    background: linear-gradient(to right, #e8f5e9, #f1f8e9);
}

h1 {
    color: #1b5e20 !important;
    text-align: center;
    font-size: 42px !important;
}

h2, h3 {
    color: #2e7d32 !important;
}

.stFileUploader {
    border: 2px dashed #4caf50;
    border-radius: 15px;
    padding: 20px;
    background-color: #ffffff;
}

.stSuccess {
    background-color: #c8e6c9 !important;
    color: #1b5e20 !important;
}

.stInfo {
    background-color: #e8f5e9 !important;
    color: #2e7d32 !important;
}

.prediction-box {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1> Crop Disease Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "### Upload a leaf image and detect crop diseases using CNN"
)

uploaded = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg","jpeg","png"]
)

if uploaded:
    img = Image.open(uploaded)
    img = img.resize((224,224))

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)

    pred = model.predict(img_array)

    disease = labels[np.argmax(pred)]

    
    prediction = model.predict(img_array)

    confidence = np.max(prediction) * 100

    st.success(f"Disease: {disease}")
    st.write(f"Confidence: {confidence:.2f}%")
    
    recommendations = {
    "Pepper__bell___Bacterial_spot":
        "Apply copper-based bactericides, remove infected leaves, avoid overhead irrigation, and use disease-free seeds.",

    "Pepper__bell___healthy":
        "Plant is healthy. Continue proper watering, balanced fertilization, and regular monitoring.",

    "Potato___Early_blight":
        "Use fungicide, improve crop rotation, remove infected leaves, and maintain proper plant spacing.",

    "Potato___Late_blight":
        "Apply recommended fungicides immediately, remove infected plants, avoid excessive moisture, and improve air circulation.",

    "Potato___healthy":
        "Plant is healthy. Maintain proper irrigation, nutrient management, and regular field inspection.",

    "Tomato_Bacterial_spot":
        "Apply copper-based bactericides, remove infected foliage, avoid working with wet plants, and ensure proper sanitation.",

    "Tomato_Early_blight":
        "Use fungicides, remove affected leaves, practice crop rotation, and provide adequate plant spacing.",

    "Tomato_Late_blight":
        "Apply fungicide and remove infected leaves. Improve airflow, avoid overhead watering, and destroy severely infected plants.",

    "Tomato_Leaf_Mold":
        "Reduce humidity, improve greenhouse ventilation, remove infected leaves, and apply suitable fungicides when necessary.",

    "Tomato_healthy":
        "Plant is healthy. Continue good agricultural practices, balanced fertilization, and regular monitoring."
}
    st.write(f"Recommendation:")
    st.info(recommendations.get(disease, "No recommendation available"))