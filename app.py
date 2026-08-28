"""
Garbage Classification Dashboard
A Streamlit app that loads a trained CNN model and classifies
an uploaded waste image into one of 6 categories.
"""

import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------- Page Config ----------
st.set_page_config(
    page_title="Garbage Classifier",
    page_icon="♻️",
    layout="centered",
)

# ---------- Constants ----------
IMG_SIZE = (150, 150)
MODEL_PATH = "model/garbage_classification_cnn.h5"
CLASS_INDICES_PATH = "model/class_indices.json"

# ---------- Load Model & Class Labels (cached so it only loads once) ----------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    # Reverse mapping: {0: "cardboard", 1: "glass", ...}
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


model, idx_to_class = load_artifacts()

# ---------- Header ----------
st.title("♻️ Garbage Classification Dashboard")
st.write(
    "Upload an image of a waste item and the CNN model will predict "
    "which recycling category it belongs to: **cardboard, glass, metal, "
    "paper, plastic, or trash**."
)

st.divider()

# ---------- Image Upload ----------
uploaded_file = st.file_uploader(
    "Upload a waste image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------- Preprocess ----------
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---------- Predict ----------
    with st.spinner("Classifying..."):
        predictions = model.predict(img_array)[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_class = idx_to_class[predicted_idx]
    confidence = float(np.max(predictions)) * 100

    # ---------- Show Result ----------
    st.divider()
    st.subheader("Prediction Result")
    st.success(f"**Predicted Class:** {predicted_class.upper()}")
    st.metric("Confidence", f"{confidence:.2f}%")

    # ---------- Confidence Bar Chart for All Classes ----------
    st.subheader("Confidence per Class")
    prob_dict = {
        idx_to_class[i]: float(predictions[i]) for i in range(len(predictions))
    }
    # Sort descending for a cleaner chart
    prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    st.bar_chart(prob_dict)

else:
    st.info("👆 Upload an image to get started.")

# ---------- Footer ----------
st.divider()
st.caption(
    "Model: Custom CNN (Transfer Learning with MobileNetV2) | "
    "Trained on Kaggle Garbage Classification dataset (6 classes)"
)