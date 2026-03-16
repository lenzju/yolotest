import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import requests
import os

st.title("Fashion Object Detection (YOLO + Streamlit)")

MODEL_URL = "https://huggingface.co/keremberke/yolov8m-clothing-detection/resolve/main/best.pt"
MODEL_PATH = "fashion_model.pt"

# Modell herunterladen falls nicht vorhanden
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# YOLO Modell laden
model = YOLO(MODEL_PATH)

uploaded_file = st.file_uploader("Upload clothing image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    results = model(img)

    annotated = results[0].plot()

    st.image(annotated, caption="Detected Clothing", use_column_width=True)
