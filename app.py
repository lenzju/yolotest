import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("Simple Object Detection")

# Modell laden
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Bild hochladen
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file)

    # YOLO kann direkt mit PIL arbeiten
    results = model(image)

    # Bild mit Bounding Boxen
    result_image = results[0].plot()

    st.image(result_image, caption="Detected Objects", use_container_width=True)
