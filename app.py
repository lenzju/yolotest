import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Sachen Finden", layout="centered")

# CSS Styling
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
h1 {
    text-align: center;
    color: #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

st.title("Detection AI")
st.caption("Upload ein Bild und erkenne alles automatisch")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

with st.container():
    file = st.file_uploader("📤 Bild hochladen", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file)

    with st.spinner("🔍 Analysiere Bild..."):
        results = model(image)

    result = results[0].plot()

    st.image(result, caption="Ergebnis", use_container_width=True)
    st.success("Fertig!")
