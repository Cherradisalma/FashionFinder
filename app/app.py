import sys
sys.path.append("src")
import streamlit as st
from PIL import Image
from predict import predict

st.title("Détection de vêtements femme")
file = st.file_uploader("Choisissez une photo", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)
    label, conf = predict(img)
    st.success(f"Catégorie : **{label}** ({conf:.1%})")
